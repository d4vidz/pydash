from r2a.ir2a import IR2A
from base.message import SSMessage
from player.parser import *
import time
from statistics import mean
from scipy.spatial import KDTree
import numpy as np

class R2A_QKNN(IR2A):
    """
    A Rate Adaptation Algorithm that uses average throughput to select video quality.
    This algorithm implements a simple throughput-based approach where it selects
    the highest quality level that is below half of the average measured throughput.
    Attributes:
        throughputs (list): List of measured throughput values
        request_time (float): Timestamp of the last request
        qi (list): List of available quality levels
    Methods:
        handle_xml_request: Handles MPD XML requests
        handle_xml_response: Processes MPD XML responses and calculates throughput
        handle_segment_size_request: Selects quality level based on average throughput
        handle_segment_size_response: Measures throughput for video segments
        initialize: Initializes the algorithm
        finalization: Performs cleanup operations
    """

    def __init__(self, id):
        """
        Initialize the R2A_QKNN algorithm.

        This class implements a QoE-driven KNN-based rate adaptation algorithm. It inherits from IR2A
        and initializes the necessary data structures for tracking throughput measurements and
        quality decisions.

        Parameters
        ----------
        id : str
            The identifier for this R2A instance.

        Attributes
        ----------
        throughputs : list
            Stores historical throughput measurements.
        request_time : float
            Tracks the timestamp of the last request.
        qi : list
            Stores the quality index decisions history.
        knn : KNeighborsRegressor
            The K-Nearest Neighbors regressor model used for quality prediction.
        replay_buffer : list
            Stores experience tuples (state, action, reward, next_state) for training.
        fitted : bool
            Indicates whether the KNN model has been fitted with data.
        """
        IR2A.__init__(self, id)
        self.throughputs = [] # store throughput values
        self.request_time = 0 # store request time
        self.qi = [] # store quality indices
        self.buffer_size = [] # store buffer size
        self.last_quality = None # last quality level selected
        # self.knn = KNeighborsRegressor(n_neighbors=3)
        self.replay_buffer = [] # store (state, action, reward, next_state)
        self.fitted = False # whether the model has been fitted
        self.epsilon = 0.1 # exploration rate
        self.gamma = 0.9  # discount factor
        self.k = 3  # number of neighbors
        self.tree : KDTree = None # KDTree for nearest neighbors
        self.X = []  # states+actions
        self.y = []  # rewards

    def handle_xml_request(self, msg: SSMessage):
        """
        Handles XML requests by recording the request time and sending the message downwards in the protocol stack.

        The request time is stored using a high-resolution performance counter and the message is 
        forwarded to the lower layer using send_down method from the parent class.

        Args:
            msg (SSMessage): The message object containing the XML request data.
                SSMessage is defined in base/messages/message.py

        Returns:
            None

        Note:
            This method is part of the request-response cycle in the QoE-driven adaptation algorithm.
            It overrides the abstract method from IR2AAlgorithm base class.
        """
        self.request_time = time.perf_counter() # record request time
        self.send_down(msg) # forward message down

    def handle_xml_response(self, msg: SSMessage):
        """
        Handles the XML response from the server containing the MPD (Media Presentation Description).
        This method is called when an XML response is received from the server. It parses the MPD,
        extracts the quality indices (QI), calculates and stores the throughput based on the response
        time, and forwards the message up the stack.
        Parameters
        ----------
        msg : SSMessage
            The message containing the MPD XML response from the server.
            Contains the payload and metadata about the response.
        Notes
        -----
        - Updates self.qi with available quality indices from parsed MPD
        - Calculates and stores throughput based on response size and time
        - Forwards original message up the protocol stack
        - Uses self.request_time which should be set when request was made
        See Also
        --------
        handle_segment_size_request : Related method for handling segment size requests
        """
        parsed_mpd = parse_mpd(msg.get_payload())
        self.qi = parsed_mpd.get_qi()

        t = time.perf_counter() - self.request_time
        self.throughputs.append(msg.get_bit_length() / t)

        self.send_up(msg)

    def get_state(self):
        """
        Returns the current state representation for the RL agent.

        The state consists of:
        - Average throughput of last 5 segments
        - Current buffer level
        - Last selected quality level

        Returns
        -------
        list
            [avg_throughput, buffer_level, last_quality]
        """
        avg_throughput = mean(self.throughputs[-5:]) if len(self.throughputs) >= 5 else 0
        buffer_level = mean(self.buffer_size) if self.buffer_size else 0
        last_quality = self.last_quality if self.last_quality is not None else 0
        return [avg_throughput, buffer_level, last_quality]

    def calculate_reward(self, quality, prev_quality, download_time, segment_size):
        """
        Calculates the reward based on QoE metrics.

        Parameters
        ----------
        quality : int
            Current quality level
        prev_quality : int
            Previous quality level
        download_time : float
            Time taken to download segment
        segment_size : int
            Size of segment in bits

        Returns
        -------
        float
            Reward value combining bitrate, smoothness and rebuffering penalties
        """
        bitrate_reward = quality
        smoothness_penalty = abs(quality - prev_quality) if prev_quality is not None else 0
        rebuffer_penalty = max(0, download_time - self.buffer_size[-1] if self.buffer_size else 0)
        return bitrate_reward - smoothness_penalty - 4.3 * rebuffer_penalty

    def select_action(self):
        """
        Selects the next quality level using epsilon-greedy policy and KDTree predictions.
        
        Uses KDTree to find nearest neighbors and predict Q-values for each possible action,
        with random exploration based on epsilon parameter.

        Returns
        -------
        int
            Selected quality level index
        """        
        # Epsilon-greedy exploration
        if np.random.random() < self.epsilon:
            return np.random.choice(self.qi)
        
        # Get current state
        state = self.get_state()
        
        # Find best action using KDTree predictions
        best_q = float('-inf')
        best_action = self.qi[0]
        
        # Predict Q-value for each possible quality level
        for action in self.qi:
            state_action = state + [action]
            q_val = self.predict(state_action) if self.fitted else 0
            
            if q_val > best_q:
                best_q = q_val
                best_action = action
                
        return best_action
    
    def update_replay(self, state, action, reward, next_state):
        """
        Updates the replay buffer with a new transition.

        Parameters
        ----------
        state : list
            Current state
        action : int
            Selected quality level
        reward : float
            Calculated reward
        next_state : list
            Resulting state
        """
        # Q-learning update
        if len(self.replay_buffer) > 1000:  # Limit buffer size
            self.replay_buffer.pop(0)
        self.replay_buffer.append((state, action, reward, next_state))
        
    def fit_knn(self):
        """
        Fits KNN model using scipy's cKDTree for efficient nearest neighbor search.
        Transforms replay buffer into X (state+action) and y (rewards) arrays.
        """
        if len(self.replay_buffer) > 0:
            self.X = []
            self.y = []
            for (s, a, r, _) in self.replay_buffer:
                self.X.append(s + [a])
                self.y.append(r)
            self.tree = KDTree(np.array(self.X))
            self.fitted = True

    def predict(self, state_action : list):
        """
        Predicts Q-value for a state-action pair using K nearest neighbors.
        
        Parameters
        ----------
        state_action : list
            Combined state and action vector
            
        Returns
        -------
        float
            Predicted Q-value based on nearest neighbors
        """
        if not self.fitted:
            return 0
            
        distances, indices = self.tree.query(
            np.array(state_action).reshape(1, -1),
            k=min(self.k, len(self.y))
        )
        
        if len(indices) == 0:
            return 0
            
        # Convert indices to proper format
        indices = indices.flatten()
        distances = distances.flatten()
        
        # Get rewards for neighbors
        neighbor_rewards = np.array([self.y[int(i)] for i in indices])
        
        # Calculate weights and return weighted average
        weights = 1.0 / (distances + 1e-6)
        return float(np.average(neighbor_rewards, weights=weights))

    def handle_segment_size_request(self, msg: SSMessage):
        """
        Handles segment size requests by selecting quality levels using KNN-based prediction.

        This method implements the segment size request handling for the QKNN R2A algorithm.
        It uses a trained KNN model to predict the optimal quality level based on the current
        state (throughput history). The selection process involves:
        1. Recording the request timestamp
        2. Using select_action() to determine the best quality level via KNN predictions
        3. Adding the selected quality to the message and forwarding it

        Parameters
        ----------
        msg : SSMessage
            A segment size message object containing the request details
            and methods to modify the quality selection.

        Returns
        -------
        None
            The method sends the modified message down but does not return any value

        Notes
        -----
        - Records request_time for throughput calculations in handle_segment_size_response
        - Uses KNN model predictions to make quality decisions
        - Quality selection considers historical performance through the replay buffer
        - The modified message is sent downstream using send_down()
        """
        self.request_time = time.perf_counter()

        chosen_qi = self.select_action()

        msg.add_quality_id(chosen_qi)
        self.send_down(msg)

    def handle_segment_size_response(self, msg: SSMessage):
        """
        
                Handles the segment size response from the server and updates the reinforcement learning model.
        1. Records download time for the received segment
        2. Gets current state by calling get_state() method
        3. Calculates reward using calculate_reward() method based on:
            - Current quality level
            - Previous quality level 
            - Download time
            - Segment size (bit length)
        4. Updates throughput measurements list
        5. Stores current quality as last_quality for next iteration
        6. Gets next state after download
        7. Updates replay memory with (state, action, reward, next_state) tuple
        8. Retrains KNN model by calling fit_knn()
        9. Forwards segment message up the pipeline
             The segment size message containing the video segment data including:
             - Quality ID
             - Bit length
             - Other segment metadata
        - Download time is measured using time.perf_counter()
        - Throughput is calculated as: segment_bit_length / download_time 
        - Quality ID represents the selected bitrate level
        - The replay memory stores state transitions for reinforcement learning
        - KNN model is retrained after each segment to adapt to network conditions
        - Part of Q-learning based adaptive bitrate selection system

        Parameters
        ----------
        msg : SSMessage
            The segment size message containing the video segment data and metadata

        Returns
        -------
        None
            The method sends the message up but does not return any value

        Notes
        -----
        - Throughput is calculated as: segment_size / elapsed_time
        - The throughput values are stored in self.throughputs for later use
        - This is part of the feedback loop for adaptive bitrate selection
        """
        download_time = time.perf_counter() - self.request_time
        current_quality = msg.get_quality_id()
        
        state = self.get_state()
        reward = self.calculate_reward(
            current_quality,
            self.last_quality,
            download_time,
            msg.get_bit_length()
        )
        
        self.throughputs.append(msg.get_bit_length() / download_time)
        self.last_quality = current_quality
        
        next_state = self.get_state()
        self.update_replay(state, current_quality, reward, next_state)
        self.fit_knn()
        
        self.send_up(msg)

    def initialize(self):
        """
        Initialize all necessary attributes for the R2A algorithm implementation.

        This method is called once during the simulation setup, before any iteration with the
        manifest manager is performed. Attributes initialized here must support the implementation
        of the throughput adaptation algorithm. In this implementation, it performs no specific tasks
        but maintains compliance with the abstract base R2AAlgorithm class interface.

        Raises:
            None

        Returns:
            None
        """
        pass

    def finalization(self):
        """
        Performs finalization/cleanup tasks when algorithm execution ends.

        This method is called once at the end of the simulation to clean up any remaining resources
        or perform final calculations. In this implementation, it performs no specific tasks
        but maintains compliance with the abstract base R2AAlgorithm class interface.

        Returns:
            None
        """
        pass

from r2a.ir2a import IR2A
from base.message import SSMessage
from player.parser import *
import time
from statistics import mean
from sklearn.neighbors import KNeighborsRegressor
import numpy as np

class R2A_QKNN_Learning(IR2A):
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
        self.throughputs = []
        self.request_time = 0
        self.qi = []
        self.knn = KNeighborsRegressor(n_neighbors=3)
        self.replay_buffer = []  # store (state, action, reward, next_state)
        self.fitted = False

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
        self.request_time = time.perf_counter()
        self.send_down(msg)

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
        # Example: just throughput and length as state
        return [mean(self.throughputs) if self.throughputs else 0]

    def update_replay(self, state, action, reward, next_state):
        self.replay_buffer.append((state, action, reward, next_state))

    def fit_knn(self):
        # Minimal demonstration
        X = []
        y = []
        for (s, a, r, s2) in self.replay_buffer:
            X.append(s + [a])
            y.append(r)  # or Q(s,a)
        if X:
            self.knn.fit(X, y)
            self.fitted = True

    def select_action(self):
        # Predict Q for each possible action (quality index)
        state = self.get_state()
        best_q, best_action = float('-inf'), self.qi[0]
        for a in self.qi:
            if self.fitted:
                q_val = self.knn.predict([state + [a]])[0]
            else:
                q_val = 0
            if q_val > best_q:
                best_q = q_val
                best_action = a
        return best_action

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
        Handles the segment size response from the server and updates the learning model.
        1. Gets the current state of the system
        2. Calculates the reward based on segment size and elapsed time
        3. Gets the next state after the segment
        4. Updates the replay memory with state-action-reward-next_state tuple 
        5. Retrains the KNN model with updated data
        6. Records throughput measurement
        7. Forwards the segment message up the pipeline
        - Reward is calculated as: segment_size / elapsed_time
        - The state-action-reward-next_state tuple is used for reinforcement learning
        - The KNN model is retrained after each segment to adapt to network conditions
        - Throughput values are stored for future QoE calculations

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
        state = self.get_state()
        reward = msg.get_bit_length() / (time.perf_counter() - self.request_time)
        next_state = self.get_state()
        self.update_replay(state, self.qi[-1], reward, next_state)
        self.fit_knn()
        t = time.perf_counter() - self.request_time
        self.throughputs.append(msg.get_bit_length() / t)
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

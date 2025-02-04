from r2a.ir2a import IR2A
from base.message import SSMessage
from player.parser import *
import time
from statistics import mean
from scipy.spatial import KDTree
from scipy.special import softmax
import numpy as np

class R2A_QKNN(IR2A):
    """
    QoE-driven KNN-based rate adaptation algorithm combining Q-Learning with K-Nearest Neighbors.
    Implements the KNN-Q Learning algorithm from the paper with buffer management and SSIM approximation.
    """

    def __init__(self, id):
        IR2A.__init__(self, id)
        # Algorithm parameters from paper Table II
        self.eta = 0.3       # Learning rate
        self.gamma = 0.95    # Discount factor
        self.k = 3           # Number of neighbors
        self.tau = 0.3       # Temperature scaling
        # self.epsilon = 0.3   # Exploration rate (temporary)
        
        # Video streaming parameters
        self.segment_duration = 1  # Hardcoded from paper (T_segment)
        self.B_safe = 10           # Safe buffer level (sec)
        self.alpha = 50.0          # Penalty coefficients
        self.beta = 0.001       
        
        # SSIM calculation defaults (from News video in paper Table I)
        self.d = [-0.0106444,-0.0229079, -0.0253096, 0.0007417]
        
        # State tracking
        self.throughputs = []     # Measured throughput values
        self.buffer_level = 0     # Current buffer level (sec)
        self.last_quality = None  # Last selected quality index
        self.bitrates = []        # Available bitrates from MPD (bps)
        self.R_max = None         # Max bitrate in representation
        
        # KNN-Q Learning components
        self.replay_buffer = []   # (state, action, reward, next_state)
        self.tree : KDTree = None          # KDTree for fast lookups
        self.X = []               # State-action vectors
        self.y = []               # Q-values
        self.fitted = False

    def handle_xml_request(self, msg: SSMessage):
        """
        Handles XML manifest request by recording request time.
        
        Parameters
        ----------
        msg : SSMessage
            Message object containing XML request
        """
        self.request_time = time.perf_counter()
        self.send_down(msg)

    def handle_xml_response(self, msg: SSMessage):
        """
        Processes MPD response using existing parser to initialize streaming parameters.
        
        Parameters
        ----------
        msg : SSMessage
            Message containing MPD XML payload
        """
        parsed_mpd = parse_mpd(msg.get_payload())
        
        # Extract representations in MPD and calculate SSIM values
        self.qi = parsed_mpd.get_qi()
        self.R_max = max(self.qi) if self.qi else 1
        
        # Calculate initial throughput (for first segment decision)
        t = time.perf_counter() - self.request_time
        self.throughputs.append(msg.get_bit_length() / t)
        
        self.send_up(msg)

    def get_state(self):
        """
        Constructs state vector from current environment observations.
        
        Returns
        -------
        list
            [current_throughput, buffer_level, last_quality]
        """
        # Current throughput (5-segment moving average)
        window = self.throughputs[-5:] if len(self.throughputs) >=5 else self.throughputs
        current_throughput = mean(window) if window else 0
        
        return [
            current_throughput,
            self.buffer_level,
            self.last_quality if self.last_quality is not None else 0
        ]

    def _calculate_ssim(self, quality_idx):
        """
        Calculates SSIM approximation using paper's equation (1).
        
        Parameters
        ----------
        quality_idx : int
            Index of selected quality level
            
        Returns
        -------
        float
            Estimated SSIM value
        """
        R_a = quality_idx
        rho = np.log10(R_a / self.R_max)
        print(f"Rho: {rho}")
        ssim = (1 + \
            self.d[0]*rho + \
            self.d[1]*(rho**2) + \
            self.d[2]*(rho**3) + \
            self.d[3]*(rho**4))
        print(f"Quality: {quality_idx}, n\ SSIM: {ssim}")
        return(ssim)
        

    def calculate_reward(self, quality, prev_quality, download_time, segment_size):
        """
        Computes QoE reward using paper's equations (4) and (5).
        
        Parameters
        ----------
        quality : int
            Current quality index
        prev_quality : int
            Previous quality index
        download_time : float
            Time taken to download segment
        segment_size : int
            Size of downloaded segment in bits
            
        Returns
        -------
        float
            Calculated reward value
        """
        # Current SSIM calculation
        ssim = self._calculate_ssim(quality)
        
        # Smoothness penalty (Δq)
        if prev_quality is not None:
            prev_ssim = self._calculate_ssim(prev_quality)
            smoothness_penalty = self.alpha * abs(ssim - prev_ssim)
        else:
            smoothness_penalty = 0

        # Buffer penalty (φ(t)) from equation (5)
        underflow_risk = max(0, self.B_safe - self.buffer_level)
        print(f"Buffer: {self.buffer_level}, Underflow: {underflow_risk}")
        overflow = max(self.buffer_level - self.B_safe, 0)
        phi = self.alpha * underflow_risk + self.beta * (overflow ** 2)

        print(f"SSIM: {ssim}, Smoothness: {smoothness_penalty}, Phi: {phi}, Reward: {ssim - smoothness_penalty - phi}")
        
        return ssim - smoothness_penalty - phi

    def handle_segment_size_request(self, msg: SSMessage):
        """
        Handles segment request using softmax policy over KNN-predicted Q-values.
        
        Parameters
        ----------
        msg : SSMessage
            Segment request message
        """
        self.request_time = time.perf_counter()
        
        # Softmax action selection
        state = self.get_state()
        q_values = np.array([self.predict(state + [a]) for a in self.qi])

        # Check for NaNs or if all Q-values are the same
        if np.any(np.isnan(q_values)):
            # fallback, e.g. uniform distribution
            probs = np.ones(len(self.qi)) / len(self.qi)
        else:
            # Compute stable softmax
            probs = softmax(q_values / self.tau)

        chosen_qi = np.random.choice(self.qi, p=probs)
        
        msg.add_quality_id(chosen_qi)
        self.send_down(msg)

    def handle_segment_size_response(self, msg: SSMessage):
        """
        Updates learning model with new experience and manages buffer state.
        
        Parameters
        ----------
        msg : SSMessage
            Received segment response
        """
        # Calculate download metrics
        download_time = time.perf_counter() - self.request_time
        current_quality = msg.get_quality_id()
        
        # Update buffer using paper's equation (3)
        self.buffer_level = max(0, self.buffer_level - download_time) + self.segment_duration
        
        # Calculate reward and store experience
        state = self.get_state()
        reward = self.calculate_reward(current_quality, self.last_quality, 
                                      download_time, msg.get_bit_length())
        next_state = self.get_state()
        
        self.throughputs.append(msg.get_bit_length() / download_time)
        self.last_quality = current_quality
        
        # KNN-Q update
        self.update_replay(state, current_quality, reward, next_state)
        self.fit_knn()
        
        self.send_up(msg)

    def update_replay(self, state, action, reward, next_state):
        """
        Updates replay buffer and performs KNN-based Q-learning update.
        
        Parameters
        ----------
        state : list
            Current state vector
        action : int
            Selected quality index
        reward : float
            Calculated reward value
        next_state : list
            Observed next state
        """
        # TD Target calculation
        next_q_values = [self.predict(next_state + [a]) for a in self.qi]
        td_target = reward + self.gamma * max(next_q_values)
        
        # Find nearest neighbors and update
        if self.fitted:
            state_action = state + [action]
            distances, indices = self.tree.query([state_action], k=self.k)
            
            for idx in indices[0]:
                if idx < len(self.y):
                    self.y[idx] += self.eta * (td_target - self.y[idx])
        
        # Maintain replay buffer size
        if len(self.replay_buffer) > 1000:
            self.replay_buffer.pop(0)
        self.replay_buffer.append((state, action, reward, next_state))

    def fit_knn(self):
        """
        Rebuilds KDTree index from current replay buffer experiences.
        """
        if len(self.replay_buffer) > 0:
            self.X = [s + [a] for (s, a, _, _) in self.replay_buffer]
            self.y = [r + self.gamma * max([self.predict(ns + [a]) for a in self.qi]) 
                     for (_, _, r, ns) in self.replay_buffer]
            
            if len(self.X) > 0:
                self.tree = KDTree(np.array(self.X))
                self.fitted = True

    def predict(self, state_action):
        """
        Predicts Q-value using weighted average of nearest neighbors.
        
        Parameters
        ----------
        state_action : list
            Combined state-action vector
            
        Returns
        -------
        float
            Predicted Q-value
        """
        if not self.fitted or len(self.y) == 0:
            return 0
            
        distances, indices = self.tree.query(
            np.array(state_action).reshape(1, -1),
            k=min(self.k, len(self.y))
        )
        
        if len(indices) == 0:
            return 0
            
        # Weighted average of neighbor rewards
        weights = 1 / (distances.flatten() + 1e-6)
        return np.average([self.y[i] for i in indices.flatten()], weights=weights)

    def initialize(self):
        pass

    def finalization(self):
        pass
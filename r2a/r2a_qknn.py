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
    Algoritmo de adaptação de taxa baseado em KNN (K-Nearest Neighbors) e Q-Learning, 
    focado em QoE (Quality of Experience). Este algoritmo combina técnicas de aprendizado 
    por reforço com KNN para selecionar a melhor taxa de bits para streaming de vídeo, 
    considerando métricas como SSIM (Structural Similarity Index) e gerenciamento de buffer.
    """

    def __init__(self, id):
        IR2A.__init__(self, id)
        # Parâmetros iniciais do algoritmo
        self.eta = 0.1       # Taxa de aprendizado (learning rate)
        self.gamma = 0.9     # Fator de desconto (discount factor)
        self.tau = 0.5       # Escala de temperatura para softmax
        self.alpha = 10.0    # Coeficiente de penalidade para suavidade (smoothness penalty)
        self.beta = 0.001   # Coeficiente de penalidade para buffer
        self.k = 3           # Número de vizinhos mais próximos (KNN)

        # Contadores e métricas para ajuste dinâmico dos parâmetros
        self.episode_count = 0  # Contador de episódios
        self.reward_history = []  # Histórico de recompensas
        self.parameter_update_interval = 10  # Intervalo para ajustar parâmetros (a cada 50 episódios)

        # Parâmetros de streaming de vídeo
        self.segment_duration = 1  # Duração de cada segmento de vídeo (em segundos)
        self.B_safe = 15           # Nível seguro de buffer (em segundos)
        
        # Vetor de coeficientes para cálculo aproximado do SSIM (extraído do vídeo BigBuckBunny)
        self.d = [0.017901738412493887, -0.006908148312984288, -0.00017270496637129657, 0.00010360865155566306]
        
        # Rastreamento de estado
        self.throughputs = []     # Histórico de throughputs medidos
        self.buffer_level = 0     # Nível atual do buffer (em segundos)
        self.last_quality = None  # Última qualidade selecionada
        self.bitrates = []        # Taxas de bits disponíveis no MPD (em bps)
        self.R_max = None         # Maior taxa de bits disponível

        # Componentes do KNN-Q Learning
        self.replay_buffer = []   # Buffer de experiências (estado, ação, recompensa, próximo estado)
        self.tree : KDTree = None # Árvore KD para buscas rápidas de vizinhos mais próximos
        self.X = []               # Vetores de estado-ação
        self.y = []               # Valores Q (Q-values)
        self.fitted = False       # Indica se o modelo foi ajustado

    def handle_xml_request(self, msg: SSMessage):
        """
        Manipula a requisição do manifesto XML, registrando o tempo da requisição.
        """
        self.request_time = time.perf_counter()
        self.send_down(msg)

    def handle_xml_response(self, msg: SSMessage):
        """
        Processa a resposta do MPD, extraindo as representações disponíveis e calculando o throughput inicial.
        """
        parsed_mpd = parse_mpd(msg.get_payload())
        
        # Extrai as representações do MPD e calcula o SSIM
        self.qi = parsed_mpd.get_qi()
        self.R_max = max(self.qi) if self.qi else 1
        
        # Calcula o throughput inicial (para decisão do primeiro segmento)
        t = time.perf_counter() - self.request_time
        self.throughputs.append(msg.get_bit_length() / t)
        
        self.send_up(msg)

    def get_state(self):
        """
        Constrói o vetor de estado a partir das observações atuais do ambiente.
        """
        # Throughput atual (média móvel dos últimos 5 segmentos)
        window = self.throughputs[-5:] if len(self.throughputs) >= 5 else self.throughputs
        current_throughput = mean(window) if window else 0
        
        return [
            current_throughput,
            self.buffer_level,
            self.last_quality if self.last_quality is not None else 0
        ]

    def _calculate_ssim(self, quality_idx):
        """
        Calcula a aproximação do SSIM usando a equação (1) do artigo.
        """
        R_a = quality_idx
        rho = np.log10(R_a / self.R_max)
        ssim = (1 + \
            self.d[0] * rho + \
            self.d[1] * (rho ** 2) + \
            self.d[2] * (rho ** 3) + \
            self.d[3] * (rho ** 4))
        return ssim

    def calculate_reward(self, quality, prev_quality, download_time, segment_size):
        """
        Calcula a recompensa de QoE usando as equações (4) e (5) do artigo.
        """
        # Cálculo do SSIM atual
        ssim = self._calculate_ssim(quality)
        
        # Penalidade de suavidade (Δq)
        if prev_quality is not None:
            prev_ssim = self._calculate_ssim(prev_quality)
            smoothness_penalty = self.alpha * abs(ssim - prev_ssim)
        else:
            smoothness_penalty = 0

        # Penalidade de buffer (φ(t)) da equação (5)
        underflow_risk = max(0, self.B_safe - self.buffer_level)
        overflow = max(self.buffer_level - self.B_safe, 0)
        phi = self.alpha * underflow_risk + self.beta * (overflow ** 2)

        print(f"SSIM: {ssim}, Smoothness: {smoothness_penalty}, Phi: {phi}, Reward: {ssim - smoothness_penalty - phi}, {self.episode_count}")

        return ssim - smoothness_penalty - phi

    def handle_segment_size_request(self, msg: SSMessage):
        """
        Seleciona a qualidade do próximo segmento usando uma política softmax sobre os valores Q preditos pelo KNN.
        """
        self.request_time = time.perf_counter()
        
        # Seleção de ação via softmax
        state = self.get_state()
        q_values = np.array([self.predict(state + [a]) for a in self.qi])

        # Verifica se há NaNs ou se todos os valores Q são iguais
        if np.any(np.isnan(q_values)):
            probs = np.ones(len(self.qi)) / len(self.qi)  # Fallback para distribuição uniforme
        else:
            probs = softmax(q_values / self.tau)  # Calcula softmax estável

        chosen_qi = np.random.choice(self.qi, p=probs)
        msg.add_quality_id(chosen_qi)
        self.send_down(msg)

    def handle_segment_size_response(self, msg: SSMessage):
        """
        Atualiza o modelo de aprendizado com novas experiências e gerencia o estado do buffer.
        """
        # Calcula métricas de download
        download_time = time.perf_counter() - self.request_time
        current_quality = msg.get_quality_id()
        
        # Atualiza o buffer usando a equação (3) do artigo
        self.buffer_level = max(0, self.buffer_level - download_time) + self.segment_duration
        
        # Calcula a recompensa e armazena a experiência
        state = self.get_state()
        reward = self.calculate_reward(current_quality, self.last_quality, 
                                      download_time, msg.get_bit_length())
        next_state = self.get_state()
        
        # Atualiza métricas
        self.episode_count += 1
        self.reward_history.append(reward)

        # Ajusta os parâmetros a cada 50 episódios
        if self.episode_count == self.parameter_update_interval :
            self.adjust_parameters()
            self.episode_count = 0

        # Atualiza o throughput e a última qualidade selecionada
        self.throughputs.append(msg.get_bit_length() / download_time)
        self.last_quality = current_quality
        self.update_replay(state, current_quality, reward, next_state)
        self.fit_knn()
        self.send_up(msg)

    def adjust_parameters(self):
        """
        Ajusta os parâmetros do algoritmo com base na recompensa média dos últimos 50 episódios.
        Só ajusta os parâmetros se a recompensa média for menor que 0.6.
        """
        # Calcula a recompensa média
        avg_reward = np.mean(self.reward_history)
        print(f"Episódio {self.episode_count}, Recompensa Média: {avg_reward}")

        # Só ajusta os parâmetros se a recompensa média for menor que 0.6
        if avg_reward < 0.6:
            # Lógica de ajuste de parâmetros principais
            if avg_reward < 0:
                # Se a recompensa média for negativa, aumenta os parâmetros para explorar mais
                self.eta = min(self.eta * 1.1, 1.0)  # Aumenta a taxa de aprendizado (até no máximo 1.0)
                self.gamma = min(self.gamma * 1.05, 0.99)  # Aumenta o fator de desconto (até no máximo 0.99)
                self.tau = max(self.tau * 0.9, 0.1)  # Reduz a temperatura (explora menos)
                self.alpha = max(self.alpha * 0.9, 1.0)  # Reduz a penalidade de suavidade
                self.beta = max(self.beta * 0.9, 0.001)  # Reduz a penalidade de buffer
                self.parameter_update_interval = max(self.parameter_update_interval - 5, 10) #Reduz o intervalo para ajustar mais rápido
            else:
                # Se a recompensa média for positiva, ajusta os parâmetros para explorar menos
                self.eta = max(self.eta * 0.9, 0.01)  # Reduz a taxa de aprendizado (até no mínimo 0.01)
                self.gamma = max(self.gamma * 0.95, 0.5)  # Reduz o fator de desconto (até no mínimo 0.5)
                self.tau = min(self.tau * 1.1, 2.0)  # Aumenta a temperatura (explora mais)
                self.alpha = min(self.alpha * 1.1, 100.0)  # Aumenta a penalidade de suavidade
                # Ajusta as penalidades com base em métricas específicas
                if self.buffer_level < self.B_safe:
                    # Se o buffer estiver abaixo do nível seguro, reduz a penalidade de buffer
                    self.beta = max(self.beta * 0.9, 0.001)
                else:
                    # Se o buffer estiver acima do nível seguro, aumenta a penalidade de buffer
                    self.beta = min(self.beta * 1.1, 0.01)

            # Ajuste dinâmico do valor de k
            if avg_reward < 0:
                # Se a recompensa média for negativa, aumenta k para explorar mais
                self.k = min(self.k + 1, 5)  # Aumenta k (até no máximo 5)
            elif avg_reward < 0.6:
                # Se a recompensa média for positiva, reduz k para explorar menos
                self.k = max(self.k - 1, 3)  # Reduz k (até no mínimo 3)

            print(f"Parâmetros ajustados: eta={self.eta}, gamma={self.gamma}, tau={self.tau}, alpha={self.alpha}, beta={self.beta}, k={self.k}")
        else:
            print("Recompensa média acima de 0.6. Parâmetros não ajustados.")
            self.parameter_update_interval = min(self.parameter_update_interval + 5, 50)

        # Reinicia o histórico de recompensas
        self.reward_history = []
        self.episode_count = 0

    def update_replay(self, state, action, reward, next_state):
        """
        Atualiza o buffer de replay e realiza a atualização do Q-Learning baseado em KNN.
        """
        # Cálculo do TD Target
        next_q_values = [self.predict(next_state + [a]) for a in self.qi]
        td_target = reward + self.gamma * max(next_q_values)
        
        # Encontra os vizinhos mais próximos e atualiza
        if self.fitted:
            state_action = state + [action]
            distances, indices = self.tree.query([state_action], k=self.k)
            
            for idx in indices[0]:
                if idx < len(self.y):
                    self.y[idx] += self.eta * (td_target - self.y[idx])
        
        # Mantém o tamanho do buffer de replay
        if len(self.replay_buffer) > 1000:
            self.replay_buffer.pop(0)
        self.replay_buffer.append((state, action, reward, next_state))

    def fit_knn(self):
        """
        Reconstrói o índice da árvore KD a partir das experiências atuais no buffer de replay.
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
        Prediz o valor Q usando a média ponderada dos vizinhos mais próximos.
        """
        if not self.fitted or len(self.y) == 0:
            return 0
            
        distances, indices = self.tree.query(
            np.array(state_action).reshape(1, -1),
            k=min(self.k, len(self.y))
        )
        if len(indices) == 0:
            return 0
            
        # Média ponderada das recompensas dos vizinhos
        weights = 1 / (distances.flatten() + 1e-6)
        return np.average([self.y[i] for i in indices.flatten()], weights=weights)

    def initialize(self):
        pass

    def finalization(self):
        pass
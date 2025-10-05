from DQN import DQN
import torch
import torch.optim as optim
from collections import deque
import random 


class DQNAgent:
    def __init__(self,lr=1e-3,gamma = 0.99,buffer_size = 10000):
        # GPU or CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # policy net と target net
        self.policy = DQN().to(self.device)
        self.target = DQN().to(self.device)
        # パラメータ
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.memory = deque(maxlen=buffer_size)
        self.gamma = gamma
        self.batch_size = 32
        self.update_target()
    # target networkの更新
    def update_target(self):
        self.target.load_state_dict(self.policy.state_dict())

    # 行動
    def actions(self,state, eps = 0.2):
        if random.random() < eps:
            return random.randint(0 , 1) # -1 or 1 or 0
        
        s = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.policy(s)
        return int(q_values.argmax().item())

    def store(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def one_step(self):
        if len(self.memory) < self.batch_size:
            return
        
        batch = random.sample(self.memory, self.batch_size)

        state, action, reward, next_state, done = zip(*batch)


        # shapeを (batch, 1) に整える
        s  = torch.tensor([x[0] for x in state], dtype=torch.float32).unsqueeze(1).to(self.device)
        ns = torch.tensor([x[0] for x in next_state], dtype=torch.float32).unsqueeze(1).to(self.device)
        a  = torch.tensor(action, dtype=torch.long).unsqueeze(1).to(self.device)
        r  = torch.tensor(reward, dtype=torch.float32).unsqueeze(1).to(self.device)
        d  = torch.tensor(done, dtype=torch.float32).unsqueeze(1).to(self.device)

        q = self.policy(s).gather(1, a)
        with torch.no_grad():
            q_next = self.target(ns).max(1, keepdim=True)[0]
            q_target = r + self.gamma * q_next * (1 - d)

        loss = torch.nn.functional.mse_loss(q, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
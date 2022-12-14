import numpy as np
import random
from tqdm.auto import tqdm

import torch
import torch.nn.functional as F
from torch import optim

from QRobot import QRobot
from Maze import Maze
from ReplayDataSet import ReplayDataSet
from Runner import Runner
from torch_py.MyNetwork import MyNetwork

class Robot(QRobot):
    valid_action = ['u', 'r', 'd', 'l']

    ''' QLearning parameters'''

    step = 0
    EveryUpdate = 8  # the interval of target model's updating

    """some parameters of neural network"""
    target_model = None
    eval_model = None
    batch_size = 8
    learning_rate = 1e-2 
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    known_place = []
    
    def __init__(self, maze):
        """
        初始化 Robot 类
        :param maze:迷宫对象
        """
        super(Robot, self).__init__(maze)
        maze.set_reward(reward={
            "hit_wall": 100.,
            "destination": -5000.,
            "default": 1.,
            # "channel": 10.,
            # "corner": 50.,
            "dead_way": 100
        })
        self.maze = maze
        self.maze_size = maze.maze_size

        """build network"""
        self.target_model = None
        self.eval_model = None
        self._build_network()

        """create the memory to store data"""
        max_size = max(self.maze_size ** 2 * 3, 1e4)
        self.memory = ReplayDataSet(max_size=max_size)
        self.gamma = 0.74
        self.epsilon0 = 0.8
        self.visit_time = {}
        self.loss = 0

        # self.memory.build_full_view(maze=maze)

        epoch = 5  # 训练轮数
        training_per_epoch = int(self.maze_size * self.maze_size * 5)

        self.runner = Runner(self)
        self.runner.run_training(epoch, training_per_epoch)

        """parameters"""
        self.q_table


    def _build_network(self):
        seed = 0
        random.seed(seed)

        """build target model"""
        self.target_model = MyNetwork(state_size=2, action_size=4, seed=seed).to(self.device)

        """build eval model"""
        self.eval_model = MyNetwork(state_size=2, action_size=4, seed=seed).to(self.device)

        """build the optimizer"""
        self.optimizer = optim.Adam(self.eval_model.parameters(), lr=self.learning_rate)

    def show_Q(self):
        Q_tabel = {}
        for i in range(self.maze.maze_size):
            for j in range(self.maze.maze_size):

                state = np.array([i, j], dtype=np.int16)
                state = torch.from_numpy(state).float().to(self.device)

                self.eval_model.eval()
                with torch.no_grad():
                    q_value = self.eval_model(state).cpu().data.numpy()

                # print(state, q_value)
                
                action = self.valid_action[np.argmin(q_value).item()]

                Q_tabel[(j, i)] = q_value
                print(i, j, action, q_value)
        return Q_tabel

    def target_replace_op(self):
        """
            Soft update the target model parameters.
            θ_target = τ*θ_local + (1 - τ)*θ_target
        """

        # self.TAU = 0.5

        # for target_param, eval_param in zip(self.target_model.parameters(), self.eval_model.parameters()):
        #     target_param.data.copy_(self.TAU * eval_param.data + (1.0 - self.TAU) * target_param.data)

        """ replace the whole parameters"""
        self.target_model.load_state_dict(self.eval_model.state_dict())

    def _choose_action(self, state):
        state = np.array(state)
        state = torch.from_numpy(state).float().to(self.device)
        if random.random() < self.epsilon:
            action = random.choice(self.valid_action)
        else:
            self.eval_model.eval()
            with torch.no_grad():
                q_next = self.eval_model(state).cpu().data.numpy()  # use target model choose action
            self.eval_model.train()

            action = self.valid_action[np.argmin(q_next).item()]
        return action

    def _learn(self, batch: int = 16):
        if len(self.memory) < batch:
            batch = len(self.memory)


        state, action_index, reward, next_state, is_terminal = self.memory.random_sample(batch)

        """ convert the data to tensor type"""
        state = torch.from_numpy(state).float().to(self.device)
        action_index = torch.from_numpy(action_index).long().to(self.device)
        reward = torch.from_numpy(reward).float().to(self.device)
        next_state = torch.from_numpy(next_state).float().to(self.device)
        is_terminal = torch.from_numpy(is_terminal).int().to(self.device)
        
        while True:
            
            self.eval_model.train()
            self.target_model.eval()

            """Get max predicted Q values (for next states) from target model"""
            Q_targets_next = self.target_model(next_state).detach().min(1)[0].unsqueeze(1)
            # Q_targets_next_ave = self.target_model(next_state).detach().mean(axis=1).unsqueeze(1)

            """Compute Q targets for current states"""
            Q_targets = reward + self.gamma * Q_targets_next * (torch.ones_like(is_terminal) - is_terminal)
            
            """Get expected Q values from local model"""
            self.optimizer.zero_grad()
            Q_expected = self.eval_model(state).gather(dim=1, index=action_index)

            """Compute loss"""
            loss = F.mse_loss(Q_expected, Q_targets)
            loss_item = loss.item()

            """ Minimize the loss"""
            loss.backward()
            self.optimizer.step()

            if loss_item < 1: break
            # print(state, action_index, Q_expected, Q_targets)

        return loss_item

    def train_update(self):
        state = self.sense_state()
        action = self._choose_action(state)

        if not action in self.known_place:
            self.known_place.append(action)

        reward = self.maze.move_robot(action)
        
        next_state = self.sense_state()

        if not next_state in self.visit_time:
            self.visit_time[next_state] = 0
        else:
            self.visit_time[next_state] += 1

        # if next_state

        is_terminal = 1 if next_state == self.maze.destination or next_state == state else 0

        self.memory.add(state, self.valid_action.index(action), reward, next_state, is_terminal)

        """--间隔一段时间更新target network权重--"""

        """---update the step and epsilon---"""
        self.step += 1
        self.epsilon = max(0.6, self.epsilon * 0.995)
        self.loss = self._learn(batch=16)
        if self.step % self.EveryUpdate == 0:
            """copy the weights of eval_model to the target_model"""
            self.target_replace_op()

        # print('memory num:', len(self.memory))

        return action, reward

    def test_update(self):
        state = np.array(self.sense_state(), dtype=np.int16)
        state = torch.from_numpy(state).float().to(self.device)

        self.eval_model.eval()
        with torch.no_grad():
            q_value = self.eval_model(state).cpu().data.numpy()

        # print(state, q_value)
        
        action = self.valid_action[np.argmin(q_value).item()]
        reward = self.maze.move_robot(action)
        return action, reward


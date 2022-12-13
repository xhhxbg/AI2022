'''
Author: xhhxbg 1049085339@qq.com
Date: 2022-12-03 21:24:43
LastEditors: xhhxbg 1049085339@qq.com
LastEditTime: 2022-12-12 14:48:51
FilePath: \AIProject4\torch_py\QNetwork.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
from abc import ABC

import torch.nn as nn
import torch


class QNetwork(nn.Module, ABC):
    """Actor (Policy) Model."""

    def __init__(self, state_size: int, action_size: int, seed: int):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """

        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.input_hidden = nn.Sequential(
            nn.Linear(state_size, 512),
            nn.ReLU(False),
            nn.Linear(512, 512),
            nn.ReLU(False),
        )

        self.final_fc = nn.Linear(512, action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = self.input_hidden(state)
        return self.final_fc(x)


if __name__ == "__main__":
    # os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    # device = torch.device("cpu")
    net = QNetwork(2, 4, 0).to(device)

    x = torch.tensor([1, 1]).float().unsqueeze(0).to(device)
    #
    # torch.nn.DataParallel(net, device_ids=[0])
    print(net(x))

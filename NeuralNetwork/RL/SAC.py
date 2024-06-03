import random
import gym
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.distributions import Normal
import matplotlib.pyplot as plt
import rl_utils


class PolicyNetContinuous(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, action_bound):
        """
        定义策略网络。策略网络输出一个高斯分布的均值和标准差来表示动作分布
        通过重参数化技巧（reparameterization trick）来进行采样
        为了处理动作边界，它使用 tanh 函数对动作进行缩放。
        :param state_dim: 状态的维度
        :param hidden_dim: 隐藏层的维度
        :param action_dim: 动作的维度
        :param action_bound: 动作的边界，用于缩放动作到适当的范围
        """
        super(PolicyNetContinuous, self).__init__()
        # 全连接层
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        # 输出均值的全连接层
        self.fc_mu = torch.nn.Linear(hidden_dim, action_dim)
        # 输出标准差的全连接层
        self.fc_std = torch.nn.Linear(hidden_dim, action_dim)
        # 动作边界
        self.action_bound = action_bound

    def forward(self, x):
        # 隐藏层，激活函数为ReLU
        x = F.relu(self.fc1(x))
        # 计算动作均值
        mu = self.fc_mu(x)
        # 计算动作标准差，使用softplus函数确保标准差为正
        std = F.softplus(self.fc_std(x))
        # 创建正态分布
        dist = Normal(mu, std)
        # 重参数化采样
        normal_sample = dist.rsample()  # rsample()是重参数化采样
        # 计算对数概率
        log_prob = dist.log_prob(normal_sample)
        # 使用tanh函数将动作限制在[-1, 1]
        action = torch.tanh(normal_sample)
        # 计算tanh_normal分布的对数概率密度, 修正tanh变换后的对数概率密度
        log_prob = log_prob - torch.log(1 - torch.tanh(action).pow(2) + 1e-7)
        # 缩放动作到指定的边界
        action = action * self.action_bound
        # 返回动作和对应的对数概率
        return action, log_prob


class QValueNetContinuous(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        """
        该网络用于估计在给定状态和动作下的Q值（即动作价值）。
        这个网络接受状态和动作的拼接向量作为输入，输出一个实数来表示Q值。
        :param state_dim: 状态的维度
        :param hidden_dim: 隐藏层的维度
        :param action_dim: 动作的维度
        """
        super(QValueNetContinuous, self).__init__()
        # 输入层: 状态维度 + 动作维度 -> 隐藏层
        self.fc1 = torch.nn.Linear(state_dim + action_dim, hidden_dim)
        # 隐藏层:
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        # 输出层: 输出层：隐藏层 -> Q值
        self.fc_out = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x, a):
        # # 将状态和动作拼接成一个输入向量
        cat = torch.cat([x, a], dim=1)
        # 输入层，全连接+ReLU激活函数
        x = F.relu(self.fc1(cat))
        # 隐藏层，全连接+ReLU激活函数
        x = F.relu(self.fc2(x))
        # 输出层，输出Q值
        return self.fc_out(x)



class SACContinuous:
    '''
    处理连续动作的SAC算法
    SAC 使用两个 Critic 网络和来使 Actor 的训练更稳定，而这两个 Critic 网络在训练时则各自需要一个目标价值网络。
    因此，SAC 算法一共用到 5 个网络，分别是一个策略网络、两个价值网络和两个目标价值网络。
    '''
    def __init__(self, state_dim, hidden_dim, action_dim, action_bound,
                 actor_lr, critic_lr, alpha_lr, target_entropy, tau, gamma,
                 device):
        # 策略网络, 用于生成动作
        self.actor = PolicyNetContinuous(state_dim, hidden_dim, action_dim, action_bound).to(device)
        # 第一个Q网络, 用于评估动作价值
        self.critic_1 = QValueNetContinuous(state_dim, hidden_dim, action_dim).to(device)
        # 第二个Q网络, 用于评估动作价值
        self.critic_2 = QValueNetContinuous(state_dim, hidden_dim, action_dim).to(device)
        # 第一个目标Q网络
        self.target_critic_1 = QValueNetContinuous(state_dim, hidden_dim, action_dim).to(device)
        # 第二个目标Q网络
        self.target_critic_2 = QValueNetContinuous(state_dim, hidden_dim, action_dim).to(device)
        # 令目标Q网络的初始参数和Q网络一样
        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())
        # 策略网络的优化器
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        # 两个评论家网络的优化器
        self.critic_1_optimizer = torch.optim.Adam(self.critic_1.parameters(), lr=critic_lr)
        self.critic_2_optimizer = torch.optim.Adam(self.critic_2.parameters(), lr=critic_lr)
        # 使用alpha的log值,可以使训练结果比较稳定
        self.log_alpha = torch.tensor(np.log(0.01), dtype=torch.float)
        # 可以对alpha求梯度
        self.log_alpha.requires_grad = True
        # log_alpha 优化器
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=alpha_lr)
        # 目标熵的大小
        self.target_entropy = target_entropy
        # 折扣因子
        self.gamma = gamma
        # 软更新系数
        self.tau = tau
        # 计算设备（CPU或GPU）
        self.device = device

    def take_action(self, state):
        """
        将输入状态转换为张量，并使用策略网络生成动作。
        :param state: 输入状态
        :return: 动作
        """
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        action = self.actor(state)[0]
        return [action.item()]

    def calc_target(self, rewards, next_states, dones):
        """
        计算目标Q值
        :param rewards:
        :param next_states:
        :param dones:
        :return:
        """
        # 使用策略网络生成下一个状态的动作和对应的对数概率
        next_actions, log_prob = self.actor(next_states)
        # 计算熵值
        entropy = -log_prob
        # 两个目标 Q 网络的最小值
        q1_value = self.target_critic_1(next_states, next_actions)
        q2_value = self.target_critic_2(next_states, next_actions)
        next_value = torch.min(q1_value, q2_value) + self.log_alpha.exp() * entropy
        # 目标 Q 值
        td_target = rewards + self.gamma * next_value * (1 - dones)
        return td_target

    def soft_update(self, net, target_net):
        """
        使用软更新方式将 Q 网络的参数更新到目标 Q 网络中
        :param net:
        :param target_net:
        :return:
        """
        for param_target, param in zip(target_net.parameters(),net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) + param.data * self.tau)

    def update(self, transition_dict):
        # 数据转换为 PyTorch 张量并移动到指定的计算设备上
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions'], dtype=torch.float).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)

        # 和之前章节一样,对倒立摆环境的奖励进行重塑以便训练
        rewards = (rewards + 8.0) / 8.0

        # 更新两个Q网络
        # 计算目标 Q 值
        td_target = self.calc_target(rewards, next_states, dones)
        # 计算并更新 Q 网络的损失
        critic_1_loss = torch.mean(F.mse_loss(self.critic_1(states, actions), td_target.detach()))
        critic_2_loss = torch.mean(F.mse_loss(self.critic_2(states, actions), td_target.detach()))
        self.critic_1_optimizer.zero_grad()
        critic_1_loss.backward()
        self.critic_1_optimizer.step()
        self.critic_2_optimizer.zero_grad()
        critic_2_loss.backward()
        self.critic_2_optimizer.step()

        # 更新策略网络
        # 生成新的动作和对应的对数概率
        new_actions, log_prob = self.actor(states)
        # 计算熵项（负的对数概率）
        entropy = -log_prob
        # 计算新的动作在两个 Q 网络下的 Q 值
        q1_value = self.critic_1(states, new_actions)
        q2_value = self.critic_2(states, new_actions)
        # 计算策略损失，并更新策略网络参数
        actor_loss = torch.mean(-self.log_alpha.exp() * entropy - torch.min(q1_value, q2_value))
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # 更新alpha值
        alpha_loss = torch.mean((entropy - self.target_entropy).detach() * self.log_alpha.exp())
        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

        # 软更新目标 Q 网络
        self.soft_update(self.critic_1, self.target_critic_1)
        self.soft_update(self.critic_2, self.target_critic_2)


if __name__ == "__main__":
    env_name = 'Pendulum-v1'
    env = gym.make(env_name)
    # 状态维度
    state_dim = env.observation_space.shape[0]
    # 动作维度
    action_dim = env.action_space.shape[0]
    # 动作边界
    action_bound = env.action_space.high[0]  # 动作最大值
    random.seed(0)
    np.random.seed(0)
    env.seed(0)
    torch.manual_seed(0)

    # 策略网络学习器
    actor_lr = 3e-4
    # 评论家的学习率
    critic_lr = 3e-3
    # 熵系数学习率
    alpha_lr = 3e-4
    num_episodes = 100
    # 隐藏层个数
    hidden_dim = 128
    gamma = 0.99
    # 软更新参数
    tau = 0.005
    buffer_size = 100000
    minimal_size = 1000
    batch_size = 64
    target_entropy = -env.action_space.shape[0]
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # 初始化经验回放缓冲区
    replay_buffer = rl_utils.ReplayBuffer(buffer_size)
    # 初始化 SAC 代理
    agent = SACContinuous(state_dim, hidden_dim, action_dim, action_bound,
                          actor_lr, critic_lr, alpha_lr, target_entropy, tau,
                          gamma, device)

    # 训练代理
    return_list = rl_utils.train_off_policy_agent(env, agent, num_episodes,
                                                  replay_buffer, minimal_size,
                                                  batch_size)

    episodes_list = list(range(len(return_list)))
    plt.plot(episodes_list, return_list)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('SAC on {}'.format(env_name))
    plt.show()

    mv_return = rl_utils.moving_average(return_list, 9)
    plt.plot(episodes_list, mv_return)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('SAC on {}'.format(env_name))
    plt.show()

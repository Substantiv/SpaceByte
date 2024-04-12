"""
    https://zhuanlan.zhihu.com/p/659089157
"""

import gym
from stable_baselines3 import DQN


def main():
    env = gym.make('MountainCar-v0')  # 创建环境
    # 学习率（learning_rate）：学习率决定了模型参数更新的速度。较高的学习率可能会导致训练更快，但可能会影响模型的稳定性和最终性能。较低的学习率可能会导致训练更慢，但可能会得到更好的性能。
    # 折扣因子（gamma）：折扣因子决定了未来奖励的价值。较高的折扣因子值（接近1）会让算法更加关注长期的奖励，而较低的值（接近0）会让算法更加关注短期的奖励。
    model = DQN("MlpPolicy", env, learning_rate=0.005, gamma=0.99, verbose=1)  # 创建模型
    model.learn(total_timesteps=20)  # 训练模型
    model.save("dqn_cartpole")  # 保存模型
    test_model(model)  # 测试模型


def test_model(model):
    env = gym.make('MountainCar-v0', render_mode='human')  # 可视化只能在初始化时指定
    obs, _ = env.reset()
    done1, done2 = False, False
    total_reward = 0

    while not done1 or done2:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done1, done2, info = env.step(action)
        total_reward += reward

    print(f'Total Reward: {total_reward}')
    env.close()


if __name__ == "__main__":
    main()

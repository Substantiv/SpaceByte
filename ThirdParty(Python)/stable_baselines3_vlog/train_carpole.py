"""
    https://zhuanlan.zhihu.com/p/659089157
"""

import gymnasium as gym
from stable_baselines3 import PPO


def main():
    env = gym.make('CartPole-v1')                    # 创建环境
    model = PPO("MlpPolicy", env, verbose=0)  # 创建模型: "MlpPolicy"多层感知机策略, verbose=1以打印训练进度
    model.learn(total_timesteps=20)                  # 训练模型
    model.save("PPO_Cartpole")                       # 保存模型
    test_model(model)                                # 测试模型


def test_model(model):
    # 创建测试环境
    env = gym.make('CartPole-v1', render_mode='human')
    # 重置环境
    obs, _ = env.reset()
    done1, done2 = False, False
    total_reward = 0

    while not done1 or done2:
        # 调用模型获取下一步的动作和状态
        action, states = model.predict(obs, deterministic=True)
        # 后去动作的观测和奖励
        obs, rewards, dones1, dones2, info = env.step(action)
        total_reward += rewards

    print(f'Total reward: {total_reward}')
    env.close()


if __name__ == "__main__":
    main()

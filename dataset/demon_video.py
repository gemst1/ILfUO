import gym
import ray
from ray.rllib.agents import ppo
from monitor_multi import Monitor_Multi

def every_episode(episode_id):
    return True

checkpoint = "./experts/PPO/PPO_Pusher3DOF-v1_0_2020-08-18_17-09-26c3c0kf_w/checkpoint_1000/checkpoint-1000"

ray.init()
agent = ppo.PPOTrainer(env="Pusher3DOF-v1")
agent.restore(checkpoint)

env = gym.make('Pusher3DOF-v1')
env = Monitor_Multi(env, './results', camera_names=["view1"], force=True, video_callable=every_episode)
# env = Monitor_Multi(env, './results_3', force=True)
while env.episode_id < 5000:
    obs = env.reset()
    env.viewer_setup()
    for _ in range(50):
        action = agent.compute_action(obs)
        obs, reward, _, _ = env.step(action) # take a random action
    print("Episode: ", env.episode_id, ", reward: ", reward)
    if reward < -0.4:
        env.episode_id -= 1
env.close()
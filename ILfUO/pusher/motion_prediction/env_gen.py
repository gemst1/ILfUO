import gym
import numpy as np
import matplotlib.pyplot as plt
import ray
from ray.rllib.agents import ppo
import imageio
import os
import copy

results_path = './evaluation_results'
if not os.path.exists(results_path):
    os.makedirs(results_path)

# Imitation learning environment setup
env_nums = 10
env_name = "Pusher3DOF-v1"
result_file = "random_envs_cond"

# Load Expert policy
checkpoint = "../../../dataset/experts/PPO/PPO_Pusher3DOF-v1_0_2020-08-18_17-09-26c3c0kf_w/checkpoint_1000/checkpoint-1000"
ray.init()
agent = ppo.PPOTrainer(env=env_name)
agent.restore(checkpoint)

envs_cond = []
i = 0

# frames for various length
vf = np.ones((25, 4), 'int') * 24
vf[:10, 0] = np.round(np.arange(1, 11) * 25 / 10).astype(int) - 1
vf[:15, 1] = np.round(np.arange(1, 16) * 25 / 15).astype(int) - 1
vf[:20, 2] = np.round(np.arange(1, 21) * 25 / 20).astype(int) - 1
vf[:25, 3] = np.round(np.arange(1, 26) * 25 / 25).astype(int) - 1

while i < env_nums:
    vp = np.random.uniform(low=0, high=360)
    imgs = []

    env = gym.make(env_name, vp=vp)
    obs = env.reset()
    ctx_img = env.render(mode='rgb_array', width=48, height=48)/127.5-1
    object = env.env.object
    goal = env.env.goal
    rgbatmp = env.env.model.geom_rgba
    geompostemp = env.env.model.geom_pos
    for j in range(50):
        action = agent.compute_action(obs)
        obs, reward, _, _ = env.step(action) # take a random action
        if j%2==1:
            img = env.render(mode='rgb_array', width=48, height=48)
            imgs.append(img)
    print("Episode: ", i, ", reward: ", reward)
    env.close()

    if reward > -0.3:
        plt.imshow((ctx_img+1)/2.)
        plt.show()

        cond = {'vp': vp,
                'object': object,
                'goal': goal,
                'geomsrgba': rgbatmp,
                'geomspos': geompostemp,
                'ctx_img': ctx_img}
        envs_cond.append(copy.deepcopy(cond))
        for k in range(4):
            imageio.mimsave(results_path + '/env_%d_vl_%d_gt.gif' % (i, k), np.asarray(imgs)[vf[:, k]], duration=0.1)
        i += 1

np.save(result_file, envs_cond, allow_pickle=True)
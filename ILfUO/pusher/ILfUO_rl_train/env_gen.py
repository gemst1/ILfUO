import gym
import numpy as np
import imageio

# Imitation learning environment setup
env_nums = 10
env_name = "Pusher3DOF_ILfUO-v1"
result_file = "random_envs_cond"

envs_cond = []
for i in range(env_nums):
    vp = np.random.uniform(low=0, high=360)

    env = gym.make(env_name, vp=vp)
    env.reset()
    ctx_img = env.render(mode='rgb_array', width=48, height=48)/127.5-1
    env.close()

    object = env.env.object
    goal = env.env.goal
    rgbatmp = env.env.model.geom_rgba
    geompostemp = env.env.model.geom_pos
    cond = {'vp': vp,
            'object': object,
            'goal': goal,
            'geomsrgba': rgbatmp,
            'geomspos': geompostemp,
            'ctx_img': ctx_img}
    envs_cond.append(cond)

np.save(result_file, envs_cond, allow_pickle=True)
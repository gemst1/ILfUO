import gym
import tensorflow as tf
from baselines import custom_run
import numpy as np
import imageio
import datetime
import os
import matplotlib.pyplot as plt
try:
    from mpi4py import MPI
except ImportError:
    MPI = None

# Parameters
envs_path = './random_envs_cond.npy'
env_name = 'Pusher3DOF_ILfUO-v1'
vidgen_model_path = '../motion_prediction/results/models/vid_gen_1000'
vidgen_model = tf.keras.models.load_model(vidgen_model_path)
enc_content = vidgen_model.encoder_content
enc_motion = vidgen_model.encoder_motion
Rm = vidgen_model.Rm
dec = vidgen_model.decoder
dm = 40
dc = 200
de = 20
en_dim = 64
# results path
results_path = './baselines_results'
nowdate = datetime.datetime.now()
nowdate_dir = '/' + nowdate.strftime("%Y%m%d_%H-%M-%S")
results_path = results_path + nowdate_dir

# Imitation learning environment setup
envs_cond = np.load(envs_path, allow_pickle=True)
vls = np.zeros((4, 24, 4)) # video length one-hot encoding
vls[0, :, 0] = 1
vls[1, :, 1] = 1
vls[2, :, 2] = 1
vls[3, :, 3] = 1

env_num = 0
vp = envs_cond[env_num]['vp']
object = envs_cond[env_num]['object']
goal = envs_cond[env_num]['goal']
rgbatmp = envs_cond[env_num]['rgbatmp']
geompostemp = envs_cond[env_num]['geompostemp']
ctx_img = envs_cond[env_num]['ctx_img']

for i in range(vls.shape[0]):
    # Demonstration translation and reward trajectory calculation
    z_c, features = enc_content(tf.reshape(ctx_img,[-1, 48, 48, 3]))
    z_m, _ = enc_motion(tf.reshape(ctx_img, [-1, 48, 48, 3]))

    vl = vls[i:i+1]
    eps = tf.concat([tf.random.truncated_normal([1, 24, 20]), vl], axis=-1)
    z_cs = tf.reshape(tf.tile(z_c, [1, 25]), [-1, 25, dc])
    z_ms = tf.concat([tf.reshape(z_m, [-1,1,dm]), Rm(eps, initial_state=z_m)], axis=1)
    z = tf.concat([z_cs, z_ms], axis=-1)

    f0 = tf.tile(tf.reshape(features[0], [-1, 1, 24, 24, en_dim]), [1, 25, 1, 1, 1])
    f1 = tf.tile(tf.reshape(features[1], [-1, 1, 12, 12, en_dim*2]), [1, 25, 1, 1, 1])
    f2 = tf.tile(tf.reshape(features[2], [-1, 1, 6, 6, en_dim*4]), [1, 25, 1, 1, 1])
    f3 = tf.tile(tf.reshape(features[3], [-1, 1, 3, 3, en_dim*8]), [1, 25, 1, 1, 1])

    inputs_dec = [z, f0, f1, f2, f3]
    vid = dec(inputs_dec, training=False)

    demon_latent = z[0]
    demon_image = vid[0]

    # Save prediction results
    if not os.path.exists(results_path):
        os.makedirs(results_path + "/vidgen_results")

    imageio.mimsave(results_path + "/vidgen_results/gen_vl_%d.gif" % (i), (vid[0]+1)/2., duration=0.1)
    plt.imshow((ctx_img + 1) / 2.)
    plt.axis('off')
    plt.savefig(results_path + '/vidgen_results/src')
    plt.close()

    # Train imitation agent
    args = ['--alg=custom_ppo2', '--env='+env_name, '--network=mlp', 'num_env=4',
            '--save_video_interval=4000',
            '--nsteps=4000', '--num_timesteps=5e5', '--ent_coef=0.0', '--lr=1e-5', '--vf_coef=1.0', '--max_grad_norm=None',
            '--lam=1.0', '--nminibatches=100',  '--noptepochs=30','--cliprange=0.3', '--save_interval=5e4']

    config = {'log_path': results_path,
              'enc_motion':enc_motion,
              'enc_content':enc_content,
              'vp': vp,
              'object': object,
              'goal': goal,
              'geomsrgba': rgbatmp,
              'geomspos': geompostemp,
              'demon_latent': demon_latent,
              'demon_image': demon_image,
              'lambda_img': 1,
              'dc': dc,
              'dm': dm}

    model, env = custom_run.main(args, config)

    # save gif
    imgs = []
    obs = env.reset()
    for j in range(50):
        actions, _, _, _ = model.step(obs)
        obs, rew, done, _ = env.step(actions.numpy())
        if j % 2 == 1:
            img = env.render(mode='rgb_array')
            imgs.append(img)

    imageio.mimsave(results_path + '/vidgen_results/agent_vl_%d.gif' % (i), imgs, duration=0.1)

    env.close()
    del env, model

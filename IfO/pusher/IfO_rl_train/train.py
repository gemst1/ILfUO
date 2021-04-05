import gym
import tensorflow as tf
from baselines import custom_run
import numpy as np
import imageio
import datetime
import os
try:
    from mpi4py import MPI
except ImportError:
    MPI = None

# Parameters
demon_path = '../../../dataset/singleview_push_sim_5000.npy'
ifo_model_path = '../translator/results/models/model_80000'
ifo_model = tf.keras.models.load_model(ifo_model_path)
enc_1 = ifo_model.encoder_1
enc_2 = ifo_model.encoder_2
trans = ifo_model.translator
dec = ifo_model.decoder
n_demon = 100
# results path
results_path = './baselines_results'
nowdate = datetime.datetime.now()
nowdate_dir = '/' + nowdate.strftime("%Y%m%d_%H-%M-%S")
results_path = results_path + nowdate_dir

# Imitation learning environment setup
vp = np.random.uniform(low=0, high=360)

# while True:
#     object_ = [np.random.uniform(low=-1.0, high=-0.4),
#                np.random.uniform(low=0.3, high=1.2)]
#     goal = [np.random.uniform(low=-1.2, high=-0.8),
#             np.random.uniform(low=0.8, high=1.2)]
#
#     if np.linalg.norm(np.array(object_)-np.array(goal)) > 0.45: break
# object = np.array(object_)
# goal = np.array(goal)
#
# for body in range(5):
#     body = body + 1
#     pos_x = np.random.uniform(low=-0.9, high=0.9)
#     pos_y = np.random.uniform(low=0, high=1.0)
#     rgba = self.getcolor()
#     isinv = np.random.random()
#     if isinv > 0.5:
#         rgba[-1] = 0.
#     rgbatmp[body, :] = rgba
#     geompostemp[body, 0] = pos_x
#     geompostemp[body, 1] = pos_y

env = gym.make('Pusher3DOF_IfO-v1', vp=vp)
env.reset()
ctx_img = env.render(mode='rgb_array', width=48, height=48)/127.5-1
env.close()

object = env.env.object
goal = env.env.goal
rgbatmp = env.env.model.geom_rgba
geompostemp = env.env.model.geom_pos

# Demonstration translation and reward trajectory calculation
demon_data = np.load(demon_path)
demon_id = np.random.choice(5000, n_demon)
demon_sample = demon_data[:, demon_id]

z1, _ = enc_1(tf.reshape(demon_sample,[-1, 48, 48, 3]))
z2, ctx_features = enc_2(tf.reshape(ctx_img, [-1, 48, 48, 3]))
z2 = tf.tile(z2, [n_demon*25, 1])
ctx_features = [tf.tile(ctx_features[0], [n_demon*25, 1, 1, 1]),
                tf.tile(ctx_features[1], [n_demon*25, 1, 1, 1]),
                tf.tile(ctx_features[2], [n_demon*25, 1, 1, 1]),
                tf.tile(ctx_features[3], [n_demon*25, 1, 1, 1])]
z3_hat = trans(z1, z2)
trans_img = dec(z3_hat, ctx_features)
z3, _ = enc_1(trans_img)

demon_latent = tf.reduce_mean(tf.reshape(z3, [25, -1, 1024]), axis=1)
demon_image = tf.reduce_mean(tf.reshape(trans_img, [25, -1, 48, 48, 3]), axis=1)

# Save translation results
if not os.path.exists(results_path):
    os.makedirs(results_path + "/trans_results")

trans_img = tf.reshape(np.clip((trans_img+1)/2, 0, 1), [25, -1, 48, 48, 3])
for i in range(10):
    imageio.mimsave(results_path + "/trans_results/%d_trans.gif" % i, trans_img[:, i], duration=0.1)
    imageio.mimsave(results_path + "/trans_results/%d_src.gif" % i, demon_sample[:, i], duration=0.1)

# Train imitation agent
args = ['--alg=ppo2', '--env=Pusher3DOF_IfO-v1', '--network=mlp', 'num_env=4',
        '--save_video_interval=4000', '--play',
        '--nsteps=4000', '--num_timesteps=1e7', '--ent_coef=0.0', '--lr=1e-5', '--vf_coef=1.0', '--max_grad_norm=None',
        '--lam=1.0', '--nminibatches=100',  '--noptepochs=30','--cliprange=0.3']

config = {'log_path': './baselines_results' + nowdate_dir,
         'enc_path':ifo_model_path,
          'vp': vp,
          'object': object,
          'goal': goal,
          'geomsrgba': rgbatmp,
          'geomspos': geompostemp,
          'demon_latent': demon_latent,
          'demon_image': demon_image,
          'lambda_img': 1}

model = custom_run.main(args, config)
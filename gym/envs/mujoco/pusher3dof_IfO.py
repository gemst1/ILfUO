import gym
from gym import spaces
import numpy as np
from gym import utils
from gym.envs.mujoco.mujoco_env import MujocoEnv
import tensorflow as tf

DICT_SPACE = spaces.Dict({"internal_state": spaces.Box(low=-10, high=10, shape=(8,)),
                          "img": spaces.Box(low=-1, high=1, shape=(48, 48, 3))})

TUPLE_SPACE = spaces.Tuple([
    spaces.Box(low=-10, high=10, shape=(8, )),
    spaces.Box(low=-1, high=1, shape=(48, 48, 3))
])

class PusherEnv3DOF_IfO(MujocoEnv, utils.EzPickle):
    def __init__(self, **kwargs):
        utils.EzPickle.__init__(self)
        self._kwargs = kwargs
        # self.observation_space = spaces.Box(low=-10, high=10, shape=(1024,))
        if 'enc_path' in self._kwargs:
            self.ifo_model = tf.keras.models.load_model(self._kwargs['enc_path'])
            self.enc = self.ifo_model.encoder_1


        MujocoEnv.__init__(self, '3link_gripper_push_2d.xml', 5)
        self.itr=0

    def step(self, a):
        self.do_simulation(a, self.frame_skip)
        pobj = self.get_body_com("object")
        pgoal = self.get_body_com("goal")
        reward_ctrl = -np.square(a).sum()
        reward_dist = -np.linalg.norm(pgoal-pobj)
        ob = self._get_obs()
        done = False
        if not hasattr(self, 'itr'):
            self.itr = 0
        if not hasattr(self, 'np_random'):
            self.seed()

        reward_true = 0
        if self.itr == 0:
            self.reward_orig = -reward_dist
        if self.itr == 49:
            reward_true = reward_dist/self.reward_orig

        if 'demon_latent' in self._kwargs:
            if self.itr % 2 == 0:
                reward_latent = -np.sum((self._kwargs['demon_latent'][int(self.itr/2)]-ob[-1024:])**2)
            else:
                if self.itr == 49:
                    self.itr = 47
                reward_latent = -np.sum((self._kwargs['demon_latent'][int((self.itr+1)/2)] - ob[-1024:]) ** 2)

            if self.itr % 2 == 0:
                reward_image = -np.sum((self._kwargs['demon_image'][int(self.itr/2),:] - self.img_obs)**2)
            else:
                if self.itr == 49:
                    self.itr = 47
                reward_image = -np.sum((self._kwargs['demon_image'][int((self.itr+1)/2),:] - self.img_obs) ** 2)

            reward_ifo = (reward_latent + self._kwargs['lambda_img']*reward_image)

        else:
            reward_ifo = reward_true
        self.itr += 1
        return ob, reward_ifo, done, {}

    def reset_model(self):
        self.itr = 0
        qpos = self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq) + self.init_qpos
        # qpos = self.init_qpos
        while True:
            object_ = [np.random.uniform(low=-1.0, high=-0.4),
                       np.random.uniform(low=0.3, high=1.2)]
            goal = [np.random.uniform(low=-1.2, high=-0.8),
                    np.random.uniform(low=0.8, high=1.2)]

            if np.linalg.norm(np.array(object_)-np.array(goal)) > 0.45: break
        self.object = np.array(object_)
        self.goal = np.array(goal)

        if hasattr(self, "_kwargs") and 'goal' in self._kwargs:
            self.object = np.array(self._kwargs['object'])
            self.goal = np.array(self._kwargs['goal'])

        rgbatmp = np.copy(self.model.geom_rgba)
        geompostemp = np.copy(self.model.geom_pos)
        for body in range(5):
            body = body+1
            if 'object' in str(self.model.geom_names[body-1]):
                pos_x = np.random.uniform(low=-0.9, high=0.9)
                pos_y = np.random.uniform(low=0, high=1.0)
                rgba = self.getcolor()
                isinv = np.random.random()
                if isinv > 0.5:
                    rgba[-1] = 0.
                rgbatmp[body, :] = rgba
                geompostemp[body, 0] = pos_x
                geompostemp[body, 1] = pos_y

        if hasattr(self, "_kwargs") and 'geomspos' in self._kwargs:
            geomspos = self._kwargs['geomspos']
            for body in range(5):
                body = body + 1
                if 'object' in str(self.model.geom_names[body - 1]):
                    geompostemp[body, 0] = geomspos[body][0]
                    geompostemp[body, 1] = geomspos[body][1]
        if hasattr(self, "_kwargs") and 'geomsrgba' in self._kwargs:
            geomsrgba = self._kwargs['geomsrgba']
            for body in range(5):
                body = body + 1
                if 'object' in str(self.model.geom_names[body - 1]):
                    rgbatmp[body, :] = geomsrgba[body]

        self.model.geom_rgba[:] = rgbatmp
        self.model.geom_pos[:] = geompostemp

        self.view1_angle = np.random.uniform(low=0, high=np.pi/2)
        # view1_angle = np.pi/4
        qpos[:2] = 4 * np.array([np.cos(self.view1_angle), -np.sin(self.view1_angle)])
        # qpos[3:7] = np.array([0,0,0,1])
        qpos[-4:-2] = self.object
        qpos[-2:] = self.goal
        qvel = self.init_qvel
        qvel[-4:] = 0
        self.set_state(qpos, qvel)
        # self.set_state(self.init_qpos,self.init_qvel)
        return self._get_obs()

    def _get_obs(self):
        if not hasattr(self, 'np_random'):
            self.seed()
        if not hasattr(self, 'object'):
            self.reset()
        if 'enc_path' in self._kwargs:
            self.img_obs = self.render('rgb_array', width=48, height=48)/127.5 - 1
            z, _ = self.enc(np.reshape(self.img_obs, (-1, 48, 48, 3)))
            obs = np.concatenate([
                self.sim.data.qpos.flat[7:-4],
                self.sim.data.qvel.flat[6:-4],
                np.reshape(z, (-1)),])
        else:
            obs = np.concatenate([
                self.sim.data.qpos.flat[7:-4],
                self.sim.data.qvel.flat[6:-4],])

        return obs

    def viewer_setup(self):
        self.itr = 0
        self.viewer.cam.trackbodyid = 0
        self.viewer.cam.distance = 4.0
        rotation_angle = np.random.uniform(low=-0, high=360)
        if hasattr(self, "_kwargs") and 'vp' in self._kwargs:
            rotation_angle = self._kwargs['vp']
        cam_dist = 4
        cam_pos = np.array([0, 0, 0, cam_dist, -45, rotation_angle])
        for i in range(3):
            self.viewer.cam.lookat[i] = cam_pos[i]
        self.viewer.cam.distance = cam_pos[3]
        self.viewer.cam.elevation = cam_pos[4]
        self.viewer.cam.azimuth = cam_pos[5]
        self.viewer.cam.trackbodyid = -1

    def getcolor(self):
        color = np.random.uniform(low=0, high=1, size=3)
        while np.linalg.norm(color - np.array([1., 0., 0.])) < 0.5:
            color = np.random.uniform(low=0, high=1, size=3)
        return np.concatenate((color, [1.0]))



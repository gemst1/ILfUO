import gym
import numpy as np
from gym import utils
from gym.envs.mujoco.mujoco_env import MujocoEnv

class PusherEnv3DOF(MujocoEnv, utils.EzPickle):
    def __init__(self, **kwargs):
        utils.EzPickle.__init__(self)
        self._kwargs = kwargs
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

        self.itr += 1
        return ob, reward_true, done, {}

    def reset_model(self):
        self.itr = 0
        qpos = self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq) + self.init_qpos
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

        if hasattr(self, "_kwargs") and 'geoms' in self._kwargs:
            geoms = self._kwargs['geoms']
            for body in range(5):
                body = body + 1
                if 'object' in str(self.model.geom_names[body - 1]):
                    rgbatmp[body, :] = geoms[body][0]
                    geompostemp[body, 0] = geoms[body][1]
                    geompostemp[body, 1] = geoms[body][2]

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
        return np.concatenate([
            self.sim.data.qpos.flat[7:-4],
            self.sim.data.qvel.flat[7:-4],
            self.get_body_com("distal_4"),
            self.get_body_com("object"),
            self.get_body_com("goal"),
        ])

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



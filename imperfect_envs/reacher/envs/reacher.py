import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
import gym
from gym import spaces
import os
import random

class ReacherCustomEnv(mujoco_env.MuJocoPyEnv, utils.EzPickle):
    metadata = {'render_modes': ['human', 'rgb_array', 'depth_array'], 'render_fps': 50}
    def __init__(self, config_file='reacher.xml', **kwargs):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        utils.EzPickle.__init__(self)
        # print("fullpath is here ", self.fullpath)
        # self._initialize_simulation()
        self.observation_space = spaces.Box(low = -np.inf, high = np.inf, shape=(11,), dtype=np.float32)
        # self.action_space = spaces.Box(low = -np.inf, high = np.inf, shape=(2,), dtype=np.float32)
        mujoco_env.MuJocoPyEnv.__init__(self, ('%s/assets/'+config_file) % dir_path, 2, self.observation_space, **kwargs)

    def step(self, a):
        vec = self.get_body_com("fingertip")-self.get_body_com("target")
        reward_dist = - np.linalg.norm(vec)
        reward_ctrl = - np.square(a).sum()
        reward = reward_dist + reward_ctrl
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        done = False
        reward_for_eval = reward_dist * 10# - np.sqrt(self.sim.data.qvel.flat[0]**2+self.sim.data.qvel.flat[1]**2) / 20.

        return ob, reward, done, False, dict(reward_dist=reward_dist, reward_ctrl=reward_ctrl, reward_eval=reward_for_eval)

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 0

    def reset_with_obs(self, obs):
        self.sim.reset()
        qpos = np.array([0., 0., 0., 0.])
        self.goal = obs[4:6]
        qpos[-2:] = self.goal
        qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        qvel[-2:] = 0
        qvel[0:2] = obs[6:8]
        self.set_state(qpos, qvel)
        return self._get_obs()

    def reset_model(self):
        #self.close_goal = False
        #qpos = self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq) + self.init_qpos
        #while True:
        #    self.goal = self.np_random.uniform(low=-.2, high=.2, size=2)
        #    if np.linalg.norm(self.goal) < 0.2:
        #        break
        qpos = np.array([0., 0., 0., 0.])
        self.goal = np.concatenate([self.np_random.uniform(low=-.1, high=.1, size=1),
                                    self.np_random.uniform(low=-.2, high=-.1, size=1) if self.np_random.uniform(low=0, high=1., size=1)[0]>0.5 else self.np_random.uniform(low=.1, high=.2, size=1)])
        qpos[-2:] = self.goal
        qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        qvel[-2:] = 0
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        theta = self.sim.data.qpos.flat[:2]
        return np.concatenate([
            np.cos(theta),
            np.sin(theta),
            self.sim.data.qpos.flat[2:],
            self.sim.data.qvel.flat[:2],
            self.get_body_com("fingertip") - self.get_body_com("target")
        ])

class ReacherCustomAction1Env(ReacherCustomEnv):
    def __init__(self, **kwargs):
        super(ReacherCustomAction1Env, self).__init__('reacher_action1.xml', **kwargs)

class ReacherCustomRAction1Env(ReacherCustomEnv):
    def __init__(self, **kwargs):
        super(ReacherCustomRAction1Env, self).__init__('reacher_action1.xml', **kwargs)
        self.action_space = gym.spaces.Box(low=np.array([-1.,-1.]).astype('float32'), high=np.array([0.,0.]).astype('float32'))

    def step(self, a):
        a = np.clip(a, -1., 0.)
        return super(ReacherCustomRAction1Env, self).step(a)

class ReacherCustomAction2Env(ReacherCustomEnv):
    def __init__(self, **kwargs):
        super(ReacherCustomAction2Env, self).__init__('reacher_action2.xml', **kwargs)

class ReacherCustomRAction2Env(ReacherCustomEnv):
    def __init__(self, **kwargs):
        super(ReacherCustomRAction2Env, self).__init__('reacher_action2.xml', kwargs)
        self.action_space = gym.spaces.Box(low=np.array([0.,0.]).astype('float32'), high=np.array([1.,1.]).astype('float32'))

    def step(self, a):
        a = np.clip(a, 0., 1.)
        return super(ReacherCustomRAction2Env, self).step(a)


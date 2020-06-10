from collections import OrderedDict
import random

import ipdb
from matplotlib import pyplot as plt
import numpy as np
from gym.spaces import Box, Dict

from multiworld.envs.env_util import get_stat_in_paths, \
    create_stats_ordered_dict, get_asset_full_path
from multiworld.core.multitask_env import MultitaskEnv
from multiworld.envs.mujoco.sawyer_xyz.base import SawyerXYZEnv


class SawyerPushAndReachXYZEnv(MultitaskEnv, SawyerXYZEnv):
    def __init__(
            self,
            puck_low=(-.4, .2),
            puck_high=(.4, 1),

            reward_type='touch_distance',
            norm_order=1,
            indicator_threshold=0.06,
            touch_threshold=0.1,  # I just chose this number after doing a few runs and looking at a histogram

            hand_low=(-0.28, 0.3, 0.05),
            hand_high=(0.28, 0.9, 0.3),

            fix_goal=True,
            fixed_goal=(0.15, 0.6, 0.02, -0.15, 0.6),
            goal_low=(-0.25, 0.3, 0.02, -0.25, .4),
            goal_high=(0.25, 0.875, 0.02, .2, .8),

            hide_goal_markers=False,
            init_puck_z=0.02,
            init_hand_xyz=(0, 0.4, 0.07),

            reset_free=False,
            xml_path='sawyer_xyz/sawyer_push_puck.xml',
            clamp_puck_on_step=False,

            puck_radius=.07,
            **kwargs
    ):
        self.quick_init(locals())
        self.model_name = get_asset_full_path(xml_path)

        self.goal_type = kwargs.pop('goal_type', 'puck')
        self.dense_reward = kwargs.pop('dense_reward', False)
        self.indicator_threshold = kwargs.pop('goal_tolerance', indicator_threshold)
        self.fixed_goal = np.array(kwargs.pop('goal', fixed_goal))

        MultitaskEnv.__init__(self)
        SawyerXYZEnv.__init__(
            self,
            hand_low=hand_low,
            hand_high=hand_high,
            model_name=self.model_name,
            **kwargs
        )
        if puck_low is None:
            puck_low = self.hand_low[:2]
        if puck_high is None:
            puck_high = self.hand_high[:2]

        self.puck_low = np.array(puck_low)
        self.puck_high = np.array(puck_high)

        if goal_low is None:
            goal_low = np.hstack((self.hand_low, puck_low))
        if goal_high is None:
            goal_high = np.hstack((self.hand_high, puck_high))
        self.goal_low = np.array(goal_low)
        self.goal_high = np.array(goal_high)

        self.reward_type = reward_type
        self.norm_order = norm_order
        self.touch_threshold = touch_threshold
        self.fix_goal = fix_goal

        self._state_goal = None

        self.hide_goal_markers = hide_goal_markers

        self.action_space = Box(np.array([-1, -1, -1]), np.array([1, 1, 1]), dtype=np.float32)
        self.hand_and_puck_space = Box(
            np.hstack((self.hand_low, puck_low)),
            np.hstack((self.hand_high, puck_high)),
            dtype=np.float32
        )
        self.hand_space = Box(self.hand_low, self.hand_high, dtype=np.float32)
        self.observation_space = Dict([
            ('observation', self.hand_and_puck_space),
            ('desired_goal', self.hand_and_puck_space),
            ('achieved_goal', self.hand_and_puck_space),
            ('state_observation', self.hand_and_puck_space),
            ('state_desired_goal', self.hand_and_puck_space),
            ('state_achieved_goal', self.hand_and_puck_space),
            ('proprio_observation', self.hand_space),
            ('proprio_desired_goal', self.hand_space),
            ('proprio_achieved_goal', self.hand_space),
        ])
        self.init_puck_z = init_puck_z
        self.init_hand_xyz = np.array(init_hand_xyz)
        self._set_puck_xy(self.sample_puck_xy())
        self.reset_free = reset_free
        self.reset_counter = 0
        self.puck_space = Box(self.puck_low, self.puck_high, dtype=np.float32)
        self.clamp_puck_on_step = clamp_puck_on_step
        self.puck_radius = puck_radius
        self.reset()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 0
        self.viewer.cam.lookat[0] = 0
        self.viewer.cam.lookat[1] = 1.0
        self.viewer.cam.lookat[2] = 0.5
        self.viewer.cam.distance = 0.3
        self.viewer.cam.elevation = -45
        self.viewer.cam.azimuth = 270
        self.viewer.cam.trackbodyid = -1

    def step(self, action):
        self.set_xyz_action(action)
        u = np.zeros(8)
        u[7] = 1
        self.do_simulation(u)
        if self.clamp_puck_on_step:
            curr_puck_pos = self.get_puck_pos()[:2]
            curr_puck_pos = np.clip(curr_puck_pos, self.puck_space.low, self.puck_space.high)
            self._set_puck_xy(curr_puck_pos)
        self._set_goal_marker(self._state_goal)
        ob = self._get_obs()
        reward = self.compute_reward(action, ob)
        info = self._get_info()
        done = self.is_goal_state(ob['observation'])
        return ob, reward, done, info

    def _get_obs(self):
        e = self.get_endeff_pos()
        b = self.get_puck_pos()[:2]
        flat_obs = np.concatenate((e, b))

        return dict(
            observation=flat_obs,
            desired_goal=self._state_goal,
            achieved_goal=flat_obs,
            state_observation=flat_obs,
            state_desired_goal=self._state_goal,
            state_achieved_goal=flat_obs,
            proprio_observation=flat_obs[:3],
            proprio_desired_goal=self._state_goal[:3],
            proprio_achieved_goal=flat_obs[:3],
        )

    def _get_info(self):
        hand_goal = self._state_goal[:3]
        puck_goal = self._state_goal[3:]

        # hand distance
        hand_diff = hand_goal - self.get_endeff_pos()
        hand_distance = np.linalg.norm(hand_diff, ord=self.norm_order)
        hand_distance_l1 = np.linalg.norm(hand_diff, 1)
        hand_distance_l2 = np.linalg.norm(hand_diff, 2)

        # puck distance
        puck_diff = puck_goal - self.get_puck_pos()[:2]
        puck_distance = np.linalg.norm(puck_diff, ord=self.norm_order)
        puck_distance_l1 = np.linalg.norm(puck_diff, 1)
        puck_distance_l2 = np.linalg.norm(puck_diff, 2)

        # touch distance
        touch_diff = self.get_endeff_pos() - self.get_puck_pos()
        touch_distance = np.linalg.norm(touch_diff, ord=self.norm_order)
        touch_distance_l1 = np.linalg.norm(touch_diff, ord=1)
        touch_distance_l2 = np.linalg.norm(touch_diff, ord=2)

        # state distance
        state_diff = np.hstack((self.get_endeff_pos(), self.get_puck_pos()[:2])) - self._state_goal
        state_distance = np.linalg.norm(state_diff, ord=self.norm_order)
        state_distance_l1 = np.linalg.norm(state_diff, ord=1)
        state_distance_l2 = np.linalg.norm(state_diff, ord=2)

        return dict(
            hand_distance=hand_distance,
            hand_distance_l1=hand_distance_l1,
            hand_distance_l2=hand_distance_l2,
            puck_distance=puck_distance,
            puck_distance_l1=puck_distance_l1,
            puck_distance_l2=puck_distance_l2,
            hand_and_puck_distance=hand_distance+puck_distance,
            hand_and_puck_distance_l1=hand_distance_l1+puck_distance_l1,
            hand_and_puck_distance_l2=hand_distance_l2+puck_distance_l2,
            touch_distance=touch_distance,
            touch_distance_l1=touch_distance_l1,
            touch_distance_l2=touch_distance_l2,
            state_distance=state_distance,
            state_distance_l1=state_distance_l1,
            state_distance_l2=state_distance_l2,
            hand_success=float(hand_distance < self.indicator_threshold),
            puck_success=float(puck_distance < self.indicator_threshold),
            hand_and_puck_success=float(
                hand_distance+puck_distance < self.indicator_threshold
            ),
            touch_success=float(touch_distance < self.indicator_threshold),
            state_success=float(state_distance < self.indicator_threshold),
        )

    def is_goal_state(self, state):
        """
        Only used by deep skill chaining.
        Args:
            state (np.ndarray): state array [endeff_x, endeff_x, endeff_x, puck_x, puck_y]
        Returns:
            True if is goal state, false otherwise
        """
        assert isinstance(state, np.ndarray), type(state)
        hand_goal = self._state_goal[:3]
        puck_goal = self._state_goal[3:]
        hand_pos = state[:3]
        puck_pos = state[3:]
        tolerance = self.indicator_threshold

        if self.goal_type == 'hand':
            dist = hand_pos - hand_goal
        elif self.goal_type == 'puck':
            dist = puck_pos - puck_goal
        elif self.goal_type == 'touch':
            tolerance = self.touch_threshold
            three_d_puck_pos = np.append(puck_pos, self.init_puck_z)
            dist = hand_pos - three_d_puck_pos
        elif self.goal_type == 'state':
            dist = state - self._state_goal
        else:
            raise NotImplementedError("Invalid/no reward type.")
        return np.linalg.norm(dist, ord=self.norm_order) < tolerance

    def get_puck_pos(self):
        return self.data.get_body_xpos('puck').copy()

    def sample_puck_xy(self):
        return np.array([0, 0.6])

    def _set_goal_marker(self, goal):
        """
        This should be use ONLY for visualization. Use self._state_goal for
        logging, learning, etc.
        """
        self.data.site_xpos[self.model.site_name2id('hand-goal-site')] = (
            goal[:3]
        )
        self.data.site_xpos[self.model.site_name2id('puck-goal-site')][:2] = (
            goal[3:]
        )
        if self.hide_goal_markers:
            self.data.site_xpos[self.model.site_name2id('hand-goal-site'), 2] = (
                -1000
            )
            self.data.site_xpos[self.model.site_name2id('puck-goal-site'), 2] = (
                -1000
            )

    def _set_puck_xy(self, pos):
        """
        WARNING: this resets the sites (because set_state resets sights do).
        """
        qpos = self.data.qpos.flat.copy()
        qvel = self.data.qvel.flat.copy()
        qpos[8:11] = np.hstack((pos.copy(), np.array([self.init_puck_z])))
        qpos[11:15] = np.array([1, 0, 0, 0])
        qvel[8:15] = 0
        self.set_state(qpos, qvel)

    def reset_model(self):
        self._reset_hand()
        if not self.reset_free:
            self._set_puck_xy(self.sample_puck_xy())

        if not (self.puck_space.contains(self.get_puck_pos()[:2])):
            self._set_puck_xy(self.sample_puck_xy())

        goal = self.sample_valid_goal()
        self.set_goal(goal)
        self.reset_counter += 1
        self.reset_mocap_welds()
        return self._get_obs()

    def _reset_hand(self):
        velocities = self.data.qvel.copy()
        angles = self.data.qpos.copy()
        angles[:7] = self.init_angles[:7]
        self.set_state(angles.flatten(), velocities.flatten())
        for _ in range(10):
            self.data.set_mocap_pos('mocap', self.init_hand_xyz.copy())
            self.data.set_mocap_quat('mocap', np.array([1, 0, 1, 0]))

    def reset(self):
        ob = self.reset_model()
        if self.viewer is not None:
            self.viewer_setup()
        return ob

    @property
    def init_angles(self):
        return [1.7244448, -0.92036369,  0.10234232,  2.11178144,  2.97668632, -0.38664629, 0.54065733,
                5.05442647e-04, 6.00496057e-01, 3.06443862e-02,
                1, 0, 0, 0]

    """
    Multitask functions
    """
    def get_goal(self):
        return {
            'desired_goal': self._state_goal,
            'state_desired_goal': self._state_goal,
        }

    def set_goal(self, goal):
        self._state_goal = goal['state_desired_goal']
        self._set_goal_marker(self._state_goal)

    def set_to_goal(self, goal):
        hand_goal = goal['state_desired_goal'][:3]
        for _ in range(10):
            self.data.set_mocap_pos('mocap', hand_goal)
            self.data.set_mocap_quat('mocap', np.array([1, 0, 1, 0]))
            self.do_simulation(None, self.frame_skip)
        puck_goal = goal['state_desired_goal'][3:]
        self._set_puck_xy(puck_goal)
        self.sim.forward()

    def sample_valid_goal(self):
        goal = self.sample_goal()
        hand_goal_xy = goal['state_desired_goal'][:2]
        puck_goal_xy = goal['state_desired_goal'][3:]
        dist = np.linalg.norm(hand_goal_xy-puck_goal_xy)
        while(dist<=self.puck_radius):
            goal = self.sample_goal()
            hand_goal_xy = goal['state_desired_goal'][:2]
            puck_goal_xy = goal['state_desired_goal'][3:]
            dist = np.linalg.norm(hand_goal_xy - puck_goal_xy)
        return goal

    def sample_goals(self, batch_size):
        if self.fix_goal:
            goals = np.repeat(
                self.fixed_goal.copy()[None],
                batch_size,
                0
            )
        else:
            goals = np.random.uniform(
                self.goal_low,
                self.goal_high,
                size=(batch_size, self.goal_low.size),
            )
        return {
            'desired_goal': goals,
            'state_desired_goal': goals,
        }

    def compute_rewards(self, actions, obs):
        achieved_goals = obs['state_achieved_goal']
        desired_goals = obs['state_desired_goal']
        hand_pos = achieved_goals[:, :3]
        puck_pos = achieved_goals[:, 3:]
        hand_goals = desired_goals[:, :3]
        puck_goals = desired_goals[:, 3:]
        threshold = self.indicator_threshold

        if self.goal_type == 'hand':
            dist = np.linalg.norm(hand_goals - hand_pos, ord=self.norm_order, axis=1)
        elif self.goal_type == 'puck':
            dist = np.linalg.norm(puck_goals - puck_pos, ord=self.norm_order, axis=1)
        elif self.goal_type == 'touch':
            threshold = self.touch_threshold
            puck_zs = self.init_puck_z * np.ones((desired_goals.shape[0], 1))
            dist = np.linalg.norm(hand_pos - np.hstack((puck_pos, puck_zs)), ord=self.norm_order,axis=1,)
        elif self.goal_type == 'state':
            dist = np.linalg.norm(achieved_goals - desired_goals, ord=self.norm_order, axis=1)
        else:
            raise NotImplementedError("Invalid/no reward type.")

        if self.dense_reward:
            return -dist
        else:
            return -(dist > threshold).astype(float)


    def get_diagnostics(self, paths, prefix=''):
        statistics = OrderedDict()
        for stat_name in [
            'hand_distance',
            'hand_distance_l1',
            'hand_distance_l2',
            'puck_distance',
            'puck_distance_l1',
            'puck_distance_l2',
            'hand_and_puck_distance',
            'hand_and_puck_distance_l1',
            'hand_and_puck_distance_l2',
            'state_distance',
            'state_distance_l1',
            'state_distance_l2',
            'touch_distance',
            'touch_distance_l1',
            'touch_distance_l2',
            'hand_success',
            'puck_success',
            'hand_and_puck_success',
            'state_success',
            'touch_success',
        ]:
            stat_name = stat_name
            stat = get_stat_in_paths(paths, 'env_infos', stat_name)
            statistics.update(create_stats_ordered_dict(
                '%s%s' % (prefix, stat_name),
                stat,
                always_show_all_stats=True,
                ))
            statistics.update(create_stats_ordered_dict(
                'Final %s%s' % (prefix, stat_name),
                [s[-1] for s in stat],
                always_show_all_stats=True,
                ))
        return statistics

    def get_env_state(self):
        base_state = super().get_env_state()
        goal = self._state_goal.copy()
        return base_state, goal

    def set_env_state(self, state):
        base_state, goal = state
        super().set_env_state(base_state)
        self._state_goal = goal
        self._set_goal_marker(goal)


class SawyerPushAndReachXYEnv(SawyerPushAndReachXYZEnv):
    def __init__(self, *args, hand_z_position=0.05, **kwargs):
        self.quick_init(locals())
        SawyerPushAndReachXYZEnv.__init__(self, *args, **kwargs)
        self.hand_z_position = hand_z_position
        self.action_space = Box(np.array([-1, -1]), np.array([1, 1]), dtype=np.float32)
        self.fixed_goal[2] = hand_z_position
        hand_and_puck_low = self.hand_and_puck_space.low.copy()
        hand_and_puck_low[2] = hand_z_position
        hand_and_puck_high = self.hand_and_puck_space.high.copy()
        hand_and_puck_high[2] = hand_z_position
        self.hand_and_puck_space = Box(hand_and_puck_low, hand_and_puck_high, dtype=np.float32)
        self.observation_space = Dict([
            ('observation', self.hand_and_puck_space),
            ('desired_goal', self.hand_and_puck_space),
            ('achieved_goal', self.hand_and_puck_space),
            ('state_observation', self.hand_and_puck_space),
            ('state_desired_goal', self.hand_and_puck_space),
            ('state_achieved_goal', self.hand_and_puck_space),
            ('proprio_observation', self.hand_space),
            ('proprio_desired_goal', self.hand_space),
            ('proprio_achieved_goal', self.hand_space),
        ])

    def step(self, action):
        delta_z = self.hand_z_position - self.data.mocap_pos[0, 2]
        action = np.hstack((action, delta_z))
        return super().step(action)


if __name__ == '__main__':
    env = SawyerPushAndReachXYEnv(goal_type='puck', dense_reward=False)
    for i in range(5000):
        ob, reward, done, info = env.step([random.uniform(-1, 1), random.uniform(-1, 1)])
        env.render()
        # ob, reward, done, info = env.step([0, 0])

    # print(np.round(ob['observation'], 3))

    # ob, reward, done, info = env.step([0.9064628, -0.8112495])
    # print(ob['observation'])
    # print(env._get_obs()['state_observation'])
    #     env.render()

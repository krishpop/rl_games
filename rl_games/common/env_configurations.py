from rl_games.common import wrappers
from rl_games.common import tr_helpers

import gym
from gym.wrappers import FlattenObservation, FilterObservation

import numpy as np
from numpy.random import default_rng

import enum

'''import rrc_simulation
from rrc_simulation.gym_wrapper.envs import cube_env
from rrc_simulation.tasks import move_cube
from rrc_simulation import visual_objects
from rrc_simulation import TriFingerPlatform


class FlatObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        low = [
            self.observation_space[name].low.flatten()
            for name in self.observation_names
        ]

        high = [
            self.observation_space[name].high.flatten()
            for name in self.observation_names
        ]

        self.observation_space = gym.spaces.Box(
            low=np.concatenate(low), high=np.concatenate(high)
        )

    def observation(self, obs):
        observation = [obs[name].flatten() for name in self.observation_names]

        observation = np.concatenate(observation)
        return observation


class SimplePushingTrainingEnv(gym.Env):
    """Gym environment for moving cubes with simulated TriFingerPro."""

    def __init__(
        self,
        initializer=None,
        action_type=cube_env.ActionType.POSITION,
        reward_type=0,
        frameskip=1,
        visualization=False,
        episode_length = 3750,
    ):
        """Initialize.

        Args:
            initializer: Initializer class for providing initial cube pose and
                goal pose. If no initializer is provided, we will initialize in a way 
                which is be helpful for learning.
            action_type (ActionType): Specify which type of actions to use.
                See :class:`ActionType` for details.
            frameskip (int):  Number of actual control steps to be performed in
                one call of step().
            visualization (bool): If true, the pyBullet GUI is run for
                visualization.
        """
        # Basic initialization
        # ====================
        
        self.initializer = initializer
        self.action_type = action_type
        self.visualization = visualization
        self.reward_type = reward_type
        self.goal_dist = 1.0
        self.success_dist = 0.005
        self.success_bonus = 250.0
        self.eps = 0.01

        self.rng = default_rng()

        if frameskip < 1:
            raise ValueError("frameskip cannot be less than 1.")
        self.frameskip = frameskip

        self.episode_length = episode_length
        move_cube.episode_length = episode_length

        # will be initialized in reset()
        self.platform = None

        # Create the action and observation spaces
        # ========================================

        spaces = TriFingerPlatform.spaces

        if self.action_type == cube_env.ActionType.TORQUE:
            self.action_space = spaces.robot_torque.gym
        elif self.action_type == cube_env.ActionType.POSITION:
            self.action_space = spaces.robot_position.gym
        elif self.action_type == cube_env.ActionType.TORQUE_AND_POSITION:
            self.action_space = gym.spaces.Dict(
                {
                    "torque": spaces.robot_torque.gym,
                    "position": spaces.robot_position.gym,
                }
            )
        else:
            raise ValueError("Invalid action_type")

        self.observation_names = [
            "robot_position",
            "robot_velocity",
            "robot_tip_positions",
            "object_position",
            "object_orientation",
            "goal_object_position",
        ]

        self.observation_space = gym.spaces.Dict(
            {
                "robot_position": spaces.robot_position.gym,
                "robot_velocity": spaces.robot_velocity.gym,
                "robot_tip_positions": gym.spaces.Box(
                    low=np.array([spaces.object_position.low] * 3),
                    high=np.array([spaces.object_position.high] * 3),
                ),
                "object_position": spaces.object_position.gym,
                "object_orientation": spaces.object_orientation.gym,
                "goal_object_position": spaces.object_position.gym,
            }
        )

    def step(self, action):
        if self.platform is None:
            raise RuntimeError("Call `reset()` before starting to step.")

        if not self.action_space.contains(action):
            raise ValueError(
                "Given action is not contained in the action space."
            )

        num_steps = self.frameskip

        # ensure episode length is not exceeded due to frameskip
        step_count_after = self.step_count + num_steps
        if step_count_after > move_cube.episode_length:
            excess = step_count_after - move_cube.episode_length
            num_steps = max(1, num_steps - excess)

        reward = 0.0
        for _ in range(num_steps):
            self.step_count += 1
            if self.step_count > move_cube.episode_length:
                raise RuntimeError("Exceeded number of steps for one episode.")

            # send action to robot
            robot_action = self._gym_action_to_robot_action(action)
            t = self.platform.append_desired_action(robot_action)

            previous_observation = self._create_observation(t)
            observation = self._create_observation(t + 1)

            if self.reward_type == 0:
                reward += self._compute_reward0(
                    previous_observation=previous_observation,
                    observation=observation,
                    success_dist=self.success_dist,
                    success_bonus=self.success_bonus,
                    eps=self.eps
                )
            elif self.reward_type == 1:
                reward += self._compute_reward1(
                    previous_observation=previous_observation,
                    observation=observation,
                    success_dist=self.success_dist,
                    success_bonus=self.success_bonus,
                    eps=self.eps
                )
            else:
                reward += self._compute_reward0(
                    previous_observation=previous_observation,
                    observation=observation,
                    success_dist=self.success_dist,
                    success_bonus=self.success_bonus,
                    eps=self.eps
                )

        self.goal_dist = np.linalg.norm(observation["goal_object_position"]
                        - observation["object_position"])

        # todo add new conditions
        is_done = (self.step_count >= move_cube.episode_length)
        #is_done = (self.step_count == move_cube.episode_length) or (self.goal_dist <= self.success_dist)

        return observation, reward, is_done, self.info

    def reset(self):
        # reset simulation
        del self.platform

        default_robot_position = TriFingerPlatform.spaces.robot_position.default
        #print("Robot default pos: ", default_robot_position)
        #print("Robot spaces: ", TriFingerPlatform.spaces)

        self.goal_dist = 1.0

        # initialize simulation
        if self.initializer is None:
            # if no initializer is given (which will be the case during training),
            # we can initialize in any way desired. here, we initialize the cube always
            # in the center of the arena, instead of randomly, as this appears to help 
            # training

            default_object_position = (
                TriFingerPlatform.spaces.object_position.default
            )
            default_object_orientation = (
                TriFingerPlatform.spaces.object_orientation.default
            )
            init_object_position = self.rng.uniform(default_object_position - 0.01, default_object_position + 0.01)
            initial_object_pose = move_cube.Pose(
                position=init_object_position,
                orientation=default_object_orientation,
            )
            
            goal_object_pose = move_cube.sample_goal(difficulty=1)   
        else:
            # if an initializer is given, i.e. during evaluation, we need to initialize
            # according to it, to make sure we remain coherent with the standard CubeEnv.
            # otherwise the trajectories produced during evaluation will be invalid.
            initial_object_pose = self.initializer.get_initial_state()
            goal_object_pose = self.initializer.get_goal()
        
        # todo in a more controlled way
        initial_robot_position = self.rng.uniform(default_robot_position - 0.1, default_robot_position + 0.1)
        #print("Robot reset pos: ", initial_robot_position)
            
        self.platform = TriFingerPlatform(
            visualization=self.visualization,
            initial_robot_position=initial_robot_position,
            initial_object_pose=initial_object_pose,
        )

        self.goal = {
            "position": goal_object_pose.position,
            "orientation": goal_object_pose.orientation,
        }

        # visualize the goal
        if self.visualization:
            self.goal_marker = visual_objects.CubeMarker(
                width=0.065,
                position=goal_object_pose.position,
                orientation=goal_object_pose.orientation,
            )

        self.info = dict()
        self.step_count = 0

        return self._create_observation(0)

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        move_cube.random = self.np_random
        return [seed]

    def _create_observation(self, t):
        robot_observation = self.platform.get_robot_observation(t)
        object_observation = self.platform.get_object_pose(t)
        robot_tip_positions = self.platform.forward_kinematics(
            robot_observation.position
        )
        robot_tip_positions = np.array(robot_tip_positions)

        observation = {
            "robot_position": robot_observation.position,
            "robot_velocity": robot_observation.velocity,
            "robot_tip_positions": robot_tip_positions,
            "object_position": object_observation.position,
            "object_orientation": object_observation.orientation,
            "goal_object_position": self.goal["position"],
        }
        return observation

    @staticmethod
    def _compute_reward0(previous_observation, observation, success_dist=0.005, success_bonus=250, eps=0.015):

        # calculate first reward term
        current_distance_from_block = np.linalg.norm(
            observation["robot_tip_positions"] - observation["object_position"]
        )
        previous_distance_from_block = np.linalg.norm(
            previous_observation["robot_tip_positions"]
            - previous_observation["object_position"]
        )

        reward_term_1 = (
            previous_distance_from_block - current_distance_from_block
        )

        # calculate second reward term
        current_dist_to_goal = np.linalg.norm(
            observation["goal_object_position"]
            - observation["object_position"]
        )
        previous_dist_to_goal = np.linalg.norm(
            previous_observation["goal_object_position"]
            - previous_observation["object_position"]
        )
        reward_term_2 = previous_dist_to_goal - current_dist_to_goal

        reward = -4.5 * current_dist_to_goal - 0.5 * current_distance_from_block
        if current_dist_to_goal < 0.005:
            reward += 1.0
            print("Success! Dist to goal = ", current_dist_to_goal)

    #    if current_dist_to_goal <= success_dist:
    #        reward += success_bonus

        return reward

    @staticmethod
    def _compute_reward1(previous_observation, observation, success_dist=0.005, success_bonus=250, eps=0.015):

        # calculate first reward term
        current_distance_from_block = np.linalg.norm(
            observation["robot_tip_positions"] - observation["object_position"]
        )
        previous_distance_from_block = np.linalg.norm(
            previous_observation["robot_tip_positions"]
            - previous_observation["object_position"]
        )

        reward_term_1 = (
            previous_distance_from_block - current_distance_from_block
        )

        # calculate second reward term
        current_dist_to_goal = np.linalg.norm(
            observation["goal_object_position"]
            - observation["object_position"]
        )
        previous_dist_to_goal = np.linalg.norm(
            previous_observation["goal_object_position"]
            - previous_observation["object_position"]
        )
        reward_term_2 = previous_dist_to_goal - current_dist_to_goal

        reward = 0.1 / (current_dist_to_goal + eps) - 2.0 * current_distance_from_block

    #    if current_dist_to_goal <= success_dist:
    #        reward += success_bonus

        return reward

    def _gym_action_to_robot_action(self, gym_action):
        # construct robot action depending on action type
        if self.action_type == cube_env.ActionType.TORQUE:
            robot_action = self.platform.Action(torque=gym_action)
        elif self.action_type == cube_env.ActionType.POSITION:
            robot_action = self.platform.Action(position=gym_action)
        elif self.action_type == cube_env.ActionType.TORQUE_AND_POSITION:
            robot_action = self.platform.Action(
                torque=gym_action["torque"], position=gym_action["position"]
            )
        else:
            raise ValueError("Invalid action_type")

        return robot_action'''

#FLEX_PATH = '/home/viktor/Documents/rl/FlexRobotics'
FLEX_PATH = '/home/trrrrr/Documents/FlexRobotics-master'


class HCRewardEnv(gym.RewardWrapper):
    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        
        self.num_stops = 0
        self.stops_decay = 0
        self.max_stops = 30

    def reset(self, **kwargs):
        self.num_stops = 0
        self.stops_decay = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        if self.num_stops > self.max_stops:
            print('too many stops!')
            reward = -100
            observation = self.reset()
            done = True
        return observation, self.reward(reward), done, info

    def reward(self, reward):
       # print('reward:', reward)
        '''
        if reward < 0.005:
            self.stops_decay = 0
            self.num_stops += 1
            #print('stops:', self.num_stops)
            return -0.1
        self.stops_decay += 1
        if self.stops_decay == self.max_stops:
            self.num_stops = 0
            self.stops_decay = 0
        '''
        return np.max([-10, reward])


class HCObsEnv(gym.ObservationWrapper):
    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)

    def observation(self, observation):
        obs = observation - [ 0.1193, -0.001 ,  0.0958, -0.0052, -0.629 , -0.01  ,  0.1604, -0.0205,
  0.7094,  0.6344,  0.0091,  0.1617, -0.0001,  0.7018,  0.4293,  0.3909,
  0.3776,  0.3662,  0.3722,  0.4043,  0.4497,  0.6033,  0.7825,  0.9575]

        obs = obs / [0.3528, 0.0501, 0.1561, 0.0531, 0.2936, 0.4599, 0.6598, 0.4978, 0.454 ,
 0.7168, 0.3419, 0.6492, 0.4548, 0.4575, 0.1024, 0.0716, 0.0918, 0.11  ,
 0.1289, 0.1501, 0.1649, 0.191 , 0.2036, 0.1095]
        obs = np.clip(obs, -5.0, 5.0)
        return obs


class DMControlReward(gym.RewardWrapper):
    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        
        self.num_stops = 0
        self.max_stops = 1000
        self.reward_threshold = 0.001

    def reset(self, **kwargs):
        self.num_stops = 0
 
        return self.env.reset(**kwargs)

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        if reward < self.reward_threshold:
            self.num_stops += 1
        else:
            self.num_stops = max(0, self.num_stops-1)
        if self.num_stops > self.max_stops:
            #print('too many stops!')
            reward = -10
            observation = self.reset()
            done = True
        return observation, self.reward(reward), done, info

    def reward(self, reward):
        return reward


class DMControlObsWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)

    def observation(self, obs):
        return obs['observations']


def create_default_gym_env(**kwargs):
    frames = kwargs.pop('frames', 1)
    name = kwargs.pop('name')
    is_procgen = kwargs.pop('procgen', False)
    limit_steps = kwargs.pop('limit_steps', False)
    env = gym.make(name, **kwargs)

    if frames > 1:
        if is_procgen:
            env = wrappers.ProcgenStack(env, frames, True)
        else:
            env = wrappers.FrameStack(env, frames, False)
    if limit_steps:
        env = wrappers.LimitStepsWrapper(env)
    return env 

def create_goal_gym_env(**kwargs):
    frames = kwargs.pop('frames', 1)
    name = kwargs.pop('name')
    limit_steps = kwargs.pop('limit_steps', False)

    env = gym.make(name, **kwargs)
    env = FlattenObservation(FilterObservation(env, ['observation', 'desired_goal']))

    if frames > 1:
        env = wrappers.FrameStack(env, frames, False)
    if limit_steps:
        env = wrappers.LimitStepsWrapper(env)
    return env 

def create_rrc_gym_env(**kwargs):
    print("Creating rrc env")

    frames = kwargs.pop('frames', 1)
    name = kwargs.pop('name')
#    limit_steps = kwargs.pop('limit_steps', False)

    initializer = cube_env.RandomInitializer(difficulty=1)
    env = SimplePushingTrainingEnv(initializer=initializer, reward_type=0, frameskip=1, visualization=True, episode_length=3750)
    env.seed(7)
    env.action_space.seed(7)
    env = FlatObservationWrapper(env)
    print(env.action_space)

    if frames > 1:
        env = wrappers.FrameStack(env, frames, False)
#    if limit_steps:
#        env = wrappers.LimitStepsWrapper(env)
    return env 

def create_atari_gym_env(**kwargs):
    #frames = kwargs.pop('frames', 1)
    name = kwargs.pop('name')
    env = wrappers.make_atari_deepmind(name, skip=4)
    return env    

def create_dm_control_env(**kwargs):
    frames = kwargs.pop('frames', 1)
    name = 'dm2gym:'+ kwargs.pop('name')
    env = gym.make(name, environment_kwargs=kwargs)
    env = DMControlReward(env)
    env = DMControlObsWrapper(env)

    if frames > 1:
        env = wrappers.FrameStack(env, frames, False)
    return env

def create_super_mario_env(name='SuperMarioBros-v1'):
    import gym
    from nes_py.wrappers import JoypadSpace
    from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT
    import gym_super_mario_bros
    env = gym_super_mario_bros.make(name)
    env = JoypadSpace(env, SIMPLE_MOVEMENT)

    env = wrappers.MaxAndSkipEnv(env, skip=4)
    env = wrappers.wrap_deepmind(env, episode_life=False, clip_rewards=False, frame_stack=True, scale=True)
    return env

def create_super_mario_env_stage1(name='SuperMarioBrosRandomStage1-v1'):
    import gym
    from nes_py.wrappers import JoypadSpace
    from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT

    import gym_super_mario_bros
    stage_names = [
        'SuperMarioBros-1-1-v1',
        'SuperMarioBros-1-2-v1',
        'SuperMarioBros-1-3-v1',
        'SuperMarioBros-1-4-v1',
    ]

    env = gym_super_mario_bros.make(stage_names[1])
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    
    env = wrappers.MaxAndSkipEnv(env, skip=4)
    env = wrappers.wrap_deepmind(env, episode_life=False, clip_rewards=False, frame_stack=True, scale=True)
    #env = wrappers.AllowBacktracking(env)
    
    return env

def create_quadrupped_env():
    import gym
    import roboschool
    import quadruppedEnv
    return wrappers.FrameStack(wrappers.MaxAndSkipEnv(gym.make('QuadruppedWalk-v1'),4, False), 2, True)

def create_roboschool_env(name):
    import gym
    import roboschool
    return gym.make(name)

def create_multiflex(path, num_instances=1):
    from flex_gym.flex_vec_env import set_flex_bin_path, make_flex_vec_env_muli_env
    from autolab_core import YamlConfig
    import gym

    set_flex_bin_path(FLEX_PATH + '/bin')

    cfg_env = YamlConfig(path)
    env = make_flex_vec_env_muli_env([cfg_env] * num_instances)

    return env

def create_flex(path):
    from flex_gym.flex_vec_env import set_flex_bin_path, make_flex_vec_env
    from autolab_core import YamlConfig
    import gym

    set_flex_bin_path(FLEX_PATH + '/bin')

    cfg_env = YamlConfig(path)
    cfg_env['gym']['rank'] = 0
    env = make_flex_vec_env(cfg_env)

    return env

def create_smac(name, **kwargs):
    from rl_games.envs.smac_env import SMACEnv
    frames = kwargs.pop('frames', 1)
    print(kwargs)
    return wrappers.BatchedFrameStack(SMACEnv(name, **kwargs), frames, transpose=False, flatten=True)

def create_smac_cnn(name, **kwargs):
    from rl_games.envs.smac_env import SMACEnv
    env = SMACEnv(name, **kwargs)
    frames = kwargs.pop('frames', 4)
    transpose = kwargs.pop('transpose', False)
    env = wrappers.BatchedFrameStack(env, frames, transpose=transpose)
    return env


configurations = {
    'CartPole-v1' : {
        'vecenv_type' : 'RAY',
        'env_creator' : lambda **kwargs : wrappers.MaskVelocityWrapper(gym.make('CartPole-v1'), 'CartPole-v1'),
    },
    'MountainCarContinuous-v0' : {
        'vecenv_type' : 'RAY',
        'env_creator' : lambda **kwargs  : gym.make('MountainCarContinuous-v0'),
    },
    'MountainCar-v0' : {
        'vecenv_type' : 'RAY',
        'env_creator' : lambda : gym.make('MountainCar-v0'),
    },
    'Acrobot-v1' : {
        'env_creator' : lambda **kwargs  : gym.make('Acrobot-v1'),
        'vecenv_type' : 'RAY'
    },
    'Pendulum-v0' : {
        'env_creator' : lambda **kwargs  : gym.make('Pendulum-v0'),
        'vecenv_type' : 'RAY'
    },
    'LunarLander-v2' : {
        'env_creator' : lambda **kwargs  : gym.make('LunarLander-v2'),
        'vecenv_type' : 'RAY'
    },
    'PongNoFrameskip-v4' : {
        'env_creator' : lambda **kwargs  :  wrappers.make_atari_deepmind('PongNoFrameskip-v4', skip=4),
        'vecenv_type' : 'RAY'
    },
    'BreakoutNoFrameskip-v4' : {
        'env_creator' : lambda  **kwargs :  wrappers.make_atari_deepmind('BreakoutNoFrameskip-v4', skip=4),
        'vecenv_type' : 'RAY'
    },
    'CarRacing-v0' : {
        'env_creator' : lambda **kwargs  :  wrappers.make_car_racing('CarRacing-v0', skip=4),
        'vecenv_type' : 'RAY'
    },
    'RoboschoolAnt-v1' : {
        'env_creator' : lambda **kwargs  : create_roboschool_env('RoboschoolAnt-v1'),
        'vecenv_type' : 'RAY'
    },
    'SuperMarioBros-v1' : {
        'env_creator' : lambda :  create_super_mario_env(),
        'vecenv_type' : 'RAY'
    },
    'SuperMarioBrosRandomStages-v1' : {
        'env_creator' : lambda :  create_super_mario_env('SuperMarioBrosRandomStages-v1'),
        'vecenv_type' : 'RAY'
    },
    'SuperMarioBrosRandomStage1-v1' : {
        'env_creator' : lambda **kwargs  :  create_super_mario_env_stage1('SuperMarioBrosRandomStage1-v1'),
        'vecenv_type' : 'RAY'
    },
    'RoboschoolHalfCheetah-v1' : {
        'env_creator' : lambda **kwargs  : create_roboschool_env('RoboschoolHalfCheetah-v1'),
        'vecenv_type' : 'RAY'
    },
    'RoboschoolHumanoid-v1' : {
        'env_creator' : lambda : wrappers.FrameStack(create_roboschool_env('RoboschoolHumanoid-v1'), 1, True),
        'vecenv_type' : 'RAY'
    },
    'LunarLanderContinuous-v2' : {
        'env_creator' : lambda **kwargs  : gym.make('LunarLanderContinuous-v2'),
        'vecenv_type' : 'RAY'
    },
    'RoboschoolHumanoidFlagrun-v1' : {
        'env_creator' : lambda **kwargs  : wrappers.FrameStack(create_roboschool_env('RoboschoolHumanoidFlagrun-v1'), 1, True),
        'vecenv_type' : 'RAY'
    },
    'BipedalWalker-v3' : {
        'env_creator' : lambda **kwargs  : gym.make('BipedalWalker-v3'),
        'vecenv_type' : 'RAY'
    },
    'BipedalWalkerCnn-v3' : {
        'env_creator' : lambda **kwargs  : wrappers.FrameStack(HCRewardEnv(gym.make('BipedalWalker-v3')), 4, False),
        'vecenv_type' : 'RAY'
    },
    'BipedalWalkerHardcore-v3' : {
        'env_creator' : lambda **kwargs  : HCRewardEnv(gym.make('BipedalWalkerHardcore-v3')),
        'vecenv_type' : 'RAY'
    },
    'BipedalWalkerHardcoreCnn-v3' : {
        'env_creator' : lambda : wrappers.FrameStack(gym.make('BipedalWalkerHardcore-v3'), 4, False),
        'vecenv_type' : 'RAY'
    },
    'QuadruppedWalk-v1' : {
        'env_creator' : lambda **kwargs  : create_quadrupped_env(),
        'vecenv_type' : 'RAY'
    },
    'FlexAnt' : {
        'env_creator' : lambda **kwargs  : create_flex(FLEX_PATH + '/demo/gym/cfg/ant.yaml'),
        'vecenv_type' : 'ISAAC'
    },
    'FlexHumanoid' : {
        'env_creator' : lambda **kwargs  : create_flex(FLEX_PATH + '/demo/gym/cfg/humanoid.yaml'),
        'vecenv_type' : 'ISAAC'
    },
    'FlexHumanoidHard' : {
        'env_creator' : lambda **kwargs  : create_flex(FLEX_PATH + '/demo/gym/cfg/humanoid_hard.yaml'),
        'vecenv_type' : 'ISAAC'
    },
    'smac' : {
        'env_creator' : lambda **kwargs : create_smac(**kwargs),
        'vecenv_type' : 'RAY_SMAC'
    },
    'smac_cnn' : {
        'env_creator' : lambda **kwargs : create_smac_cnn(**kwargs),
        'vecenv_type' : 'RAY_SMAC'
    },
    'dm_control' : {
        'env_creator' : lambda **kwargs : create_dm_control_env(**kwargs),
        'vecenv_type' : 'RAY'
    },
    'openai_gym' : {
        'env_creator' : lambda **kwargs : create_default_gym_env(**kwargs),
        'vecenv_type' : 'RAY'
    },
    'openai_robot_gym' : {
        'env_creator' : lambda **kwargs : create_goal_gym_env(**kwargs),
        'vecenv_type' : 'RAY'
    },
    'rrc_gym' : {
        'env_creator' : lambda **kwargs : create_rrc_gym_env(**kwargs),
        'vecenv_type' : 'RAY'
    },
    'atari_gym' : {
        'env_creator' : lambda **kwargs : create_atari_gym_env(**kwargs),
        'vecenv_type' : 'RAY'
    },
}


def get_obs_and_action_spaces(name):
    env = configurations[name]['env_creator']()
    observation_space = env.observation_space
    action_space = env.action_space
    env.close()
    # workaround for deepmind control
    if isinstance(observation_space, gym.spaces.dict.Dict):
        observation_space = observation_space['observations']
    return observation_space, action_space

def get_obs_and_action_spaces_from_config(config):
    env_config = config.get('env_config', {})
    env = configurations[config['env_name']]['env_creator'](**env_config)
    observation_space = env.observation_space
    action_space = env.action_space
    env.close()
    # workaround for deepmind control

    if isinstance(observation_space, gym.spaces.dict.Dict):
        observation_space = observation_space['observations']
    return observation_space, action_space

def get_env_info(env):
    observation_space = env.observation_space
    action_space = env.action_space
    agents = 1
    if hasattr(env, "get_number_of_agents"):
        agents = env.get_number_of_agents()
    if isinstance(observation_space, gym.spaces.dict.Dict):
        observation_space = observation_space['observations']
    return observation_space, action_space, agents

def register(name, config):
    configurations[name] = config
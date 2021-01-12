import gym
import numpy as np
from gym import wrappers
from gym import spaces
import math
from scipy.integrate import solve_ivp

def odefun(t, y, action):
  dydt = np.zeros((2,), dtype=float)

  dydt[0] = y[1]
  dydt[1] = action[0]
  
  return dydt

class UAV_nav(gym.Env):
  """
  UAV navigation environment. Simplest environment with no obstacles.
  """
  metadata = {'render.modes': ['console']}
  
  def __init__(self, **kwargs):
    super(UAV_nav, self).__init__()

    self.n_states = 2
    # Initialize the agent at the origin
    self.agent_pos = np.zeros((self.n_states,), dtype=np.float32)
    self.curr_time = 0.0
    self._max_episode_steps = 5000
    self.dist_coef = kwargs.pop('dist_coef', 1.0/ 200.0)
    self.vel_coef = kwargs.pop('vel_coef', 2.0)
    # Define agent variables

    self.goal = 20.0
    # Define action and observation space
    self.n_actions = 1
    low_arr = -2.0*np.ones((self.n_actions,),dtype=np.float32)
    high_arr = 2.0*np.ones((self.n_actions,),dtype=np.float32)

    self.action_space = spaces.Box(low=low_arr, high=high_arr,
                                        shape=(self.n_actions,), dtype=np.float32)

    low_arr = -200.0*np.ones((self.n_states,),dtype=np.float32)
    high_arr = 200.0*np.ones((self.n_states,),dtype=np.float32)
    self.observation_space = spaces.Box(low=low_arr, high=high_arr,
                                        shape=(self.n_states,), dtype=np.float32)

  def reset(self):
    """
    Important: the observation must be a numpy array
    :return: (np.array) 
    """
    # Initialize the agent at the (0,0)
    self.agent_pos = np.zeros((self.n_states,), dtype=np.float32)
    self.curr_time = 0.0
    return self.agent_pos

  def step(self, action):

    DEL_T = 0.1

    # Carry out 1 time step of x_ddot = a
    
    tspan = (0.0, DEL_T)
    y0 = self.agent_pos

    sol_int = solve_ivp(odefun, tspan, y0, args=(action,), t_eval=[DEL_T], method='RK45')
    
    x_new = sol_int.y[0]

    self.curr_time += DEL_T

    # Is the goal reached?
    diffx = x_new - self.goal
    diffv = sol_int.y[1] - 0
    if (abs(diffx) < 1.0):
      done = True
    else:
      done = False

    # Reward is sum of incremental decrease in position + Time penalty + Episode end bonus
    reward_dist = -np.abs(((diffx*self.dist_coef) + (diffv*self.vel_coef)))
    reward_time = -0.01
    reward_done = 0


      
    goal_reached = False
    if done and np.abs(sol_int.y[1]) < 1:
      goal_reached = True

    if goal_reached:
      reward_done = 10 - 2*np.abs(sol_int.y[1])

    reward = reward_dist + reward_done + reward_time

    if (self.curr_time >= 30.0):
      done = True

    for i in range(0,2):
      self.agent_pos[i] = sol_int.y[i]

    info = {}
    if done:
      info = {'scores' : goal_reached}

    return self.agent_pos, reward[0], done, info

  def render(self, mode='console'):
    print(self.agent_pos)

  def close(self):
    pass
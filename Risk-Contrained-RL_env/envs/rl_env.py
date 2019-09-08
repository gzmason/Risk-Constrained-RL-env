
# coding: utf-8

# In[1]:


import gym
from gym import spaces
import pandas as pd
import numpy as np


# In[ ]:


class CustomEnv(gym.Env):
  """Custom Environment that follows gym interface"""
  metadata = {'render.modes': ['human']}

    def __init__(self, df):
    super(CustomEnv, self).__init__()
    
    #df comes from the dataset (PV, required load...)
    self.df = pd.read_csv("TrainingData.csv")
    
    # Define action and observation space
    # They must be gym.spaces objects
    
    #action space: as mentioned in trials.py line 24 and problem defining, 3 actions at each step: "lt,ct,ft"
    # low and high: 3 values, each range from 0 to 1, need to change according to the dataset
    self.action_space = spaces.Box(
      low=np.array([0,0,0]), high=np.array([1,1,1]), dtype=np.float16)
    
    # observation space: as mentioned in trials.py line 108, obs.shape: [3]
    # low and high: 3 values, each range from 0 to 1
    self.observation_space = spaces.Box(
      low=np.array([0,0,0]), high=np.array([1,1,1]), dtype=np.float16)

 


# In[ ]:


def reset(self):
    # Reset the state of the environment to an initial state, can be used in "render" to show current situation
    # Can define other variables: self.VariableName
    self.energy_bought=0
    self.storage_remaining=0
    self.generator_fuel_remaining=0
    
    # Set the current step to 0 
    # According to problem definition, T=0~96, and assume each line in df corresponds to one timestep
    
    self.current_step=0
    
    #randomly select a day, jump to the start of that day (has to be multiple of 96)
    self.currentTimeStamp=random.randint(0,154)*96

    return self._next_observation()


# In[ ]:


def _next_observation(self):
    # Get the data points for the current time interval (or can include history i.e. last 5 days)
    # Need to replace Observation1 (PV) , Observation2 (D nonCritical), Observation3 (D Critical) by their names in dataset
    
    # go over each building at currentTimeStamp, find if it is critical or not, add to corresponding sum
    
    frame = np.array([
    self.df.loc[self.currentTimeStamp, 'PV'],
    self.df.loc[self.currentTimeStamp,'nonCriticalLoad'],
    self.df.loc[self.currentTimeStamp,'criticalLoad']
    ])
    
    """
    frame = np.array([
    self.df.loc[self.currentTimeStamp, 'PV'].values,
    self.df.loc[self.current_step, 'Observation2'].values,
    self.df.loc[self.current_step, 'Observation3'].values,
    ])
    """
    
    #also possible to add other parameters in obs
    """
    # Append additional data and scale each value to between 0-1
    obs = np.append(frame, [[
    self.balance / MAX_ACCOUNT_BALANCE,
    self.max_net_worth / MAX_ACCOUNT_BALANCE,
    self.shares_held / MAX_NUM_SHARES,
    self.cost_basis / MAX_SHARE_PRICE,
    self.total_shares_sold / MAX_NUM_SHARES,
    self.total_sales_value / (MAX_NUM_SHARES * MAX_SHARE_PRICE),
    ]], axis=0)
    """
    obs=frame
    return obs


# In[ ]:


def step(self, action, obs):
    #At each step we will take the specified action (chosen by our model), 
    #calculate the reward, and return the next observation.
    
    # Execute one time step within the environment
    self._take_action(action,obs)
    self.current_step += 1
    self.currentTimestamp+=1
    
    #reward according to problem definition is based on three variables in action space :lt,ct,ft and obs
    #Since it needs to use the data in obs, in trials.py line 115 we can pass "obs" as a parameter after "act"
    #Rlt denotes the reward function of lt.
    #Rft denotes the reward function of ft. (There are predetermined coefficients)
    #REt denotes the reward function of Et. (There are predetermined coefficients)
    Rlt=((1-action[0])**2)*((obs[2]+obs[1])**2)
    Rft= ALPHA + BETA * action[2] + ETA * (action[2]**2)
    # Et=D Critical + D nonCritical - PV - ct - DELTA * ft according to problem definition
    REt= PHI + OMEGA * (obs[2]+obs[1]-obs[0]-action[1]-DELTA*action[2])
    reward = Rlt+Rft+REt
    
    # In trials.py line 116, when interval reaches t_dis, the environment is supposed to return done as false
    done = (self.current_step!=96)
    
    obs = self._next_observation()
    return obs, reward, done, {}


# In[ ]:


def _take_action(self, action,obs):
    #update these values for rendering purpose
    self.energy_bought+=(obs[1]+obs[2]-obs[0]-action[1]-DELTA*action[2])
    self.storage_remaining-=action[1]
    self.generator_fuel_remaining-=action[2]
    


# In[ ]:


def render(self, mode='human', close=False):
    # Render the environment to the screen
    print(f'Step: {self.current_step}')
    print(f'External Energy Bought: {self.energy_bought}')
    print(f'Storage Remaining: {self.storage_remaining}')
    print(f'Fuel Remaining: {self.generator_fuel_remaining}')


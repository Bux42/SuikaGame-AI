import time
from gymnasium.envs.registration import register
import gymnasium as gym
from sb3_contrib import MaskablePPO
from gymnasium import spaces
import numpy as np

register(id='suika-env', entry_point='suika_env.SuikaEnv:SuikaEnv')

class SuikaEnv(gym.Env):
    metadata = {"render_modes": ["human"], 'render_fps': 5}
    
    def __init__(self, render_mode=None):
        self.model: MaskablePPO = None
        
        if(self.render_mode == 'human'):
            # self.gui = GuiV2(self)
            pass
        
        self.masking_time_array = []
        self.masking_time_array_max_len = 100
        
        total_actions = 5
        
        self.action_space = gym.spaces.Discrete(total_actions)
        
        obsSpaceLen = 10
        
        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(obsSpaceLen,),
            dtype=np.float32
        )
    
    def set_model(self, model: MaskablePPO):
        self.model = model
        
    def render(self, obs):
        # # for rendering the logits, visualizing the "score" of each action
        # if self.model is not None and not self.gui.disable_render:
        #     self.get_actions_score(obs)
            
        # self.gui.render()
        pass

    def action_masks(self) -> np.ndarray:
        """Return boolean mask of valid actions"""
        maskStartTime = time.time()
        
        mask = np.zeros(self.action_space.n, dtype=np.bool_)
        
        # EndTurn is always valid
        mask[0] = True
        
                            
        maskEndTime = time.time()
        maskTimeMs = (maskEndTime - maskStartTime) * 1000
        
        while len(self.masking_time_array) > self.masking_time_array_max_len:
            self.masking_time_array.pop(0)
        self.masking_time_array.append(maskTimeMs)
        
        self.masking_time_avg = sum(self.masking_time_array) / len(self.masking_time_array)
        
        return mask
    
    def step(self, action):
        stepStartTime = time.time()
        
        # Construct the observation state: 
        obs = self.get_observation()
        
        if self.render_mode == 'human':
            # # print("🧍 Algorithm Action:", "None" if nextAction == None else nextAction.fighActionType, "TotalDamage:", recievedDamage)
            
            self.render(obs)
            
        stepEndTime = time.time()
        stepTimeMs = (stepEndTime - stepStartTime) * 1000
        
        while len(self.step_time_array) > self.step_time_array_max_len:
            self.step_time_array.pop(0)
        self.step_time_array.append(stepTimeMs)
        
        self.step_time_avg = sum(self.step_time_array) / len(self.step_time_array)
        
        return obs, self.step_reward, terminated, False, {}
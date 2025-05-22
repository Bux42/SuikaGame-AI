import gymnasium as gym
import datetime
import os
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
import suika_env

def train_sb3(session_model_dir, session_log_dir, render_mode, num_envs, datetimeStr):
    learning_rate = 0.0003
    gae_lambda = 0.8
    # 0.0003 is MaskablePPO's default learning rate
    ent_coef = learning_rate / 1000
    
    model_n_steps = 2048
    device = 'cpu'
    
    if num_envs == 1:
        TIMESTEPS = 4000
        env = gym.make('suika_env', render_mode=render_mode)
        # env = Monitor(env)
        
        model = MaskablePPO(
            MaskableActorCriticPolicy,
            env,
            verbose=1,
            device=device,
            tensorboard_log=session_log_dir,
            learning_rate=learning_rate,
            n_steps=model_n_steps,
            gamma=0.99,
            gae_lambda=gae_lambda,
            policy_kwargs=dict(net_arch=dict(pi=[64, 64], vf=[64, 64])),
        )
    else:
        TIMESTEPS = 10_000_000
        model_n_steps = 256
        device = 'cuda'
        
        env = make_vec_env('suika_env', n_envs=num_envs, vec_env_cls=SubprocVecEnv)
        model = MaskablePPO(
            MaskableActorCriticPolicy,
            env,
            verbose=1,
            device=device,
            n_steps=model_n_steps,
            tensorboard_log=session_log_dir,
            learning_rate=learning_rate,
            gamma=0.99,
            gae_lambda=gae_lambda,
            policy_kwargs=dict(net_arch=dict(pi=[64, 64], vf=[64, 64])),
        )
    if device == 'cpu':
        env.unwrapped.set_model(model)
        
    iters = 0
    while True:
        iters += 1

        model.learn(
            total_timesteps=TIMESTEPS,
            reset_num_timesteps=False,
            # callback=tensorboardCallback,
            progress_bar=False)

def create_train_folders(root_model_dir, session_model_dir, root_log_dir, session_log_dir):
    os.makedirs(root_model_dir, exist_ok=True)
    os.makedirs(root_log_dir, exist_ok=True)
    
    os.makedirs(session_model_dir, exist_ok=True)
    os.makedirs(session_log_dir, exist_ok=True)

def start_training(render_mode, num_envs):
    currentFile = os.path.basename(__file__).split('.')[0]
    print(f"Current file: {currentFile}")
    
    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y_%Hh%Mm%Ss")
    print("date and time =", dt_string)
    
    root_model_dir = f"{currentFile}_models"
    root_log_dir = f"{currentFile}_logs"
    
    session_model_dir = f"{root_model_dir}/{dt_string}"
    session_log_dir = f"{root_log_dir}/{dt_string}"
    
    create_train_folders(root_model_dir, session_model_dir, root_log_dir, session_log_dir)
    train_sb3(session_model_dir, session_log_dir, render_mode, num_envs, dt_string)


if __name__ == '__main__':
    start_training(render_mode="human", num_envs=1)
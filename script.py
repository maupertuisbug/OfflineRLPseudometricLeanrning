from twindelayedddpg.core import TD3Agent 
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import wandb
from omegaconf import OmegaConf
from tqdm import tqdm
import gymnasium as gym
import argparse
import gc


def run_exp():
    run = wandb.init()
    wandb_config = wandb.config

    seed = wandb_config.seed 
    env_name = str(wandb_config.env_name)
    env_id = str(wandb_config.env_id)
    train_epochs = int(wandb_config.train_epochs)
    eval_freq = int(wandb_config.eval_freq)
    batch_size = 256 
    train_bonus = int(wandb_config.bonus_train_steps)
    agent = TD3Agent(env_name, env_id, batch_size, seed)
    re  = []
    rloss = []
    ev = 0
    for i in tqdm(range(0, train_bonus)):
        l1, l2 = agent.train_bonus_learner(i)
    
    for i in tqdm(range(0, train_epochs)):
        agent.train(i)
        if i%eval_freq == 0:
            result = agent.eval(10, seed)
            run.log({"Average Reward" : result}, step = ev)
            ev+=1 

    agent = None 
    gc.collect()

parser = argparse.ArgumentParser()
parser.add_argument("--config")
args = parser.parse_args()
conf = OmegaConf.load(args.config)
conf = OmegaConf.to_container(conf)
sweep_id = wandb.sweep(sweep=conf, project="OfflinePseudo")
wandb.agent(sweep_id, function=run_exp)

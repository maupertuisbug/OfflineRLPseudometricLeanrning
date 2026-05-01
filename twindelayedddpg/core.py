import gymnasium as gym 
import mujoco
from twindelayedddpg.network import Actor, QFunction 
import torch 
from torchrl.data import LazyTensorStorage, TensorDictReplayBuffer 
from torchrl.data.datasets.minari_data import MinariExperienceReplay
from torchrl.data.replay_buffers import SamplerWithoutReplacement
from tensordict import TensorDict 
import copy 
import numpy as np 
import matplotlib.pyplot as plt 
from gymnasium.wrappers import RecordVideo 
import minari
import os 
os.environ["MUJOCO_GL"] = "glx"
from bonus_learning.core import BonusLearner



def softupdate(target, source, tau):

    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            tau * source_param.data + (1.0 - tau)*target_param.data
        )

class TD3Agent:

    def __init__(self, env_name, env_id, batch_size, seed, scale_reward=True):

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        self.env = gym.make(env_name)
        self.state_n = self.env.observation_space.shape[0]
        self.action_n = self.env.action_space.shape[0]
        self.device   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.alpha_c = 1 
        self.alpha_a = 5
        self.batch_size = batch_size

        self.replay_buffer = MinariExperienceReplay(
            env_id, 
            split_trajs = False, 
            batch_size = batch_size, 
            sampler = SamplerWithoutReplacement()
        )
        dataset = minari.load_dataset(env_id, download=True)
        episodes = dataset.iterate_episodes()
        self.reward_min = int(100000)
        for ep in episodes:
            min_ep_reward = ep.rewards.min()
            self.reward_min = np.minimum(self.reward_min, min_ep_reward)

        episodes = dataset.iterate_episodes()
        self.reward_max = float(-0.00001)
        for ep in episodes:
            max_ep_reward = ep.rewards.max()
            self.reward_max = np.maximum(self.reward_max, max_ep_reward)

        self.scale_reward = scale_reward
        

        self.input_dim = self.state_n

        self.actor = Actor(self.input_dim, self.action_n, self.env.action_space).to(self.device)
        self.actor_target = copy.deepcopy(self.actor).to(self.device)

        self.qvalue_a = QFunction(self.input_dim+self.action_n).to(self.device)
        self.qvalue_a_target = copy.deepcopy(self.qvalue_a).to(self.device)

        self.qvalue_b = QFunction(self.input_dim+self.action_n).to(self.device)
        self.qvalue_b_target = copy.deepcopy(self.qvalue_b).to(self.device)

        self.training = False 

        self.bonus_learner = BonusLearner(self.input_dim, self.action_n, self.env.action_space, batch_size)

        self.mean_episode = [] 
        self.mean_loss = [] 
        self.mean_loss_vs = [] 

    def get_data(self):

        while True:
            data = self.replay_buffer.sample()
            if data.shape[0] == self.batch_size:
                break
        
        return data

    def train_bonus_learner(self, i):

        data_one = self.get_data()
        data_two = self.get_data()

        states_batch_one = data_one['observation'].to(self.device)
        states_batch_two = data_two['observation'].to(self.device)

        loss_psi = self.bonus_learner.train_psi(states_batch_one, states_batch_two)
        loss_phi = self.bonus_learner.train_phi(data_one, data_two)

        return loss_psi.detach().cpu().numpy(), loss_phi.detach().cpu().numpy()

    def compute_bonus(self, states, actions):

        data = self.get_data()

        states_other = data['observation'].to(self.device)
        actions_other = data['action'].to(self.device)

        bonus = self.bonus_learner.compute_bonus(states, actions, states_other, actions_other)
        return bonus


    def train(self, i):

        loss_qnet = [] 
        loss_anet = []
        update_freq = 1 
        total_steps_per_epoch = 0 


        data = self.get_data()

        obs_ = data['observation'].to(self.device)
        action_ = data['action'].to(self.device)
        reward_  = data['next']['reward'].squeeze(1).to(self.device)
        
        if self.scale_reward == True:
            reward_ = (reward_ - self.reward_min)/(self.reward_max - self.reward_min)
        
        next_obs_ = data['next']['observation'].to(self.device)
        done_    = data['next']['done'].int().squeeze(1).to(self.device)
        obs_repr_ = obs_
        next_obs_repr_ = next_obs_


        predicted_actions = self.actor_target.forward_pred(next_obs_repr_)

        target_values_a  = self.qvalue_a_target(torch.cat([next_obs_repr_, predicted_actions], dim=1)).squeeze(1)
        target_values_b  = self.qvalue_b_target(torch.cat([next_obs_repr_, predicted_actions], dim=1)).squeeze(1)

        target_values = torch.min(target_values_a, target_values_b)
        target_q      = reward_ + 0.99*(1-done_)*(target_values)

        bonus_one = self.compute_bonus(next_obs_repr_, predicted_actions)
        bonus_one = torch.exp(-0.5*bonus_one).detach()
        bonus_one = self.alpha_c * bonus_one
        target_q     = target_q + bonus_one

        current_q_a = self.qvalue_a(torch.cat([obs_repr_, action_], dim=1)).squeeze(1)
        current_q_b = self.qvalue_b(torch.cat([obs_repr_, action_], dim=1)).squeeze(1)

        loss_q_a = torch.mean((current_q_a - target_q)**2, dim=0)
        loss_q_b = torch.mean((current_q_b - target_q)**2, dim=0)

        loss = loss_q_a + loss_q_b

        self.qvalue_a.optimizer.zero_grad()
        self.qvalue_b.optimizer.zero_grad()

        loss.backward()

        self.qvalue_a.optimizer.step()
        self.qvalue_b.optimizer.step()

    
        if i%2==0:

            action_pred =  self.actor(obs_repr_)
            actor_pred = self.qvalue_a(torch.cat([obs_repr_,action_pred], dim=1)).squeeze(1)
            bonus_two = self.compute_bonus(obs_repr_, action_pred)
            bonus_two = torch.exp(-0.5 * bonus_two).detach()
            bonus_two = self.alpha_a * bonus_two
            
            loss_p = -torch.mean(actor_pred+bonus_two, dim=0)

            self.actor.optimizer.zero_grad()
            loss_p.backward()
            self.actor.optimizer.step()

            softupdate(self.actor_target, self.actor, 0.005)
            softupdate(self.qvalue_a_target, self.qvalue_a, 0.005)
            softupdate(self.qvalue_b_target, self.qvalue_b, 0.005)

            loss_qnet.append(loss.detach().cpu().numpy())
            loss_anet.append(loss_p.detach().cpu().numpy())



    def eval(self, episodes, seed):

        epr_reward = [] 
        for ep in range(0, episodes):
            done = False 
            obs, _ = self.env.reset(seed=seed)
            ep_reward = 0 
            steps = 0 

            while not done and steps < 1000:
                with torch.no_grad():
                    obs_repr = torch.tensor(obs, dtype=torch.float64, device=self.device).unsqueeze(0)
                    action = self.actor.net(obs_repr)
                    action = torch.tanh(action).squeeze(0)

                next_obs, reward, done, _, _ = self.env.step(action.detach().cpu().numpy())
                ep_reward = ep_reward + reward
                obs   = next_obs 
                steps+=1 

            epr_reward.append(ep_reward)

        return np.mean(epr_reward)





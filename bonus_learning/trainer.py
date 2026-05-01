import torch 
from bonus_learning.core import BonusLearner
from torchrl.data.datasets.minari_data import MinariExperienceReplay
from torchrl.data.replay_buffers import SamplerWithoutReplacement
import gymnasium as gym
from plot import plot



dataset_id = 'mujoco/walker2d/medium-v0'
batch_size = 256

replay_buffer = MinariExperienceReplay(
    dataset_id,
    split_trajs=False,
    batch_size=batch_size,
    sampler=SamplerWithoutReplacement(),
)

env = gym.make('Walker2d-v5')
state_n = env.observation_space.shape[0]
action_n = env.action_space.shape[0]
action_spec = env.action_space

bl = BonusLearner(state_n, action_n, action_spec, batch_size)
loss_psis = []
loss_phis = []

for i in range(0, 10000):

    while True:
        data_one = replay_buffer.sample()
        if data_one.shape[0] == batch_size:
                break
    
    while True:
        data_two = replay_buffer.sample()
        if data_two.shape[0] == batch_size:
                break

    states_batch_one = data_one['observation'].to('cuda')
    states_batch_two = data_two['observation'].to('cuda')
    
    loss_psi = bl.train_psi(states_batch_one, states_batch_two)
    loss_phi = bl.train_phi(data_one, data_two)
    if i%500==0:
        loss_psis.append(loss_psi.detach().cpu().numpy())
        loss_phis.append(loss_phi.detach().cpu().numpy())

    if i%1000==0:
        plot(loss_psis, loss_phis, "Loss Phi", "Loss Psi", "Step", "Loss", "BonusLearning", True)

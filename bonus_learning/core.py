import torch 
from bonus_learning.network import Phi, Psi
import copy


def softupdate(target, source, tau):

    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            tau * source_param.data + (1.0 - tau)*target_param.data
        )

class BonusLearner:

    def __init__(self, state_n, action_n, env_action_space, batch_size, repr_dim=None):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.phi = Phi(state_n, action_n).to(self.device)
        self.psi = Psi(state_n).to(self.device)
        self.phi_target = copy.deepcopy(self.phi)
        self.action_space = env_action_space
        self.action_n = action_n
        self.batch_size = batch_size

    def compute_loss_over_uniform_actions(self, state_one, state_two):
        
        low = torch.tensor(self.action_space.low)
        high = torch.tensor(self.action_space.high)

        actions_batch_one = low + (high-low) * torch.randn(self.batch_size, self.action_n)
        actions_batch_two = low + (high-low) * torch.randn(self.batch_size, self.action_n)
        actions_batch_one = actions_batch_one.to(self.device)
        actions_batch_two = actions_batch_two.to(self.device)

        states_batch_one = state_one.unsqueeze(0).repeat(self.batch_size, 1)
        states_batch_two = state_two.unsqueeze(0).repeat(self.batch_size, 1)
        phi_output_one = self.phi_target(torch.cat([states_batch_one, actions_batch_one], dim=1))
        phi_output_two = self.phi_target(torch.cat([states_batch_two, actions_batch_two], dim=1))

        l1_over_actions = torch.sum(torch.abs(torch.subtract(phi_output_one, phi_output_two)), dim=1) #batch_size, 32
        l1_over_actions = torch.mean(l1_over_actions)

        return l1_over_actions



    def train_psi(self, states_batch_one, states_batch_two):

        l1_loss = torch.sum(torch.abs(torch.subtract(self.psi(states_batch_one), self.psi(states_batch_two))), dim=1)

        l2_loss = []
        for state_one, state_two in zip(states_batch_one, states_batch_two):
            loss = self.compute_loss_over_uniform_actions(state_one, state_two)
            l2_loss.append(loss)

        l2_loss = torch.stack(l2_loss).detach()

        total_loss = torch.mean(torch.square(l1_loss-l2_loss), dim=0)

        self.psi.optimizer.zero_grad()
        total_loss.backward()
        self.psi.optimizer.step()

        return total_loss

    def train_phi(self, batch_one, batch_two):

        states_one = batch_one['observation'].to(self.device)
        action_one = batch_one['action'].to(self.device)
        next_states_one = batch_one['next']['observation'].to(self.device)
        rewards_one = batch_one['next']['reward'].squeeze(1).to(self.device)
        phi_input_one = torch.cat([states_one, action_one], dim=1)

        states_two = batch_two['observation'].to(self.device)
        action_two = batch_two['action'].to(self.device)
        next_states_two = batch_two['next']['observation'].to(self.device)
        rewards_two = batch_two['next']['reward'].squeeze(1).to(self.device)
        phi_input_two  = torch.cat([states_two, action_two], dim=1)

        phi_output = torch.sum(torch.abs(self.phi(phi_input_one)-self.phi(phi_input_two)), dim=1)

        rewards_diff = torch.abs(rewards_one-rewards_two)

        psi_output = torch.sum(torch.abs(self.psi(next_states_one) - self.psi(next_states_two)), dim=1).detach()

        loss = phi_output - (rewards_diff) - 0.99*(psi_output)
        total_loss = torch.mean(loss, dim=0)

        self.phi.optimizer.zero_grad()
        total_loss.backward()
        self.phi.optimizer.step()

        softupdate(self.phi_target, self.phi, 0.005)
        return total_loss


    def compute_bonus(self, states_one, actions_one, states_two, actions_two):

        phi_input_one = torch.cat([states_one, actions_one], dim=1)

        phi_output_one = self.phi(phi_input_one) # batchsize, 32

        phi_input_two = torch.cat([states_two, actions_two], dim=1)

        phi_output_two = self.phi(phi_input_two)

        similarity = phi_output_one @ phi_output_two.T 
        similarity = torch.sum(similarity, dim=1)

        return similarity

        






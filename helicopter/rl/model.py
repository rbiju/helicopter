import torch
from torch.distributions import Normal
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl


class ActionCorrector(nn.Module):
    def __init__(self):
        super().__init__()
        self.multiplicative_factor = torch.Tensor([0.5, 1.0, 1.0])
        self.additive_factor = torch.Tensor([0.5, 0.0, 0.0])

    def forward(self, actions):
        return actions * self.multiplicative_factor + self.additive_factor


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, corrector: ActionCorrector = ActionCorrector()):
        super().__init__()
        self.corrector = corrector

        self.hidden_layers = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )

        self.actor_mean = nn.Linear(64, action_dim)
        self.actor_log_std = nn.Parameter(torch.zeros(1, action_dim))

        self.critic_value = nn.Linear(64, 1)

    def forward(self, state):
        features = self.hidden_layers(state)

        raw_mean = self.actor_mean(features)
        action_mean = self.corrector(torch.tanh(raw_mean))

        action_std = self.actor_log_std.exp().expand_as(action_mean)
        state_value = self.critic_value(features)

        return action_mean, action_std, state_value


class PPOLoss(nn.Module):
    def __init__(self, clip_ratio=0.2, value_coef=0.5, entropy_coef=0.01):
        super().__init__()
        self.clip_ratio = clip_ratio
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef

    def forward(self, old_log_probs, new_log_probs, advantages, returns, values, entropy):
        ratio = torch.exp(new_log_probs - old_log_probs)

        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * advantages

        policy_loss = -torch.min(surr1, surr2).mean()

        value_loss = F.mse_loss(values, returns)

        entropy_loss = -entropy.mean()

        total_loss = policy_loss + (self.value_coef * value_loss) + (self.entropy_coef * entropy_loss)

        return {
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "entropy_loss": entropy_loss,
            "loss": total_loss
        }


class FlightEnvironmentDataset(torch.utils.data.Dataset):
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim


class FlightAgentPPO(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.agent = ActorCritic(18, 3)

    @staticmethod
    @torch.compile
    def calculate_returns(rewards: torch.Tensor, gamma: float) -> torch.Tensor:
        returns = torch.empty_like(rewards)
        R = torch.tensor(0.0, device=rewards.device)

        for i in range(len(rewards) - 1, -1, -1):
            R = rewards[i] + gamma * R
            returns[i] = R

        return returns

    def forward(self, state):
        action_mean, action_std, state_value = self.agent(state)
        dist = Normal(action_mean, action_std)
        action = dist.sample()

        return action, state_value

    def episode(self):
        output = {'saved_log_probs': [], 'saved_values': [], 'rewards': [], 'saved_entropies': []}
        state, info = env.reset()
        done = False

        while not done:
            action, state_value = self.forward(state)
            next_state, reward, terminated, truncated, info = env.step(action.detach().cpu().numpy()[0])

            done = terminated or truncated

            output['saved_log_probs'].append(dist.log_prob(action))
            output['saved_values'].append(state_value)
            output['rewards'].append(reward)
            output['saved_entropies'].append(dist.entropy())
            state = next_state

        saved_log_probs = torch.stack(output['saved_log_probs']).to(self.device)
        saved_values = torch.stack(output['saved_values']).to(self.device)
        saved_entropies = torch.stack(output['saved_entropies']).to(self.device)
        rewards = torch.tensor(output['rewards'], device=self.device, dtype=torch.float32)

        return saved_log_probs, saved_values, rewards, saved_entropies

    def training_step(self, batch, batch_idx):
        episode_output = self.episode()

        returns = self.calculate_returns(episode_output['rewards'], gamma=0.99)
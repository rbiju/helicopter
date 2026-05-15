import jax
import jax.numpy as jnp

import torch
from torch.distributions import Normal
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.dlpack import from_dlpack

import pytorch_lightning as pl
from pytorch_lightning.utilities.types import OptimizerLRScheduler

from .environment import FlightEnvironment


class ActionCorrector(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('multiplicative_factor', torch.tensor([0.5, 1.0, 1.0]))
        self.register_buffer('additive_factor', torch.tensor([0.5, 0.0, 0.0]))

    def forward(self, actions):
        return ((torch.clip(actions, -1.0, 1.0) * self.multiplicative_factor)
                + self.additive_factor)


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, corrector: ActionCorrector = ActionCorrector()):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.actor = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim * 2)
        )

        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        self.corrector = corrector

    def evaluate(self, state):
        return self.critic(state)

    def forward(self, state):
        actor_out = self.actor(state).reshape(-1, self.action_dim)

        action_mean = self.corrector(torch.tanh(actor_out[0::2, :]))
        action_std = actor_out[1::2, :].exp()

        state_value = self.evaluate(state)

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

        actor_loss = self.value_coef * value_loss + self.entropy_coef * entropy_loss

        return {
            "policy_loss": policy_loss,
            "actor_loss": actor_loss,
            "entropy_loss": entropy_loss,
            "critic_loss": value_loss,
        }


class StepCounterDataset(torch.utils.data.Dataset):
    def __init__(self, num_steps: int):
        super().__init__()
        self.num_steps = num_steps

    def __len__(self):
        return self.num_steps

    def __getitem__(self, idx):
        return torch.tensor(0.0)


class FlightAgentPPO(pl.LightningModule):
    def __init__(self,
                 agent: ActorCritic = ActorCritic(state_dim=21, action_dim=3),
                 loss: PPOLoss = PPOLoss(),
                 env: FlightEnvironment = FlightEnvironment(),
                 dataset: StepCounterDataset = StepCounterDataset(num_steps=1_000),
                 num_envs: int = 10_000,
                 rollout_steps: int = 250,
                 iters_per_rollout: int = 5,):
        super().__init__()
        self.save_hyperparameters(ignore=['agent', 'loss', 'env'])
        self.agent = agent
        self.loss = loss
        self.env = env

        self.dataset = dataset
        self.num_envs = num_envs
        self.rollout_steps = rollout_steps
        self.iters_per_rollout = iters_per_rollout

        self.vmap_reset = jax.vmap(env.reset_env, in_axes=(0, None))
        self.vmap_step = jax.vmap(env.step_env, in_axes=(0, 0, 0, None))

        self.jax_rng = jax.random.key(0)
        self.jax_rng, reset_rng = jax.random.split(self.jax_rng)
        reset_rngs = jax.random.split(reset_rng, self.num_envs)

        self.current_obs_jax, self.current_state_jax = self.vmap_reset(
            reset_rngs, self.env.default_params
        )

        self.automatic_optimization = False

    @staticmethod
    @torch.compile
    def calculate_gae(rewards: torch.Tensor, masks: torch.Tensor, values: torch.Tensor,
                      gamma: float = 0.99, lam: float = 0.95) -> tuple[torch.Tensor, torch.Tensor]:
        advantages = torch.empty_like(rewards)

        last_gae_lam = torch.zeros_like(rewards[0])
        next_values = torch.cat([values[1:], torch.zeros_like(values[0:1])])

        for t in range(rewards.size(0) - 1, -1, -1):
            delta = rewards[t] + gamma * next_values[t] * masks[t] - values[t]
            last_gae_lam = delta + gamma * lam * masks[t] * last_gae_lam
            advantages[t] = last_gae_lam

        returns = advantages + values

        return returns, advantages

    def episode(self):
        rollout = {
            'obs': [], 'actions': [], 'log_probs': [],
            'values': [], 'entropy': [], 'rewards': [], 'masks': []
        }

        obs_jax = self.current_obs_jax
        state_jax = self.current_state_jax

        for _ in range(self.rollout_steps):
            obs_pt = from_dlpack(obs_jax)

            action_mean, action_std, state_value = self.agent(obs_pt)
            dist = Normal(action_mean, action_std)
            action_raw_pt = dist.sample()
            action_pt = self.agent.corrector(action_raw_pt)

            log_prob = dist.log_prob(action_raw_pt).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1)

            action_jax = jax.dlpack.from_dlpack(action_pt)

            self.jax_rng, step_rng = jax.random.split(self.jax_rng)
            step_rngs = jax.random.split(step_rng, self.num_envs)

            next_obs_jax, next_state_jax, reward_jax, done_jax, _ = self.vmap_step(
                step_rngs, state_jax, action_jax, self.env.default_params
            )

            rollout['obs'].append(obs_pt)
            rollout['actions'].append(action_raw_pt)
            rollout['log_probs'].append(log_prob)
            rollout['values'].append(state_value.squeeze())
            rollout['entropy'].append(entropy)
            rollout['rewards'].append(from_dlpack(reward_jax))
            rollout['masks'].append(1.0 - from_dlpack(done_jax.astype(jnp.float32)))

            obs_jax = next_obs_jax
            state_jax = next_state_jax

        self.current_obs_jax = obs_jax
        self.current_state_jax = state_jax

        return {k: torch.stack(v) for k, v in rollout.items()}

    def train_dataloader(self):
        return torch.utils.data.DataLoader(dataset=self.dataset, batch_size=1, num_workers=0)

    def training_step(self, batch, batch_idx):
        actor_opt, critic_opt = self.optimizers()
        data = self.episode()

        b_obs = data['obs'].view(-1, 21)
        b_actions = data['actions'].view(-1, 3)
        b_old_log_probs = data['log_probs'].view(-1)

        b_returns, b_advantages = self.calculate_gae(data['rewards'],
                                                     data['masks'],
                                                     data['values'],
                                                     gamma=0.99)
        b_returns = b_returns.view(-1)
        b_advantages = b_advantages.view(-1)
        b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)

        for _ in range(self.iters_per_rollout):
            action_mean, action_std, new_values = self.agent(b_obs)
            dist = Normal(action_mean, action_std)

            new_log_probs = dist.log_prob(b_actions).sum(dim=-1)
            new_entropy = dist.entropy().sum(dim=-1)

            loss_dict = self.loss(
                old_log_probs=b_old_log_probs,
                new_log_probs=new_log_probs,
                advantages=b_advantages,
                returns=b_returns,
                values=new_values.squeeze(),
                entropy=new_entropy
            )

            actor_opt.zero_grad()
            self.manual_backward(loss_dict['actor_loss'])
            actor_opt.step()

            critic_opt.zero_grad()
            self.manual_backward(loss_dict['critic_loss'])
            critic_opt.step()

    def configure_optimizers(self) -> OptimizerLRScheduler:
        actor_optimizer = torch.optim.Adam(self.agent.actor.parameters(), lr=self.lr)
        critic_optimizer = torch.optim.Adam(self.agent.critic.parameters(), lr=self.lr)

        return [actor_optimizer, critic_optimizer]

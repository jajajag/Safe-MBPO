"""Small model-free RL building blocks used by RPT and SSkP.

The original repository's :mod:`src.ssac` implements the safety critic used by
SMBPO.  RPT and SSkP need ordinary SAC targets instead, so this module keeps a
separate implementation rather than changing the SMBPO baseline.
"""

import copy
import math

import torch
from torch import nn

from .config import BaseConfig, Configurable, Optional
from .policy import BasePolicy, SquashedGaussianPolicy, TorchPolicy
from .torch_util import Module, device, freeze_module, mlp, update_ema


class TwinQ(Module):
    """Two independent Q functions."""

    def __init__(self, state_dim, action_dim, hidden_dim=256, hidden_layers=2):
        super().__init__()
        dims = [state_dim + action_dim] + [hidden_dim] * hidden_layers + [1]
        self.q1 = mlp(dims, squeeze_output=True)
        self.q2 = mlp(dims, squeeze_output=True)

    def forward(self, states, actions):
        inputs = torch.cat((states, actions), dim=-1)
        return self.q1(inputs), self.q2(inputs)

    def minimum(self, states, actions):
        q1, q2 = self(states, actions)
        return torch.min(q1, q2)


class GaussianPolicy(TorchPolicy):
    """Unsquashed diagonal Gaussian, used for latent skills."""

    def __init__(self, net, log_std_bounds=(-10.0, 2.0)):
        super().__init__(net)
        self.log_std_bounds = log_std_bounds

    def _distr(self, net_out):
        mean, log_std = net_out.chunk(2, dim=-1)
        log_std = log_std.clamp(*self.log_std_bounds)
        return torch.distributions.Independent(
            torch.distributions.Normal(mean, log_std.exp()), 1
        )

    def _special_eval(self, distribution):
        return distribution.mean


class SAC(BasePolicy, Configurable, Module):
    """Soft Actor-Critic with a tanh Gaussian actor.

    ``prior`` is optional.  When supplied to :meth:`update`, the entropy term
    becomes ``log pi(z|s) - log q(z|s)``, i.e. the SPiRL KL regularizer.
    """

    class Config(BaseConfig):
        discount = 0.99
        tau = 0.005
        batch_size = 256
        hidden_dim = 256
        hidden_layers = 2
        actor_lr = 3e-4
        critic_lr = 3e-4
        alpha_lr = 3e-4
        init_alpha = 0.1
        autotune_alpha = True
        target_entropy = Optional(float)
        squash_actions = True

    def __init__(self, config, state_dim, action_dim):
        Configurable.__init__(self, config)
        Module.__init__(self)
        self.state_dim = state_dim
        self.action_dim = action_dim

        actor_net = mlp(
            [state_dim] + [self.hidden_dim] * self.hidden_layers + [2 * action_dim]
        )
        self.actor = (SquashedGaussianPolicy(actor_net) if self.squash_actions
                      else GaussianPolicy(actor_net))
        self.critic = TwinQ(state_dim, action_dim, self.hidden_dim, self.hidden_layers)
        self.critic_target = copy.deepcopy(self.critic)
        freeze_module(self.critic_target)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.critic_lr)
        self.log_alpha = nn.Parameter(
            torch.tensor(math.log(self.init_alpha), dtype=torch.float, device=device)
        )
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.alpha_lr)
        if self.target_entropy is None or isinstance(self.target_entropy, Optional):
            self.target_entropy = -float(action_dim)
        self.register_buffer('total_updates', torch.zeros([], dtype=torch.long))

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def distribution(self, states):
        return self.actor.distr(states)

    def act(self, states, eval=False):
        with torch.no_grad():
            distribution = self.distribution(states)
            return distribution.mean if eval else distribution.sample()

    @staticmethod
    def _regularizer(distribution, actions, prior=None):
        log_pi = distribution.log_prob(actions)
        if prior is None:
            return log_pi
        # Detach prior parameters while preserving d log q(a|s) / d a.  A
        # blanket no_grad block would incorrectly remove the prior's force on
        # the policy sample and would not implement KL(pi || q).
        if (isinstance(prior, torch.distributions.Independent)
                and isinstance(prior.base_dist, torch.distributions.Normal)):
            base = prior.base_dist
            detached_prior = torch.distributions.Independent(
                torch.distributions.Normal(
                    base.loc.detach(), base.scale.detach()
                ),
                prior.reinterpreted_batch_ndims
            )
            log_prior = detached_prior.log_prob(actions)
        else:
            log_prior = prior.log_prob(actions)
        return log_pi - log_prior

    def update(self, samples, prior=None):
        states, actions, next_states, rewards, dones = samples[:5]

        with torch.no_grad():
            next_distribution = self.distribution(next_states)
            next_actions = next_distribution.rsample()
            next_prior = prior(next_states) if callable(prior) else None
            next_regularizer = self._regularizer(
                next_distribution, next_actions, next_prior
            )
            next_q = self.critic_target.minimum(next_states, next_actions)
            target = rewards + self.discount * (1.0 - dones.float()) * (
                next_q - self.alpha.detach() * next_regularizer
            )

        q1, q2 = self.critic(states, actions)
        critic_loss = ((q1 - target).pow(2).mean() + (q2 - target).pow(2).mean())
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        distribution = self.distribution(states)
        sampled_actions = distribution.rsample()
        action_prior = prior(states) if callable(prior) else None
        regularizer = self._regularizer(distribution, sampled_actions, action_prior)
        actor_q = self.critic.minimum(states, sampled_actions)
        actor_loss = (self.alpha.detach() * regularizer - actor_q).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        alpha_loss = torch.zeros([], device=device)
        if self.autotune_alpha:
            # For ordinary SAC regularizer is log pi and target_entropy is
            # negative.  For prior SAC it is a sampled KL and the target is a
            # positive divergence budget.
            if prior is None:
                alpha_loss = -(
                    self.log_alpha * (regularizer.detach() + self.target_entropy)
                ).mean()
            else:
                alpha_loss = (
                    self.alpha * (self.target_entropy - regularizer.detach())
                ).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

        update_ema(self.critic_target, self.critic, self.tau)
        self.total_updates += 1
        return {
            'critic loss': critic_loss.detach().item(),
            'actor loss': actor_loss.detach().item(),
            'alpha loss': alpha_loss.detach().item(),
            'alpha': self.alpha.detach().item(),
        }

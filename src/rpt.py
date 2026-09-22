from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from tqdm import trange

from .config import BaseConfig, Configurable
from .env.util import env_dims, get_max_episode_steps
from .log import TabularLog, default_log as log
from .model_free import SAC
from .policy import UniformPolicy
from .shared import SafetySampleBuffer
from .torch_util import DummyModuleWrapper, Module, device, mlp


N_EVAL_EPISODES = 5


class ContrastiveRiskPredictor(Configurable, Module):

    class Config(BaseConfig):
        hidden_dim = 256
        hidden_layers = 2
        learning_rate = 3e-4
        batch_size = 1024
        positive_prior = 0.0
        augmentation_copies = 0
        augmentation_std = 1.0
        max_grad_norm = 10.0

    def __init__(self, config, state_dim, action_dim):
        Configurable.__init__(self, config)
        Module.__init__(self)
        dims = [state_dim + action_dim] + [self.hidden_dim] * self.hidden_layers + [1]
        self.network = mlp(dims, squeeze_output=True)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.learning_rate)

    def logits(self, states, actions):
        return self.network(torch.cat((states, actions), dim=-1))

    def classifier_probability(self, states, actions):
        return torch.sigmoid(self.logits(states, actions))

    def risk(self, states, actions):
        classifier_probability = 0.5 * self.classifier_probability(states, actions)
        return classifier_probability / (1.0 - classifier_probability + 1e-8)

    def update(self, positive_states, positive_actions, general_states, general_actions):
        if self.augmentation_copies > 0:
            expanded_states = [positive_states]
            expanded_actions = [positive_actions]
            for _ in range(self.augmentation_copies):
                noise = torch.randn_like(positive_states) * self.augmentation_std
                expanded_states.append(positive_states + noise)
                expanded_actions.append(positive_actions)
            positive_states = torch.cat(expanded_states, dim=0)
            positive_actions = torch.cat(expanded_actions, dim=0)

        positive_logits = self.logits(positive_states, positive_actions)
        general_logits = self.logits(general_states, general_actions)
        loss = (
            self.positive_prior * F.softplus(-positive_logits).mean()
            + F.softplus(general_logits).mean()
        )
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
        self.optimizer.step()
        return loss.detach().item()


class RPT(Configurable, Module):

    class Config(BaseConfig):
        sac_cfg = SAC.Config()
        risk_cfg = ContrastiveRiskPredictor.Config()
        buffer_capacity = 10 ** 6
        initial_random_steps = 5000
        steps_per_epoch = 1000
        policy_updates_per_epoch = 200
        risk_updates_per_epoch = 1
        reward_penalty = 0.05
        prevention_threshold = 0.9
        prevention_enabled = True
        reward_shaping_enabled = True
        save_trajectories = True

    def __init__(self, config, env_factory, data):
        Configurable.__init__(self, config)
        Module.__init__(self)
        self.data = data
        self.real_env = env_factory()
        self.eval_env_factory = env_factory
        self.state_dim, self.action_dim = env_dims(self.real_env)
        self.max_episode_steps = get_max_episode_steps(self.real_env)

        self.sac = SAC(self.sac_cfg, self.state_dim, self.action_dim)
        self.risk_predictor = ContrastiveRiskPredictor(
            self.risk_cfg, self.state_dim, self.action_dim
        )
        self.uniform_policy = UniformPolicy(self.real_env)
        replay = SafetySampleBuffer(
            self.state_dim, self.action_dim, self.buffer_capacity, device=device
        )
        self.replay_buffer = DummyModuleWrapper(replay)

        self.register_buffer('epochs_completed', torch.zeros([], dtype=torch.long))
        self.register_buffer('steps_sampled', torch.zeros([], dtype=torch.long))
        self.register_buffer('episodes_sampled', torch.zeros([], dtype=torch.long))
        self.register_buffer('n_violations', torch.zeros([], dtype=torch.long))
        self.register_buffer('raw_violations', torch.zeros([], dtype=torch.long))
        self.episode_log = TabularLog(log.dir, 'episodes.csv')
        self._is_setup = False

    @property
    def actor(self):
        return self.sac

    def setup(self):
        if self._is_setup:
            return
        self.episodes_dir = Path(log.dir) / 'episodes'
        if self.save_trajectories:
            self.episodes_dir.mkdir(exist_ok=True)
            for path in sorted(self.episodes_dir.glob('episode-*.h5py')):
                episode = SafetySampleBuffer.from_h5py(path, device=device)
                self.replay_buffer.extend(**episode.get(as_dict=True))
        self._reset_episode()
        self._is_setup = True

    def _reset_episode(self):
        self._state = self.real_env.reset()
        self._episode = SafetySampleBuffer(
            self.state_dim, self.action_dim, self.max_episode_steps, device=device
        )
        self._episode_return = 0.0
        self._episode_raw_return = 0.0
        self._episode_prevented = False

    def _choose_action(self, state):
        if self.steps_sampled.item() < self.initial_random_steps:
            return self.uniform_policy.act1(state)
        return self.sac.act1(state)

    def _risk(self, state, action):
        with torch.no_grad():
            return self.risk_predictor.risk(state[None], action[None])[0].item()

    def _finish_episode(self, actual_violation):
        self.episodes_sampled += 1
        reported_violation = bool(actual_violation and not self._episode_prevented)
        if actual_violation:
            self.raw_violations += 1
        if reported_violation:
            self.n_violations += 1
        row = {
            'episodes sampled': self.episodes_sampled.item(),
            'steps sampled': self.steps_sampled.item(),
            'total violations': self.n_violations.item(),
            'collect return': self._episode_raw_return,
            'collect length': len(self._episode),
        }
        for key, value in row.items():
            self.data.append(key, value, verbose=reported_violation)
        row.update({
            'collect shaped return': self._episode_return,
            'collect violation': reported_violation,
            'collect prevented': bool(self._episode_prevented),
        })
        for key in ('collect shaped return', 'collect violation', 'collect prevented'):
            self.data.append(key, row[key])
        self.episode_log.row(row)
        if self.save_trajectories:
            path = self.episodes_dir / 'episode-{}.h5py'.format(
                self.episodes_sampled.item()
            )
            self._episode.save_h5py(path)
        self._reset_episode()

    def _collect_step(self):
        state = self._state
        action = self._choose_action(state)
        risk = self._risk(state, action)
        next_state, raw_reward, env_done, info = self.real_env.step(action)
        violation = bool(info['violation'])

        shaped_reward = raw_reward
        if self.reward_shaping_enabled:
            shaped_reward -= self.reward_penalty * risk

        timeout = len(self._episode) + 1 >= self.max_episode_steps
        if not (env_done or violation or timeout) and self.prevention_enabled:
            next_action = self._choose_action(next_state)
            next_risk = self._risk(next_state, next_action)
            # Latch for the whole trajectory, without terminating the rollout.
            self._episode_prevented |= next_risk >= self.prevention_threshold

        replay_done = bool(env_done or violation or timeout)
        transition = dict(
            states=state,
            actions=action,
            next_states=next_state,
            rewards=shaped_reward,
            dones=replay_done,
            violations=violation,
        )
        self.replay_buffer.append(**transition)
        self._episode.append(**transition)
        self.steps_sampled += 1
        self._episode_return += shaped_reward
        self._episode_raw_return += raw_reward

        if replay_done:
            self._finish_episode(violation)
        else:
            self._state = next_state

    def _update_risk_predictor(self):
        states, actions, violations = self.replay_buffer.get(
            'states', 'actions', 'violations'
        )
        positive_indices = torch.nonzero(violations).flatten()
        if len(positive_indices) == 0:
            return None
        if self.risk_predictor.positive_prior <= 0:
            episode_count = max(1, self.episodes_sampled.item())
            positive_prior = max(1e-4, self.raw_violations.item() / episode_count)
        else:
            positive_prior = self.risk_predictor.positive_prior
        batch_size = self.risk_cfg.batch_size
        positive_choice = positive_indices[
            torch.randint(len(positive_indices), (batch_size,), device=device)
        ]
        general_choice = torch.randint(len(states), (batch_size,), device=device)
        configured_prior = self.risk_predictor.positive_prior
        self.risk_predictor.positive_prior = positive_prior
        loss = self.risk_predictor.update(
            states[positive_choice], actions[positive_choice],
            states[general_choice], actions[general_choice]
        )
        self.risk_predictor.positive_prior = configured_prior
        return loss

    def epoch(self):
        self.setup()
        for _ in trange(self.steps_per_epoch, desc='collect RPT'):
            self._collect_step()

        update_stats = []
        if len(self.replay_buffer) >= self.sac_cfg.batch_size:
            for _ in trange(self.policy_updates_per_epoch, desc='update SAC'):
                samples = self.replay_buffer.sample(self.sac_cfg.batch_size)
                update_stats.append(self.sac.update(samples))

        risk_losses = []
        for _ in range(self.risk_updates_per_epoch):
            loss = self._update_risk_predictor()
            if loss is not None:
                risk_losses.append(loss)
        if update_stats:
            for key in update_stats[0]:
                self.data.append(key, np.mean([row[key] for row in update_stats]))
        if risk_losses:
            self.data.append('risk predictor loss', np.mean(risk_losses))
        self.epochs_completed += 1

    def evaluate(self):
        returns, lengths, violations = [], [], []
        for _ in range(N_EVAL_EPISODES):
            env = self.eval_env_factory()
            state = env.reset()
            episode_return = 0.0
            episode_violation = False
            for step in range(get_max_episode_steps(env)):
                action = self.sac.act1(state, eval=True)
                state, reward, done, info = env.step(action)
                episode_return += reward
                episode_violation = episode_violation or bool(info['violation'])
                if done or info['violation']:
                    break
            returns.append(episode_return)
            lengths.append(step + 1)
            violations.append(episode_violation)
            env.close()
        result = {
            'eval return mean': float(np.mean(returns)),
            'eval return std': float(np.std(returns)),
            'eval length mean': float(np.mean(lengths)),
            'eval violations': int(np.sum(violations)),
        }
        for key, value in result.items():
            self.data.append(key, value, verbose=True)
        return result

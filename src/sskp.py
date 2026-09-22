from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from tqdm import trange

from .cem_risk import (
    CEMRiskTraceRecorder,
    TRACE_FILENAME,
    summarize_candidate_risks,
)
from .config import BaseConfig, Configurable, Require
from .env.util import env_dims, get_max_episode_steps
from .log import TabularLog, default_log as log
from .model_free import SAC
from .shared import SafetySampleBuffer
from .squashed_gaussian import SquashedGaussian
from .torch_util import DummyModuleWrapper, Module, device, mlp


N_EVAL_EPISODES = 5
SKILL_HORIZON_MODE = 'skill_horizon'
SKILL_HORIZON_CURRENT_PAIR_MODE = 'skill_horizon_current_pair'
PER_ACTION_MODE = 'per_action'
OPEN_LOOP_DECODER_MODE = 'open_loop'
CLOSED_LOOP_DECODER_MODE = 'closed_loop'
DECODER_MODES = (OPEN_LOOP_DECODER_MODE, CLOSED_LOOP_DECODER_MODE)
DECISION_MODES = (
    SKILL_HORIZON_MODE,
    SKILL_HORIZON_CURRENT_PAIR_MODE,
    PER_ACTION_MODE,
)


def _episode_file_number(path):
    try:
        return int(Path(path).stem.rsplit('-', 1)[1])
    except (IndexError, ValueError):
        raise ValueError('Invalid skill episode filename: {}'.format(path))


def _demonstration_sort_key(path):
    path = Path(path)
    parts = path.stem.rsplit('-', 1)
    if len(parts) == 2 and parts[0] == 'episode' and parts[1].isdigit():
        return 0, int(parts[1])
    return 1, path.name


def _read_episode(path):
    with h5py.File(str(path), 'r') as handle:
        arrays = {key: np.asarray(handle[key]) for key in handle.keys()}
    if 'actions' not in arrays or 'states' not in arrays:
        raise ValueError('{} must contain states and actions'.format(path))
    n_steps = len(arrays['actions'])
    states = arrays['states'][:n_steps]
    if len(states) != n_steps:
        raise ValueError('{} has incompatible states/actions lengths'.format(path))
    if 'violations' in arrays:
        violations = arrays['violations'].astype(np.bool_)
    elif 'dones' in arrays:
        violations = arrays['dones'].astype(np.bool_)
    else:
        violations = np.zeros(n_steps, dtype=np.bool_)
    if len(violations) != n_steps:
        raise ValueError(
            '{} has incompatible violations/actions lengths: {} and {}'
            .format(path, len(violations), n_steps)
        )
    return {
        'states': torch.tensor(states, dtype=torch.float, device=device),
        'actions': torch.tensor(arrays['actions'], dtype=torch.float, device=device),
        'violations': torch.tensor(violations, dtype=torch.bool, device=device),
    }


def load_demonstrations(path):
    path = Path(path)
    paths = (sorted(path.glob('*.h5py'), key=_demonstration_sort_key)
             if path.is_dir() else [path])
    if not paths or not all(item.is_file() for item in paths):
        raise FileNotFoundError('No demonstration HDF5 files found at {}'.format(path))
    return [_read_episode(item) for item in paths]


class SkillModel(Configurable, Module):

    class Config(BaseConfig):
        horizon = 5
        latent_dim = 16
        hidden_dim = 256
        encoder_hidden_dim = 256
        learning_rate = 3e-4
        batch_size = 256
        pretrain_steps = 50000
        kl_weight = 0.001
        prior_weight = 1.0
        min_std = 1e-4
        normalize_actions = True
        decoder_mode = OPEN_LOOP_DECODER_MODE
        closed_loop_action_noise_std = 0.0

    def __init__(self, config, state_dim, action_dim):
        Configurable.__init__(self, config)
        Module.__init__(self)
        self.state_dim = state_dim
        self.action_dim = action_dim
        if self.decoder_mode not in DECODER_MODES:
            raise ValueError(
                'decoder_mode must be one of {}, got {!r}'.format(
                    DECODER_MODES, self.decoder_mode
                )
            )
        encoder_input_dim = self.action_dim
        if self.decoder_mode == CLOSED_LOOP_DECODER_MODE:
            encoder_input_dim += self.state_dim
        self.encoder = nn.LSTM(
            input_size=encoder_input_dim, hidden_size=self.encoder_hidden_dim,
            num_layers=1, batch_first=True
        )
        self.encoder_head = nn.Linear(self.encoder_hidden_dim, 2 * self.latent_dim)
        self.prior_network = mlp(
            [state_dim, self.hidden_dim, self.hidden_dim, 2 * self.latent_dim]
        )
        if self.decoder_mode == CLOSED_LOOP_DECODER_MODE:
            decoder_input_dim = self.state_dim + self.latent_dim
            decoder_output_dim = self.action_dim
        else:
            decoder_input_dim = self.latent_dim
            decoder_output_dim = 2 * self.horizon * self.action_dim
        self.decoder_network = mlp([
            decoder_input_dim, self.hidden_dim, self.hidden_dim,
            decoder_output_dim
        ])
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        self.register_buffer('state_mean', torch.zeros(state_dim))
        self.register_buffer('state_std', torch.ones(state_dim))
        self.register_buffer('action_mean', torch.zeros(action_dim))
        self.register_buffer('action_std', torch.ones(action_dim))

    def set_normalization(self, episodes):
        states = torch.cat([episode['states'] for episode in episodes], dim=0)
        actions = torch.cat([episode['actions'] for episode in episodes], dim=0)
        self.state_mean.copy_(states.mean(dim=0))
        self.state_std.copy_(states.std(dim=0).clamp(min=1e-4))
        self.action_mean.copy_(actions.mean(dim=0))
        self.action_std.copy_(actions.std(dim=0).clamp(min=1e-4))

    def normalize_states(self, states):
        return (states - self.state_mean) / self.state_std

    def normalize_actions_tensor(self, actions):
        if not self.normalize_actions:
            return actions
        return (actions - self.action_mean) / self.action_std

    def denormalize_actions_tensor(self, actions):
        if not self.normalize_actions:
            return actions
        return actions * self.action_std + self.action_mean

    @staticmethod
    def _normal(parameters, min_std=1e-4):
        mean, log_std = parameters.chunk(2, dim=-1)
        log_std = log_std.clamp(-10.0, 2.0)
        return torch.distributions.Independent(
            torch.distributions.Normal(mean, log_std.exp().clamp(min=min_std)), 1
        )

    def encode(self, action_sequences, state_sequences=None):
        encoder_inputs = self.normalize_actions_tensor(action_sequences)
        if self.decoder_mode == CLOSED_LOOP_DECODER_MODE:
            if state_sequences is None:
                raise ValueError(
                    'closed_loop encoder requires aligned state sequences'
                )
            if state_sequences.shape[:-1] != action_sequences.shape[:-1]:
                raise ValueError(
                    'Incompatible state/action sequence shapes: {} and {}'
                    .format(tuple(state_sequences.shape),
                            tuple(action_sequences.shape))
                )
            encoder_inputs = torch.cat(
                [encoder_inputs, self.normalize_states(state_sequences)],
                dim=-1
            )
        _, (hidden, _) = self.encoder(encoder_inputs)
        return self._normal(self.encoder_head(hidden[-1]), self.min_std)

    def prior(self, states):
        return self._normal(
            self.prior_network(self.normalize_states(states)), self.min_std
        )

    def _closed_loop_mean(self, skills, states):
        if states is None:
            raise ValueError('closed_loop decoder requires current states')
        normalized_states = self.normalize_states(states)
        if normalized_states.ndim == skills.ndim:
            decoder_skills = skills
        elif normalized_states.ndim == skills.ndim + 1:
            decoder_skills = skills.unsqueeze(-2).expand(
                *normalized_states.shape[:-1], self.latent_dim
            )
        else:
            raise ValueError(
                'Incompatible state/skill shapes for closed_loop decoder: '
                '{} and {}'.format(tuple(states.shape), tuple(skills.shape))
            )
        return self.decoder_network(torch.cat(
            [normalized_states, decoder_skills], dim=-1
        ))

    def decode_distribution(self, skills, states=None):
        if self.decoder_mode == CLOSED_LOOP_DECODER_MODE:
            mean = self._closed_loop_mean(skills, states)
            return torch.distributions.Normal(mean, torch.ones_like(mean))

        output = self.decoder_network(skills)
        mean, log_std = output.chunk(2, dim=-1)
        mean = mean.view(-1, self.horizon, self.action_dim)
        log_std = log_std.clamp(-10.0, 2.0).view(-1, self.horizon, self.action_dim)
        return SquashedGaussian(mean, log_std.exp().clamp(min=self.min_std))

    def decode(self, skills, deterministic=False, states=None):
        distribution = self.decode_distribution(skills, states=states)
        if self.decoder_mode == CLOSED_LOOP_DECODER_MODE:
            normalized = distribution.mean
            if not deterministic and self.closed_loop_action_noise_std > 0:
                normalized = normalized + self.closed_loop_action_noise_std * \
                    torch.randn_like(normalized)
            normalized = normalized.tanh()
        else:
            normalized = (distribution.mean if deterministic
                          else distribution.sample())
        actions = self.denormalize_actions_tensor(normalized)
        return actions.clamp(-1.0, 1.0)

    def loss(self, states, action_sequences, state_sequences=None):
        posterior = self.encode(
            action_sequences, state_sequences=state_sequences
        )
        skills = posterior.rsample()
        if self.decoder_mode == CLOSED_LOOP_DECODER_MODE:
            reconstruction_mean = self._closed_loop_mean(
                skills, state_sequences
            )
            targets = self.normalize_actions_tensor(action_sequences)
            reconstruction_loss = 0.5 * (
                (reconstruction_mean - targets).pow(2).sum(dim=-1).mean()
            )
        else:
            reconstruction = self.decode_distribution(skills)
            targets = self.normalize_actions_tensor(action_sequences).clamp(
                -0.999, 0.999
            )
            reconstruction_loss = -reconstruction.log_prob(targets).sum(
                dim=-1
            ).mean()

        standard = torch.distributions.Independent(
            torch.distributions.Normal(
                torch.zeros_like(skills), torch.ones_like(skills)
            ), 1
        )
        kl_loss = torch.distributions.kl_divergence(posterior, standard).mean()
        prior_loss = -self.prior(states).log_prob(skills.detach()).mean()
        total = (reconstruction_loss + self.kl_weight * kl_loss
                 + self.prior_weight * prior_loss)
        return total, reconstruction_loss, kl_loss, prior_loss


def nnpu_loss(positive_logits, unlabeled_logits, positive_prior,
              slack=0.0, correction=True, correction_rate=1.0):
    positive_risk = positive_prior * F.softplus(-positive_logits).mean()
    negative_risk = (
        F.softplus(unlabeled_logits).mean()
        - positive_prior * F.softplus(positive_logits).mean()
    )
    if correction and negative_risk.detach().item() < -slack:
        objective = -correction_rate * negative_risk
    else:
        objective = positive_risk + torch.clamp(negative_risk, min=-slack)
    return objective, positive_risk.detach(), negative_risk.detach()


class SkillRiskPredictor(Configurable, Module):
    class Config(BaseConfig):
        hidden_dim = 256
        hidden_layers = 2
        learning_rate = 3e-4
        batch_size = 256
        pretrain_steps = 20000
        positive_prior = 0.0
        slack = 0.0
        nnpu_correction = True
        correction_rate = 1.0
        online_updates_per_epoch = 2
        max_grad_norm = 10.0

    def __init__(self, config, state_dim, latent_dim, state_normalizer):
        Configurable.__init__(self, config)
        Module.__init__(self)
        self.state_normalizer = state_normalizer
        dims = [state_dim + latent_dim] + [self.hidden_dim] * self.hidden_layers + [1]
        self.network = mlp(dims, squeeze_output=True)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.learning_rate)

    def logits(self, states, skills):
        normalized_states = self.state_normalizer(states)
        return self.network(torch.cat((normalized_states, skills), dim=-1))

    def risk(self, states, skills):
        return torch.sigmoid(self.logits(states, skills))

    def update(self, positive_states, positive_skills, unlabeled_states, unlabeled_skills):
        objective, positive_risk, negative_risk = nnpu_loss(
            self.logits(positive_states, positive_skills),
            self.logits(unlabeled_states, unlabeled_skills),
            self.positive_prior, self.slack, self.nnpu_correction,
            self.correction_rate
        )
        self.optimizer.zero_grad()
        objective.backward()
        nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
        self.optimizer.step()
        return {
            'PU objective': objective.detach().item(),
            'PU positive risk': positive_risk.item(),
            'PU negative risk': negative_risk.item(),
        }


class PairBuffer:

    def __init__(self, state_dim, skill_dim, capacity):
        self.capacity = capacity
        self.states = torch.empty(capacity, state_dim, device=device)
        self.skills = torch.empty(capacity, skill_dim, device=device)
        self.pointer = 0

    def __len__(self):
        return min(self.pointer, self.capacity)

    def append(self, states, skills):
        if states.ndim == 1:
            states, skills = states[None], skills[None]
        for state, skill in zip(states, skills):
            index = self.pointer % self.capacity
            self.states[index].copy_(state)
            self.skills[index].copy_(skill)
            self.pointer += 1

    def sample(self, batch_size):
        if len(self) == 0:
            raise ValueError('Cannot sample an empty pair buffer')
        indices = torch.randint(len(self), (batch_size,), device=device)
        return self.states[indices], self.skills[indices]


class SSKP(Configurable, Module):
    class Config(BaseConfig):
        demo_path = Require(str)
        skill_cfg = SkillModel.Config()
        sac_cfg = SAC.Config(squash_actions=False, target_entropy=1.0)
        risk_cfg = SkillRiskPredictor.Config()
        decision_mode = SKILL_HORIZON_MODE
        buffer_capacity = 10 ** 6
        pu_buffer_capacity = 200000
        steps_per_epoch = 1000
        policy_updates_per_epoch = 200
        initial_prior_steps = 5000
        cem_samples = 512
        cem_elites = 64
        cem_iterations = 6
        dual_cem = True
        weak_cem_sample_ratio = 0.25
        weak_cem_elite_ratio = 0.5
        weak_cem_iteration_ratio = 0.33
        cem_min_std = 0.05
        collect_cem_risk = False
        cem_risk_collect_interval = 100
        cem_risk_rank_bins = 32
        save_skill_transitions = True

    def __init__(self, config, env_factory, data):
        Configurable.__init__(self, config)
        Module.__init__(self)
        if self.decision_mode not in DECISION_MODES:
            raise ValueError(
                'decision_mode must be one of {}, got {!r}'.format(
                    DECISION_MODES, self.decision_mode
                )
            )
        self._validate_cem_settings(
            'strong', self.cem_samples, self.cem_elites, self.cem_iterations
        )
        if self.dual_cem:
            self._validate_weak_cem_ratios()
            self.weak_cem_samples = self._scaled_cem_value(
                self.cem_samples, self.weak_cem_sample_ratio
            )
            self.weak_cem_elites = self._scaled_cem_value(
                self.cem_elites, self.weak_cem_elite_ratio
            )
            self.weak_cem_iterations = self._scaled_cem_value(
                self.cem_iterations, self.weak_cem_iteration_ratio
            )
            self._validate_cem_settings(
                'weak', self.weak_cem_samples,
                self.weak_cem_elites, self.weak_cem_iterations
            )
        self.data = data
        self.real_env = env_factory()
        self.eval_env_factory = env_factory
        self.state_dim, self.action_dim = env_dims(self.real_env)
        self.max_episode_steps = get_max_episode_steps(self.real_env)
        self.skill_model = SkillModel(self.skill_cfg, self.state_dim, self.action_dim)
        self.sac = SAC(self.sac_cfg, self.state_dim, self.skill_cfg.latent_dim)
        self.risk_predictor = SkillRiskPredictor(
            self.risk_cfg, self.state_dim, self.skill_cfg.latent_dim,
            self.skill_model.normalize_states
        )
        replay = SafetySampleBuffer(
            self.state_dim, self.skill_cfg.latent_dim,
            self.buffer_capacity, device=device
        )
        self.replay_buffer = DummyModuleWrapper(replay)
        self.positive_pairs = PairBuffer(
            self.state_dim, self.skill_cfg.latent_dim, self.pu_buffer_capacity
        )
        self.unlabeled_pairs = PairBuffer(
            self.state_dim, self.skill_cfg.latent_dim, self.pu_buffer_capacity
        )

        self.register_buffer('epochs_completed', torch.zeros([], dtype=torch.long))
        self.register_buffer('steps_sampled', torch.zeros([], dtype=torch.long))
        self.register_buffer('episodes_sampled', torch.zeros([], dtype=torch.long))
        self.register_buffer('n_violations', torch.zeros([], dtype=torch.long))
        self.register_buffer('skill_pretrain_complete', torch.tensor(False))
        self.register_buffer('risk_pretrain_complete', torch.tensor(False))
        self.register_buffer(
            'best_eval_violations',
            torch.tensor(N_EVAL_EPISODES + 1, dtype=torch.long)
        )
        self.register_buffer('best_eval_return', torch.tensor(float('-inf')))
        self.register_buffer('best_eval_epoch', torch.tensor(-1, dtype=torch.long))
        self.episode_log = TabularLog(log.dir, 'episodes.csv')
        self._cem_training_calls = 0
        self._cem_risk_recorder = None
        self._is_setup = False

    @property
    def actor(self):
        return self.sac

    @staticmethod
    def _validate_cem_settings(name, samples, elites, iterations):
        if samples <= 0:
            raise ValueError('{} CEM samples must be positive'.format(name))
        if not 0 < elites <= samples:
            raise ValueError(
                '{} CEM elites must be in [1, samples]'.format(name)
            )
        if iterations <= 0:
            raise ValueError('{} CEM iterations must be positive'.format(name))

    @staticmethod
    def _scaled_cem_value(strong_value, ratio):
        return max(1, int(round(strong_value * ratio)))

    def _validate_weak_cem_ratios(self):
        for name, ratio in (
                ('sample', self.weak_cem_sample_ratio),
                ('elite', self.weak_cem_elite_ratio),
                ('iteration', self.weak_cem_iteration_ratio)):
            if not 0.0 < ratio <= 1.0:
                raise ValueError(
                    'weak CEM {} ratio must be in (0, 1]'.format(name)
                )

    def record_evaluation(self, result):
        violations = int(result['eval violations'])
        episode_return = float(result['eval return mean'])
        current_violations = self.best_eval_violations.item()
        current_return = self.best_eval_return.item()
        improved = (
            violations < current_violations or
            (violations == current_violations and
             episode_return > current_return)
        )
        if improved:
            self.best_eval_violations.fill_(violations)
            self.best_eval_return.fill_(episode_return)
            self.best_eval_epoch.fill_(self.epochs_completed.item())
        return improved

    def _skill_windows(self, episodes):
        states, state_sequences, actions = [], [], []
        horizon = self.skill_cfg.horizon
        for episode in episodes:
            for start in range(0, len(episode['actions']) - horizon + 1):
                states.append(episode['states'][start])
                state_sequences.append(
                    episode['states'][start:start + horizon]
                )
                actions.append(episode['actions'][start:start + horizon])
        if not states:
            raise ValueError('Demonstrations contain no complete skill windows of length {}'.format(horizon))
        return (torch.stack(states), torch.stack(state_sequences),
                torch.stack(actions))

    def _pretrain_skill_model(self, episodes):
        states, state_sequences, action_sequences = self._skill_windows(episodes)
        for _ in trange(self.skill_cfg.pretrain_steps, desc='pretrain skill model'):
            indices = torch.randint(
                len(states), (self.skill_cfg.batch_size,), device=device
            )
            losses = self.skill_model.loss(
                states[indices], action_sequences[indices],
                state_sequences=state_sequences[indices]
            )
            self.skill_model.optimizer.zero_grad()
            losses[0].backward()
            nn.utils.clip_grad_norm_(self.skill_model.parameters(), 10.0)
            self.skill_model.optimizer.step()
        self.skill_pretrain_complete.fill_(True)

    def _build_offline_pu_data(self, episodes):
        horizon = self.skill_cfg.horizon
        positive_count = 0
        unlabeled_count = 0
        with torch.no_grad():
            for episode in episodes:
                n_steps = len(episode['actions'])
                full_count = max(0, n_steps - horizon + 1)
                encoded = None
                if full_count:
                    sequences = torch.stack([
                        episode['actions'][start:start + horizon]
                        for start in range(full_count)
                    ])
                    state_sequences = torch.stack([
                        episode['states'][start:start + horizon]
                        for start in range(full_count)
                    ])
                    encoded = self.skill_model.encode(
                        sequences, state_sequences=state_sequences
                    ).mean
                for start in range(n_steps):
                    state = episode['states'][start]
                    if start < full_count:
                        skill = encoded[start]
                    else:
                        skill = self.skill_model.prior(state[None]).mean[0]
                    unsafe = episode['violations'][start:min(start + horizon, n_steps)].any()
                    if unsafe.item():
                        positive_count += 1
                        target = self.positive_pairs
                    else:
                        unlabeled_count += 1
                        target = self.unlabeled_pairs
                    target.append(state, skill)
        if positive_count == 0:
            raise ValueError(
                'SSkP needs at least one demonstration safety violation. '
                'Use HDF5 files with a violations field.'
            )
        if unlabeled_count == 0:
            raise ValueError('SSkP needs at least one unlabeled demonstration skill')
        if self.risk_predictor.positive_prior <= 0:
            self.risk_predictor.positive_prior = (
                float(positive_count)
                / float(positive_count + unlabeled_count)
            )
        if log.dir is not None:
            log(
                'Offline PU pairs: {} positive, {} unlabeled; retained {} and {}'
                .format(
                    positive_count, unlabeled_count,
                    len(self.positive_pairs), len(self.unlabeled_pairs)
                )
            )

    def _update_risk(self):
        batch_size = self.risk_cfg.batch_size
        positive = self.positive_pairs.sample(batch_size)
        unlabeled = self.unlabeled_pairs.sample(batch_size)
        return self.risk_predictor.update(*positive, *unlabeled)

    def _pretrain_risk_predictor(self):
        for _ in trange(self.risk_cfg.pretrain_steps, desc='pretrain PU risk'):
            self._update_risk()
        self.risk_pretrain_complete.fill_(True)

    def setup(self):
        if self._is_setup:
            return
        log('SSkP decision mode: {}'.format(self.decision_mode))
        log('SSkP decoder mode: {}'.format(self.skill_model.decoder_mode))
        if self.dual_cem:
            log(
                'SSkP dual CEM: strong {}/{}/{} and derived weak {}/{}/{}; '
                'selecting by target twin-Q'.format(
                    self.cem_samples, self.cem_elites, self.cem_iterations,
                    self.weak_cem_samples, self.weak_cem_elites,
                    self.weak_cem_iterations,
                )
            )
        else:
            log(
                'SSkP single CEM: {}/{}/{}'.format(
                    self.cem_samples, self.cem_elites, self.cem_iterations
                )
            )
        if self.collect_cem_risk:
            if self.cem_risk_collect_interval <= 0:
                raise ValueError('cem_risk_collect_interval must be positive')
            self._cem_risk_recorder = CEMRiskTraceRecorder(
                Path(log.dir) / TRACE_FILENAME,
                self.cem_samples,
                self.cem_elites,
                self.cem_iterations,
                self.cem_risk_rank_bins,
            )
            self._cem_training_calls = (
                self._cem_risk_recorder.next_planning_call
            )
            log(
                'Collecting compact CEM risk traces every {} training '
                'planning calls in {}'.format(
                    self.cem_risk_collect_interval,
                    self._cem_risk_recorder.path,
                )
            )
        self.demonstrations = load_demonstrations(self.demo_path)
        self.skill_model.set_normalization(self.demonstrations)
        if not self.skill_pretrain_complete.item():
            self._pretrain_skill_model(self.demonstrations)
        self._build_offline_pu_data(self.demonstrations)
        if not self.risk_pretrain_complete.item():
            self._pretrain_risk_predictor()
        self.skill_episodes_dir = Path(log.dir) / 'skill_episodes'
        if self.save_skill_transitions:
            self.skill_episodes_dir.mkdir(exist_ok=True)
            completed_episodes = self.episodes_sampled.item()
            paths = sorted(
                self.skill_episodes_dir.glob('episode-*.h5py'),
                key=_episode_file_number
            )
            paths = [
                path for path in paths
                if _episode_file_number(path) <= completed_episodes
            ]
            for path in paths:
                episode = SafetySampleBuffer.from_h5py(path, device=device)
                self.replay_buffer.extend(**episode.get(as_dict=True))
                states, skills, violations = episode.get(
                    'states', 'actions', 'violations'
                )
                if violations.any().item():
                    self.positive_pairs.append(states[violations], skills[violations])
                if (~violations).any().item():
                    self.unlabeled_pairs.append(states[~violations], skills[~violations])
        self._reset_episode()
        self._is_setup = True

    def _reset_episode(self):
        self._state = self.real_env.reset()
        self._episode_steps = 0
        self._episode_return = 0.0
        self._episode_skills = SafetySampleBuffer(
            self.state_dim, self.skill_cfg.latent_dim,
            self.max_episode_steps, device=device
        )

    def _proposal_distribution(self, state):
        if self.steps_sampled.item() < self.initial_prior_steps:
            return self.skill_model.prior(state)
        return self.sac.distribution(state)

    def _run_cem(self, state_batch, initial_mean, initial_std, samples,
                 elites, iterations, collect_trace=False):
        mean, std = initial_mean, initial_std
        trace = None
        if collect_trace:
            initial_risk = self.risk_predictor.risk(
                state_batch, mean[None]
            )[0]
            trace = {
                'candidate_risk_mean': [],
                'elite_risk_mean': [],
                'candidate_risk_min': [],
                'risk_rank_bins': [],
                'proposal_mean_risk': [initial_risk.item()],
                'proposal_std_mean': [std.mean().item()],
            }
        for _ in range(iterations):
            skills = mean + std * torch.randn(
                samples, self.skill_cfg.latent_dim, device=device
            )
            states = state_batch.expand(samples, -1)
            risks = self.risk_predictor.risk(states, skills)
            if collect_trace:
                summary = summarize_candidate_risks(
                    risks, elites, self.cem_risk_rank_bins
                )
                for name in (
                        'candidate_risk_mean', 'elite_risk_mean',
                        'candidate_risk_min'):
                    trace[name].append(summary[name].item())
                trace['risk_rank_bins'].append(
                    summary['risk_rank_bins'].cpu().numpy()
                )
            elite_indices = torch.topk(
                risks, elites, largest=False
            ).indices
            elite_skills = skills[elite_indices]
            mean = elite_skills.mean(dim=0)
            std = elite_skills.std(dim=0).clamp(min=self.cem_min_std)
            if collect_trace:
                mean_risk = self.risk_predictor.risk(
                    state_batch, mean[None]
                )[0]
                trace['proposal_mean_risk'].append(mean_risk.item())
                trace['proposal_std_mean'].append(std.mean().item())
        return mean, std, trace

    @staticmethod
    def _cem_candidate(mean, std, eval):
        return mean if eval else mean + std * torch.randn_like(mean)

    def plan_skill(self, state, eval=False):
        state_batch = state[None] if state.ndim == 1 else state
        planning_call = self._cem_training_calls
        collect_trace = (
            not eval and self.collect_cem_risk and
            planning_call % self.cem_risk_collect_interval == 0
        )
        if not eval:
            self._cem_training_calls += 1
        with torch.no_grad():
            proposal = self._proposal_distribution(state_batch)
            initial_mean = proposal.mean[0]
            initial_std = proposal.stddev[0]
            strong_mean, strong_std, trace = self._run_cem(
                state_batch, initial_mean, initial_std,
                self.cem_samples, self.cem_elites, self.cem_iterations,
                collect_trace=collect_trace,
            )
            strong_skill = self._cem_candidate(strong_mean, strong_std, eval)
            if self.dual_cem:
                weak_mean, weak_std, _ = self._run_cem(
                    state_batch, initial_mean, initial_std,
                    self.weak_cem_samples, self.weak_cem_elites,
                    self.weak_cem_iterations,
                )
                weak_skill = self._cem_candidate(
                    weak_mean, weak_std, eval
                )
                candidates = torch.stack((strong_skill, weak_skill))
                candidate_states = state_batch.expand(len(candidates), -1)
                candidate_q = self.sac.critic_target.minimum(
                    candidate_states, candidates
                )
                skill = candidates[torch.argmax(candidate_q)]
            else:
                skill = strong_skill
            if collect_trace:
                if self._cem_risk_recorder is None:
                    raise RuntimeError(
                        'CEM risk collection is enabled but setup() has not '
                        'initialized the recorder'
                    )
                self._cem_risk_recorder.append(
                    {
                        'planning_call': planning_call,
                        'steps_sampled': self.steps_sampled.item(),
                        'epoch': self.epochs_completed.item(),
                        'episode': self.episodes_sampled.item(),
                    },
                    trace,
                )
            return skill

    def _decoded_actions(self, skill, deterministic=False):
        if self.skill_model.decoder_mode == CLOSED_LOOP_DECODER_MODE:
            raise RuntimeError(
                'closed_loop skills must be decoded one action at a time'
            )
        with torch.no_grad():
            actions = self.skill_model.decode(
                skill[None], deterministic=deterministic
            )[0]
        if self.decision_mode == PER_ACTION_MODE:
            return actions[:1]
        return actions

    def _decoded_action(self, state, skill, deterministic=False):
        if self.skill_model.decoder_mode != CLOSED_LOOP_DECODER_MODE:
            raise RuntimeError('_decoded_action requires a closed_loop decoder')
        with torch.no_grad():
            action = self.skill_model.decode(
                skill[None], deterministic=deterministic,
                states=state[None]
            )[0]
        return action

    def _online_decision_pairs(self, decision_state, decision_skill,
                               intermediate_states):
        if self.decision_mode in (
                SKILL_HORIZON_CURRENT_PAIR_MODE, PER_ACTION_MODE):
            return decision_state[None], decision_skill[None]

        pair_states = torch.stack(intermediate_states)
        with torch.no_grad():
            pair_skills = self.skill_model.prior(pair_states).sample()
            pair_skills[0].copy_(decision_skill)
        return pair_states, pair_skills

    def _collect_skill(self):
        start_state = self._state
        skill = self.plan_skill(start_state)
        closed_loop = (
            self.skill_model.decoder_mode == CLOSED_LOOP_DECODER_MODE
        )
        if closed_loop:
            action_count = (1 if self.decision_mode == PER_ACTION_MODE
                            else self.skill_cfg.horizon)
            actions = None
        else:
            actions = self._decoded_actions(skill)
            action_count = len(actions)
        intermediate_states = []
        total_reward = 0.0
        done = False
        violation = False
        for action_index in range(action_count):
            intermediate_states.append(self._state)
            if closed_loop:
                action = self._decoded_action(self._state, skill)
            else:
                action = actions[action_index]
            next_state, reward, env_done, info = self.real_env.step(action)
            total_reward += reward
            self.steps_sampled += 1
            self._episode_steps += 1
            self._episode_return += reward
            self._state = next_state
            violation = bool(info['violation'])
            timeout = self._episode_steps >= self.max_episode_steps
            done = bool(env_done or violation or timeout)
            if done or self.steps_sampled.item() % self.steps_per_epoch == 0:
                break

        transition = dict(
            states=start_state, actions=skill, next_states=self._state,
            rewards=total_reward, dones=done, violations=violation
        )
        self.replay_buffer.append(**transition)
        self._episode_skills.append(**transition)

        pair_states, pair_skills = self._online_decision_pairs(
            start_state, skill, intermediate_states
        )
        target_buffer = self.positive_pairs if violation else self.unlabeled_pairs
        target_buffer.append(pair_states, pair_skills)

        if done:
            self.episodes_sampled += 1
            reported_violation = violation
            if violation:
                self.n_violations += 1
            row = {
                'episodes sampled': self.episodes_sampled.item(),
                'steps sampled': self.steps_sampled.item(),
                'total violations': self.n_violations.item(),
                'collect return': self._episode_return,
                'collect length': self._episode_steps,
            }
            for key, value in row.items():
                self.data.append(key, value, verbose=reported_violation)
            row['collect violation'] = reported_violation
            self.data.append('collect violation', reported_violation)
            self.episode_log.row(row)
            if self.save_skill_transitions:
                path = self.skill_episodes_dir / 'episode-{}.h5py'.format(
                    self.episodes_sampled.item()
                )
                self._episode_skills.save_h5py(path)
            self._reset_episode()

    def epoch(self):
        self.setup()
        target_steps = self.steps_sampled.item() + self.steps_per_epoch
        while self.steps_sampled.item() < target_steps:
            self._collect_skill()

        update_stats = []
        if len(self.replay_buffer) >= self.sac_cfg.batch_size:
            for _ in trange(self.policy_updates_per_epoch, desc='update skill SAC'):
                samples = self.replay_buffer.sample(self.sac_cfg.batch_size)
                update_stats.append(self.sac.update(samples, prior=self.skill_model.prior))
        risk_stats = []
        for _ in range(self.risk_cfg.online_updates_per_epoch):
            risk_stats.append(self._update_risk())
        if update_stats:
            for key in update_stats[0]:
                self.data.append(key, np.mean([row[key] for row in update_stats]))
        if risk_stats:
            for key in risk_stats[0]:
                self.data.append(key, np.mean([row[key] for row in risk_stats]))
        self.epochs_completed += 1

    def evaluate(self):
        returns, lengths, violations = [], [], []
        for _ in range(N_EVAL_EPISODES):
            env = self.eval_env_factory()
            state = env.reset()
            episode_return = 0.0
            episode_steps = 0
            episode_violation = False
            max_episode_steps = get_max_episode_steps(env)
            closed_loop = self.skill_model.decoder_mode == CLOSED_LOOP_DECODER_MODE
            while episode_steps < max_episode_steps:
                skill = self.plan_skill(state, eval=True)
                if closed_loop:
                    action_count = (1 if self.decision_mode == PER_ACTION_MODE
                                    else self.skill_cfg.horizon)
                    actions = None
                else:
                    actions = self._decoded_actions(skill, deterministic=True)
                    action_count = len(actions)
                stop = False
                for action_index in range(action_count):
                    if closed_loop:
                        action = self._decoded_action(
                            state, skill, deterministic=True
                        )
                    else:
                        action = actions[action_index]
                    state, reward, done, info = env.step(action)
                    episode_return += reward
                    episode_steps += 1
                    episode_violation = episode_violation or bool(info['violation'])
                    if done or info['violation'] or episode_steps >= max_episode_steps:
                        stop = True
                        break
                if stop:
                    break
            returns.append(episode_return)
            lengths.append(episode_steps)
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

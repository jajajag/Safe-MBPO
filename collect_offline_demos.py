"""Train SMBPO and collect the two offline datasets used by SSkP.

The first dataset contains the real environment transitions observed throughout
SMBPO training.  The second is collected after training with the final policy.
Both are written as per-episode HDF5 files compatible with ``src.sskp``.
"""

import json
from pathlib import Path

import h5py
import torch
from tqdm import tqdm, trange

from src import cli
from src.checkpoint import CheckpointableData, Checkpointer
from src.config import BaseConfig, Require
from src.env.util import get_max_episode_steps
from src.log import default_log as log
from src.shared import SafetySampleBuffer, get_env
from src.smbpo import SMBPO
from src.torch_util import device


class CollectionSMBPO(SMBPO):
    """SMBPO variant that skips costly evaluation inside every training episode."""

    def evaluate(self):
        # SMBPO.step_generator calls self.evaluate() at every episode boundary.
        # The normal ten-episode evaluation there would dominate a 1M-step
        # collection run. Scheduled evaluations call SMBPO.evaluate explicitly.
        return {}


    env_name = Require(str)
    seed = 1
    train_steps = 10 ** 7
    post_train_steps = 10 ** 6
    output_dir = ''
    deterministic_post_train = False
    evaluation_period_epochs = 10
    overwrite_datasets = False
    alg_cfg = SMBPO.Config()


def _episode_ranges(dones, violations, max_episode_steps):
    """Infer episode boundaries in a sequential replay buffer.

    Safe-MBPO resets on either termination, safety violation, or the Gym time
    limit.  Time-limit resets are not stored in ``dones``, so the counter is
    needed to recover those boundaries exactly.
    """
    start = 0
    episode_steps = 0
    for index in range(len(dones)):
        episode_steps += 1
        terminal = bool(dones[index].item() or violations[index].item())
        timeout = episode_steps >= max_episode_steps
        if terminal or timeout:
            yield start, index + 1
            start = index + 1
            episode_steps = 0
    if start < len(dones):
        yield start, len(dones)


def _dataset_is_complete(directory, expected_steps):
    metadata_path = directory / 'metadata.json'
    if not metadata_path.is_file():
        return False
    with metadata_path.open('r') as handle:
        metadata = json.load(handle)
    return metadata.get('num_transitions') == expected_steps


def _prepare_dataset_dir(directory, expected_steps, overwrite,
                         allow_incomplete_resume=False):
    directory.mkdir(parents=True, exist_ok=True)
    if _dataset_is_complete(directory, expected_steps):
        return False
    existing = list(directory.glob('episode-*.h5py'))
    metadata_path = directory / 'metadata.json'
    if existing or metadata_path.exists():
        if allow_incomplete_resume and not overwrite and not metadata_path.exists():
            return True
        if not overwrite:
            raise FileExistsError(
                '{} contains an incomplete dataset. Set '
                'overwrite_datasets=true to rebuild it.'.format(directory)
            )
        for path in existing:
            path.unlink()
        if metadata_path.exists():
            metadata_path.unlink()
    return True


def _write_metadata(directory, cfg, source, num_transitions, num_episodes):
    metadata = {
        'format_version': 1,
        'source': source,
        'env_name': cfg.env_name,
        'seed': cfg.seed,
        'num_transitions': num_transitions,
        'num_episodes': num_episodes,
        'smbpo_train_steps': cfg.train_steps,
        'deterministic_policy': (
            cfg.deterministic_post_train if source == 'fully_trained' else None
        ),
        'fields': [
            'states', 'actions', 'next_states', 'rewards', 'dones', 'violations'
        ],
    }
    with (directory / 'metadata.json').open('w') as handle:
        json.dump(metadata, handle, indent=2)


def export_training_replay(algorithm, directory, cfg):
    """Export all real transitions encountered during SMBPO training."""
    if not _prepare_dataset_dir(
            directory, cfg.train_steps, cfg.overwrite_datasets):
        log.message('Training-process dataset already complete: {}'.format(directory))
        return

    data = algorithm.replay_buffer.get(as_dict=True)
    if len(data['rewards']) != cfg.train_steps:
        raise RuntimeError(
            'Expected {} training transitions, found {}. Ensure alg_cfg.buffer_max '
            'is at least train_steps.'.format(cfg.train_steps, len(data['rewards']))
        )
    ranges = list(_episode_ranges(
        data['dones'], data['violations'],
        get_max_episode_steps(algorithm.real_env)
    ))
    for episode_number, (start, end) in enumerate(
            tqdm(ranges, desc='export training replay'), 1):
        episode = SafetySampleBuffer(
            algorithm.state_dim, algorithm.action_dim, end - start, device=device
        )
        episode.extend(**{
            name: values[start:end] for name, values in data.items()
        })
        episode.save_h5py(
            directory / 'episode-{}.h5py'.format(episode_number),
            remove_duplicate_states=False
        )
    _write_metadata(
        directory, cfg, 'training_process', cfg.train_steps, len(ranges)
    )


def _policy_action(algorithm, state, deterministic):
    with torch.no_grad():
        distribution = algorithm.solver.actor.distr(state.unsqueeze(0))
        action = distribution.mean if deterministic else distribution.sample()
    return action[0]


def collect_fully_trained(algorithm, directory, cfg):
    """Collect fresh real-environment data with the final SMBPO policy."""
    if not _prepare_dataset_dir(
            directory, cfg.post_train_steps, cfg.overwrite_datasets,
            allow_incomplete_resume=True):
        log.message('Fully-trained dataset already complete: {}'.format(directory))
        return

    existing_paths = sorted(
        directory.glob('episode-*.h5py'),
        key=lambda path: int(path.stem.split('-')[-1])
    )
    already_collected = 0
    for path in existing_paths:
        with h5py.File(str(path), 'r') as handle:
            already_collected += len(handle['rewards'])
    if already_collected > cfg.post_train_steps:
        raise RuntimeError(
            '{} already contains {} transitions, beyond requested {}'.format(
                directory, already_collected, cfg.post_train_steps
            )
        )
    env = get_env(cfg.env_name)
    max_episode_steps = get_max_episode_steps(env)
    state = env.reset()
    episode = SafetySampleBuffer(
        algorithm.state_dim, algorithm.action_dim, max_episode_steps, device=device
    )
    episode_number = len(existing_paths)
    episode_steps = 0

    progress = tqdm(
        total=cfg.post_train_steps, initial=already_collected,
        desc='collect fully trained'
    )
    for step in range(already_collected, cfg.post_train_steps):
        action = _policy_action(
            algorithm, state, cfg.deterministic_post_train
        )
        next_state, reward, done, info = env.step(action)
        violation = bool(info['violation'])
        episode_steps += 1
        timeout = episode_steps >= max_episode_steps
        episode_done = bool(done or violation or timeout)
        episode.append(
            states=state, actions=action, next_states=next_state,
            rewards=reward, dones=bool(done), violations=violation
        )
        progress.update(1)

        final_step = step + 1 == cfg.post_train_steps
        if episode_done or final_step:
            episode_number += 1
            episode.save_h5py(
                directory / 'episode-{}.h5py'.format(episode_number),
                remove_duplicate_states=False
            )
            if not final_step:
                state = env.reset()
                episode = SafetySampleBuffer(
                    algorithm.state_dim, algorithm.action_dim,
                    max_episode_steps, device=device
                )
                episode_steps = 0
        else:
            state = next_state
    progress.close()
    env.close()
    _write_metadata(
        directory, cfg, 'fully_trained', cfg.post_train_steps, episode_number
    )


def _train_exact_steps(algorithm, target_steps):
    if algorithm.steps_sampled.item() > target_steps:
        raise RuntimeError(
            'Loaded checkpoint already has {} steps, beyond requested {}'.format(
                algorithm.steps_sampled.item(), target_steps
            )
        )
    while algorithm.steps_sampled.item() < target_steps:
        block_steps = min(
            algorithm.steps_per_epoch,
            target_steps - algorithm.steps_sampled.item()
        )
        for _ in trange(block_steps, desc='train SMBPO'):
            next(algorithm.stepper)
        algorithm.epochs_completed += 1
        yield


def main(cfg):
    if cfg.train_steps <= 0 or cfg.post_train_steps <= 0:
        raise ValueError('train_steps and post_train_steps must be positive')
    if cfg.alg_cfg.buffer_max < cfg.train_steps:
        raise ValueError('alg_cfg.buffer_max must be at least train_steps')

    # Per-episode files are also used to reconstruct SMBPO's replay buffer when
    # resuming the long training run.
    cfg.alg_cfg.save_trajectories = True
    env_factory = lambda: get_env(cfg.env_name)
    data = CheckpointableData()
    algorithm = CollectionSMBPO(cfg.alg_cfg, env_factory, data)
    algorithm.to(device)
    algorithm_checkpointer = Checkpointer(
        algorithm, log.dir, 'offline_collector_smbpo.pt'
    )
    data_checkpointer = Checkpointer(
        data, log.dir, 'offline_collector_data.pt'
    )
    if data_checkpointer.try_load():
        if algorithm_checkpointer.try_load():
            log.message('Resumed SMBPO collector checkpoint')

    algorithm.setup()
    for _ in _train_exact_steps(algorithm, cfg.train_steps):
        epoch = algorithm.epochs_completed.item()
        if cfg.evaluation_period_epochs > 0 and (
                epoch % cfg.evaluation_period_epochs == 0):
            evaluation = SMBPO.evaluate(algorithm)
            for key, value in evaluation.items():
                data.append(key, value, verbose=True)

        # A single rolling checkpoint keeps long collection runs resumable.
        algorithm_checkpointer.save()
        data_checkpointer.save()

    output_root = (Path(cfg.output_dir) if cfg.output_dir
                   else Path(log.dir) / 'offline_demos')
    output_root.mkdir(parents=True, exist_ok=True)
    export_training_replay(
        algorithm, output_root / 'training_process', cfg
    )
    collect_fully_trained(
        algorithm, output_root / 'fully_trained', cfg
    )
    log.message('Offline demonstrations written to {}'.format(output_root))


if __name__ == '__main__':
    cli.main(Config(), main)

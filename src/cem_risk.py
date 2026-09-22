"""Compact training-time CEM risk traces for SSkP.

The recorder keeps distribution summaries rather than all sampled skills or
rollout tensors.  Each recorded planning call contains per-CEM-iteration
population statistics and a small number of equal-rank risk bins.
"""

from pathlib import Path

import h5py
import numpy as np
import torch


TRACE_FILENAME = 'cem_risk_training.h5'
TRACE_SCHEMA_VERSION = 1


def summarize_candidate_risks(risks, elite_count, rank_bin_count):
    """Return compact scalar and rank-bin summaries for one CEM population."""
    if risks.ndim != 1:
        raise ValueError('candidate risks must be a one-dimensional tensor')
    sample_count = len(risks)
    if not 0 < elite_count <= sample_count:
        raise ValueError('elite_count must be in [1, number of candidates]')
    if not 0 < rank_bin_count <= sample_count:
        raise ValueError('rank_bin_count must be in [1, number of candidates]')

    sorted_risks = torch.sort(risks).values
    # Integer linspace makes the bins exhaustive even when the candidate count
    # is not exactly divisible by the requested bin count.
    edges = np.linspace(
        0, sample_count, rank_bin_count + 1, dtype=np.int64
    )
    rank_bins = torch.stack([
        sorted_risks[int(edges[index]):int(edges[index + 1])].mean()
        for index in range(rank_bin_count)
    ])
    return {
        'candidate_risk_mean': risks.mean(),
        'elite_risk_mean': sorted_risks[:elite_count].mean(),
        'candidate_risk_min': sorted_risks[0],
        'risk_rank_bins': rank_bins,
    }


class CEMRiskTraceRecorder:
    """Append compressed, fixed-shape CEM traces to an HDF5 file."""

    _SCALAR_METADATA = (
        ('planning_call', np.int64),
        ('steps_sampled', np.int64),
        ('epoch', np.int64),
        ('episode', np.int64),
    )
    _ITERATION_METRICS = (
        'candidate_risk_mean',
        'elite_risk_mean',
        'candidate_risk_min',
    )

    def __init__(self, path, cem_samples, cem_elites, cem_iterations,
                 rank_bin_count):
        self.path = Path(path)
        self.cem_samples = int(cem_samples)
        self.cem_elites = int(cem_elites)
        self.cem_iterations = int(cem_iterations)
        self.rank_bin_count = int(rank_bin_count)
        if not 0 < self.cem_elites <= self.cem_samples:
            raise ValueError('cem_elites must be in [1, cem_samples]')
        if self.cem_iterations <= 0:
            raise ValueError('cem_iterations must be positive')
        if not 0 < self.rank_bin_count <= self.cem_samples:
            raise ValueError('rank_bin_count must be in [1, cem_samples]')
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_file()

    def _initialize_file(self):
        with h5py.File(str(self.path), 'a') as handle:
            settings = {
                'schema_version': TRACE_SCHEMA_VERSION,
                'cem_samples': self.cem_samples,
                'cem_elites': self.cem_elites,
                'cem_iterations': self.cem_iterations,
                'rank_bin_count': self.rank_bin_count,
            }
            for name, value in settings.items():
                if name in handle.attrs and int(handle.attrs[name]) != value:
                    raise ValueError(
                        '{} was created with {}={}, but the current run uses {}'
                        .format(self.path, name, handle.attrs[name], value)
                    )
                handle.attrs[name] = value
            handle.attrs['rank_order'] = 'ascending_risk'
            handle.attrs['elite_definition'] = 'lowest_risk_candidates'

            for name, dtype in self._SCALAR_METADATA:
                self._require_dataset(handle, name, (), dtype)
            for name in self._ITERATION_METRICS:
                self._require_dataset(
                    handle, name, (self.cem_iterations,), np.float32
                )
            self._require_dataset(
                handle, 'risk_rank_bins',
                (self.cem_iterations, self.rank_bin_count), np.float32
            )
            self._require_dataset(
                handle, 'proposal_mean_risk',
                (self.cem_iterations + 1,), np.float32
            )
            self._require_dataset(
                handle, 'proposal_std_mean',
                (self.cem_iterations + 1,), np.float32
            )

    @property
    def next_planning_call(self):
        """Return a non-duplicating call index when appending after resume."""
        with h5py.File(str(self.path), 'r') as handle:
            calls = handle['planning_call']
            return 0 if len(calls) == 0 else int(calls[-1]) + 1

    @staticmethod
    def _require_dataset(handle, name, tail_shape, dtype):
        expected_tail = tuple(tail_shape)
        if name in handle:
            actual_tail = tuple(handle[name].shape[1:])
            if actual_tail != expected_tail:
                raise ValueError(
                    'Dataset {} has shape tail {}, expected {}'
                    .format(name, actual_tail, expected_tail)
                )
            return
        # Group several sampled calls per chunk so long runs do not pay HDF5
        # chunk metadata/compression overhead once per individual trace.
        chunk_shape = (64,) + expected_tail
        handle.create_dataset(
            name,
            shape=(0,) + expected_tail,
            maxshape=(None,) + expected_tail,
            chunks=chunk_shape,
            dtype=dtype,
            compression='gzip',
            compression_opts=4,
            shuffle=True,
        )

    def append(self, metadata, trace):
        """Append one sampled planning call and flush it to disk."""
        payload = {}
        for name, _ in self._SCALAR_METADATA:
            payload[name] = metadata[name]
        for name in self._ITERATION_METRICS:
            payload[name] = np.asarray(trace[name], dtype=np.float32)
        payload['risk_rank_bins'] = np.asarray(
            trace['risk_rank_bins'], dtype=np.float32
        )
        payload['proposal_mean_risk'] = np.asarray(
            trace['proposal_mean_risk'], dtype=np.float32
        )
        payload['proposal_std_mean'] = np.asarray(
            trace['proposal_std_mean'], dtype=np.float32
        )

        expected_shapes = {
            name: (self.cem_iterations,)
            for name in self._ITERATION_METRICS
        }
        expected_shapes.update({
            'risk_rank_bins': (
                self.cem_iterations, self.rank_bin_count
            ),
            'proposal_mean_risk': (self.cem_iterations + 1,),
            'proposal_std_mean': (self.cem_iterations + 1,),
        })
        for name, expected in expected_shapes.items():
            if payload[name].shape != expected:
                raise ValueError(
                    '{} has shape {}, expected {}'
                    .format(name, payload[name].shape, expected)
                )

        with h5py.File(str(self.path), 'a') as handle:
            index = len(handle['planning_call'])
            for name, value in payload.items():
                dataset = handle[name]
                dataset.resize(index + 1, axis=0)
                dataset[index] = value
            handle.flush()

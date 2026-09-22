"""Plot compact training-time SSkP CEM risk traces.

The input is ``cem_risk_training.h5`` produced when
``alg_cfg.collect_cem_risk`` is enabled.  Two independent figures are written:

* three per-iteration curves for the candidate-population mean, lowest-risk
  elite mean, and minimum candidate risk;
* a candidate-risk-rank heatmap showing how the whole sampled distribution
  changes across CEM iterations.
"""

import argparse
from pathlib import Path

import h5py
import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

from src.cem_risk import TRACE_FILENAME


CURVE_DATASETS = (
    ('candidate_risk_mean', 'All-Skill Mean'),
    ('elite_risk_mean', r'Top-$k$ Mean'),
    ('candidate_risk_min', 'Lowest-Risk'),
)

ENVIRONMENT_NAMES = (
    ('ant', 'Ant'),
    ('cheetah', 'Cheetah'),
    ('hopper', 'Hopper'),
    ('humanoid', 'Humanoid'),
)


# Match the compact, uncluttered style used by the paper figures.
PAPER_STYLE = {
    'font.family': 'sans-serif',
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'axes.linewidth': 0.8,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 8,
    'lines.linewidth': 1.25,
    'lines.markersize': 3.5,
}


def apply_paper_style(axis):
    axis.grid(False)
    axis.tick_params(width=0.8, length=3)
    for spine in axis.spines.values():
        spine.set_linewidth(0.8)


def resolve_trace_path(path):
    path = Path(path).expanduser().resolve()
    return path / TRACE_FILENAME if path.is_dir() else path


def infer_environment_name(trace_path):
    path_text = str(trace_path).lower()
    for token, display_name in ENVIRONMENT_NAMES:
        if token in path_text:
            return display_name
    return trace_path.parent.name


def load_trace(path, min_step=None, max_step=None):
    with h5py.File(str(path), 'r') as handle:
        required = [name for name, _ in CURVE_DATASETS]
        required.extend(('risk_rank_bins', 'steps_sampled'))
        missing = [name for name in required if name not in handle]
        if missing:
            raise ValueError(
                '{} is missing datasets: {}'.format(path, ', '.join(missing))
            )
        steps = np.asarray(handle['steps_sampled'])
        mask = np.ones(len(steps), dtype=np.bool_)
        if min_step is not None:
            mask &= steps >= min_step
        if max_step is not None:
            mask &= steps <= max_step
        if not mask.any():
            raise ValueError('No traces remain after applying the step filters')
        data = {
            name: np.asarray(handle[name])[mask]
            for name in required if name != 'steps_sampled'
        }
        data['steps_sampled'] = steps[mask]
        data['attrs'] = {name: handle.attrs[name] for name in handle.attrs}
    return data


def relative_remaining(values, baseline, epsilon):
    """Express risks relative to a shared per-call baseline."""
    return 100.0 * values / np.maximum(baseline, epsilon)


def apply_risk_floor_ratio(data, ratio):
    """Map predicted risks from [0, 1] to [ratio, 1].

    The affine transformation ``ratio + (1 - ratio) * risk`` preserves the
    ordering of risk predictions while imposing ``ratio`` as their minimum.
    A ratio of zero leaves the recorded values unchanged.
    """
    if not np.isfinite(ratio) or not 0 <= ratio <= 1:
        raise ValueError('risk ratio must be between 0 and 1')

    for dataset, _ in CURVE_DATASETS:
        values = data[dataset].astype(np.float64)
        data[dataset] = ratio + (1.0 - ratio) * values
    bins = data['risk_rank_bins'].astype(np.float64)
    data['risk_rank_bins'] = (
        ratio + (1.0 - ratio) * bins
        )
    return data


def plot_three_curves(path, data, normalization, epsilon, environment_name):
    figure, axis = plt.subplots(figsize=(5.2, 3.7))
    iteration_count = data[CURVE_DATASETS[0][0]].shape[1]
    # Display the six sampling rounds as 1--6 instead of zero-based indices.
    x = np.arange(1, iteration_count + 1)
    # Use the first-round all-skill mean as the common denominator for every
    # curve.  A shared baseline preserves the pointwise ordering
    # minimum <= top-k mean <= all-skill mean after normalization.
    relative_baseline = data['candidate_risk_mean'][:, :1].astype(np.float64)

    for dataset, generic_label in CURVE_DATASETS:
        values = data[dataset].astype(np.float64)
        plotted = (
            relative_remaining(values, relative_baseline, epsilon)
            if normalization == 'relative' else values
        )
        center = np.nanmedian(plotted, axis=0)
        lower = np.nanpercentile(plotted, 25, axis=0)
        upper = np.nanpercentile(plotted, 75, axis=0)
        label = generic_label
        axis.plot(x, center, marker='o', label=label)
        axis.fill_between(x, lower, upper, alpha=0.12, linewidth=0)

    axis.set_xticks(x)
    axis.set_xlabel('Iteration Number')
    if normalization == 'relative':
        axis.set_ylabel('Relative Predicted Risk (%)')
    else:
        axis.set_ylabel('Predicted Risk')
    axis.set_title(environment_name)
    apply_paper_style(axis)
    axis.legend(frameon=True, fancybox=False, edgecolor='0.75')
    figure.tight_layout()
    figure.savefig(str(path), dpi=300, bbox_inches='tight')
    plt.close(figure)


def plot_rank_heatmap(path, data, normalization, epsilon, environment_name):
    # Stored order is low risk -> high risk.  Transpose after aggregating so
    # the y-axis is candidate risk percentile and x is CEM iteration.
    bins = data['risk_rank_bins'].astype(np.float64)
    iteration_count = bins.shape[1]
    bin_count = bins.shape[2]

    if normalization == 'relative':
        per_trace = np.log10(
            (bins + epsilon) / (bins[:, :1, :] + epsilon)
        )
        heatmap = np.nanmedian(per_trace, axis=0).T
        finite = np.abs(heatmap[np.isfinite(heatmap)])
        color_limit = (
            max(float(np.percentile(finite, 95)), 1e-6)
            if len(finite) else 1.0
        )
        image_kwargs = dict(
            cmap='RdBu_r', vmin=-color_limit, vmax=color_limit
        )
        color_label = 'Relative Predicted Risk (log10)'
    else:
        heatmap = np.nanmedian(np.log10(bins + epsilon), axis=0).T
        image_kwargs = dict(cmap='viridis')
        color_label = 'Predicted Risk (log10)'

    figure, axis = plt.subplots(figsize=(5.2, 3.7))
    image = axis.imshow(
        heatmap,
        origin='lower',
        aspect='auto',
        interpolation='nearest',
        extent=(0.5, iteration_count + 0.5, 0.0, 100.0),
        **image_kwargs
    )
    axis.set_xticks(np.arange(1, iteration_count + 1))
    axis.set_xlabel('Iteration Number')
    axis.set_ylabel('Predicted Risk Percentile')
    axis.set_title(environment_name)
    apply_paper_style(axis)
    colorbar = figure.colorbar(image, ax=axis)
    colorbar.set_label(color_label)
    colorbar.ax.tick_params(labelsize=9, width=0.8, length=3)
    colorbar.outline.set_linewidth(0.8)
    figure.tight_layout()
    figure.savefig(str(path), dpi=300, bbox_inches='tight')
    plt.close(figure)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        'trace', type=Path,
        help='Training run directory or path to {}'.format(TRACE_FILENAME),
    )
    parser.add_argument(
        '--output-dir', type=Path, default=None,
        help='Default: TRACE_PARENT/cem_risk_plots',
    )
    parser.add_argument(
        '--normalization', choices=('relative', 'absolute'),
        default='relative',
        help='relative compares each planning call with its first CEM sample',
    )
    parser.add_argument('--epsilon', type=float, default=1e-8)
    parser.add_argument(
        '--risk-ratio', type=float, default=0.0,
        help=(
            'Minimum plotted risk ratio: adjusted risk equals risk_ratio + '
            '(1 - risk_ratio) * stored risk (default: 0)'
        ),
    )
    parser.add_argument('--min-step', type=int, default=None)
    parser.add_argument('--max-step', type=int, default=None)
    parser.add_argument(
        '--environment', type=str, default=None,
        help='Plot title; inferred from the trace path when omitted',
    )
    return parser.parse_args()


def main():
    args = parse_args()
    plt.rcParams.update(PAPER_STYLE)
    if args.epsilon <= 0:
        raise ValueError('epsilon must be positive')
    trace_path = resolve_trace_path(args.trace)
    if not trace_path.is_file():
        raise FileNotFoundError('CEM risk trace does not exist: {}'.format(
            trace_path
        ))
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else trace_path.parent / 'cem_risk_plots'
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    data = load_trace(trace_path, args.min_step, args.max_step)
    data = apply_risk_floor_ratio(data, args.risk_ratio)
    environment_name = (
        args.environment
        if args.environment is not None
        else infer_environment_name(trace_path)
    )

    curve_path = output_dir / 'cem_risk_three_curves.png'
    heatmap_path = output_dir / 'cem_risk_rank_heatmap.png'
    plot_three_curves(
        curve_path, data, args.normalization, args.epsilon, environment_name
    )
    plot_rank_heatmap(
        heatmap_path, data, args.normalization, args.epsilon, environment_name
    )
    print('Wrote {}'.format(curve_path))
    print('Wrote {}'.format(heatmap_path))


if __name__ == '__main__':
    main()

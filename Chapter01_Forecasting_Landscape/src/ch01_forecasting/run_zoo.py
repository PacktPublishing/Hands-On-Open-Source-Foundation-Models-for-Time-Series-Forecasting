"""The Forecasting Zoo: ETTh1, 96 hourly steps, three explicit distributions.

Run: python scripts/run_zoo.py --models all
New results go to a fresh runs/ directory; bundled reference outputs stay intact.
Default protocol is fixed before evaluation. This is a bounded teaching run,
not a tuned benchmark or an assertion of uncontaminated zero-shot transfer.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import platform
import time
import traceback
import urllib.request

import numpy as np
import pandas as pd
from .metrics import score_forecast, verify_scores
from .artifacts import fresh_output_dir, package_versions, utc_now

ROOT = Path(__file__).resolve().parents[2]
DATA_REVISION = '1d16c8f4f943005d613b5bc962e9eeb06058cf07'
DATA_SHA256 = 'f18de3ad269cef59bb07b5438d79bb3042d3be49bdeecf01c1cd6d29695ee066'
DATA_URL = (f'https://raw.githubusercontent.com/zhouhaoyi/ETDataset/'
            f'{DATA_REVISION}/ETT-small/ETTh1.csv')
MODEL_ID = 'amazon/chronos-t5-small'
MODEL_REVISION = 'a971ba21945c4f1796b17a91fe69214b5f4ad472'
HORIZON, TRAIN_END, VAL_END, TEST_END = 96, 8640, 11520, 14400
# Three non-overlapping windows at the start of the original test segment.
ORIGINS = (11520, 11616, 11712)
SEED, N_SAMPLES = 20261114, 128


def load_data(*, download=True):
    path = ROOT / 'data' / 'ETTh1.csv'
    if not path.exists():
        if not download:
            raise FileNotFoundError(f'Bundled dataset is missing: {path}')
        with urllib.request.urlopen(DATA_URL, timeout=60) as response:
            payload = response.read()
        if hashlib.sha256(payload).hexdigest() != DATA_SHA256:
            raise ValueError('Downloaded dataset checksum mismatch; no file was saved')
        path.parent.mkdir(exist_ok=True, parents=True)
        # The small download is verified in memory before touching its cache.
        path.write_bytes(payload)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != DATA_SHA256:
        raise ValueError(f'Dataset checksum mismatch: {digest}')
    raw = pd.read_csv(path, parse_dates=['date'])
    if raw['date'].duplicated().any() or not raw['date'].is_monotonic_increasing:
        raise ValueError('Timestamps must be unique and increasing')
    if not raw['date'].diff().dropna().eq(pd.Timedelta(hours=1)).all():
        raise ValueError('ETTh1 is not regularly hourly')
    if raw['OT'].isna().any() or not np.isfinite(raw['OT']).all():
        raise ValueError('Target has missing or nonfinite values')
    frame = raw.rename(columns={'date': 'ds', 'OT': 'y'})[['ds', 'y']].copy()
    frame.insert(0, 'unique_id', 'ETTh1_OT')
    return frame


def gaussian_samples(location, scale, rng):
    """Marginal samples; these are not coherent joint temporal trajectories."""
    location, scale = np.asarray(location), np.asarray(scale)
    if not np.isfinite(location).all() or not np.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError('Invalid predictive Normal parameters')
    return rng.normal(location, scale, size=(N_SAMPLES, HORIZON))


def arima_forecasts(frame):
    from scipy.stats import norm
    from statsforecast.models import AutoARIMA
    rng = np.random.default_rng(SEED + 1)
    outputs = []
    for origin in ORIGINS:
        model = AutoARIMA(season_length=24, max_p=3, max_q=3,
                          max_P=1, max_Q=1, max_order=5,
                          stepwise=True, approximation=True)
        fitted = model.fit(frame.y.iloc[:origin].to_numpy())
        pred = fitted.predict(h=HORIZON, level=[80, 95])
        loc = np.asarray(pred['mean'])
        scale = (np.asarray(pred['hi-95']) - np.asarray(pred['lo-95'])) / (2 * norm.ppf(.975))
        samples = gaussian_samples(loc, scale, rng)
        outputs.append((origin, samples, loc))
    return outputs, {'season_length': 24, 'order_search': 'bounded stepwise, approximate selection',
                     'refit': 'at every origin on all available preceding values',
                     'distribution': 'native Normal marginal prediction',
                     'max_p': 3, 'max_q': 3, 'max_P': 1, 'max_Q': 1}


def nhits_forecasts(frame, max_steps):
    import torch
    from neuralforecast import NeuralForecast
    from neuralforecast.models import NHITS
    from neuralforecast.losses.pytorch import DistributionLoss
    torch.set_num_threads(2)
    torch.manual_seed(SEED)
    # A probabilistic loss is essential: default NHITS MAE is point-only.
    loss = DistributionLoss(distribution='Normal', level=[80, 95],
                            num_samples=N_SAMPLES, return_params=True)
    model = NHITS(h=HORIZON, input_size=192, loss=loss,
                  max_steps=max_steps, val_check_steps=50,
                  early_stop_patience_steps=-1, learning_rate=1e-3,
                  scaler_type='standard', random_seed=SEED,
                  batch_size=1, windows_batch_size=64,
                  n_blocks=[1, 1, 1], mlp_units=[[128, 128]] * 3,
                  accelerator='cpu', devices=1, logger=False,
                  enable_checkpointing=False, enable_progress_bar=False,
                  alias='NHITS_Normal')
    nf = NeuralForecast(models=[model], freq='h')
    nf.fit(df=frame.iloc[:VAL_END], val_size=VAL_END-TRAIN_END)
    rng = np.random.default_rng(SEED + 2)
    outputs = []
    for origin in ORIGINS:
        pred = nf.predict(df=frame.iloc[:origin])
        loc_name = next(c for c in pred.columns if c.endswith('-loc'))
        scale_name = next(c for c in pred.columns if c.endswith('-scale'))
        loc, scale = pred[loc_name].to_numpy(), pred[scale_name].to_numpy()
        outputs.append((origin, gaussian_samples(loc, scale, rng), loc))
    return outputs, {'input_size': 192, 'max_steps': max_steps,
                     'training_targets': [0, TRAIN_END], 'validation_targets': [TRAIN_END, VAL_END],
                     'validation_use': 'loss monitoring only; fixed budget, no early stopping',
                     'scaler': 'training-window standard scaler inside NeuralForecast',
                     'distribution': 'Normal negative log likelihood, learned loc and scale',
                     'weights': 'frozen after fit; context updated at each origin',
                     'hidden_layers': [[128, 128]] * 3, 'windows_batch_size': 64}


def chronos_forecasts(frame, sample_batch=N_SAMPLES):
    import torch
    from chronos import ChronosPipeline
    torch.set_num_threads(2)
    torch.manual_seed(SEED)
    pipeline = ChronosPipeline.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION,
        device_map='cpu', torch_dtype=torch.float32)
    outputs = []
    for origin in ORIGINS:
        torch.manual_seed(SEED + origin)
        context = torch.tensor(frame.y.iloc[origin-512:origin].to_numpy(), dtype=torch.float32)
        # Native horizon is 64; longer requests use the documented median-chunk extension.
        if sample_batch >= N_SAMPLES:
            # Preserve the exact API call used for the recorded experiment.
            pred = pipeline.predict(context, prediction_length=HORIZON,
                                    num_samples=N_SAMPLES, limit_prediction_length=False,
                                    temperature=1.0, top_k=50, top_p=1.0)
            samples = pred[0].detach().cpu().numpy()
        else:
            # Optional lower-memory run: pool all samples BEFORE forming the
            # common median context for the next native-horizon chunk.
            # Different batch grouping changes RNG consumption; this is not
            # guaranteed to reproduce the recorded seed's exact samples.
            chunks = []
            for start in range(0, HORIZON, 64):
                length = min(64, HORIZON - start)
                batches = []
                for offset in range(0, N_SAMPLES, sample_batch):
                    count = min(sample_batch, N_SAMPLES - offset)
                    pred = pipeline.predict(
                        context, prediction_length=length, num_samples=count,
                        limit_prediction_length=True,
                        temperature=1.0, top_k=50, top_p=1.0)
                    batches.append(pred[0].detach().cpu().numpy())
                chunk = np.concatenate(batches, axis=0)
                chunks.append(chunk)
                median = torch.tensor(np.median(chunk, axis=0), dtype=context.dtype)
                context = torch.cat([context, median])[-512:]
            samples = np.concatenate(chunks, axis=1)
        if samples.shape != (N_SAMPLES, HORIZON):
            raise ValueError(f'Unexpected Chronos sample shape {samples.shape}')
        outputs.append((origin, samples, np.median(samples, axis=0)))
    return outputs, {'checkpoint': MODEL_ID, 'revision': MODEL_REVISION,
                     'context': 512, 'native_horizon': 64, 'requested_horizon': HORIZON,
                     'generation_sample_batch': sample_batch,
                     'long_horizon_extension': 'median of previous generated chunk fed into next chunk',
                     'temperature': 1.0, 'top_k': 50, 'top_p': 1.0,
                     'weights': 'pretrained and frozen', 'pretraining_overlap': 'unknown',
                     'joint_path_uncertainty_beyond_first_chunk': 'not fully propagated'}


def save_model(name, forecasts, frame, results_dir):
    results_dir = Path(results_dir)
    rows, summaries = [], []
    arrays = {}
    for origin, samples, point in forecasts:
        actual = frame.y.iloc[origin:origin+HORIZON].to_numpy()
        summaries.append({'model': name, 'origin_index': origin,
                          **score_forecast(actual, samples, point)})
        arrays[f'samples_{origin}'] = samples
        arrays[f'point_{origin}'] = point
        arrays[f'actual_{origin}'] = actual
        quantiles = np.quantile(samples, [.025, .1, .5, .9, .975], axis=0)
        for h in range(HORIZON):
            rows.append({'model': name, 'origin_index': origin, 'horizon': h+1,
                         'ds': str(frame.ds.iloc[origin+h]), 'actual': actual[h],
                         'point': point[h], 'lo95': quantiles[0,h], 'lo80': quantiles[1,h],
                         'hi80': quantiles[3,h], 'hi95': quantiles[4,h]})
    slug = name.lower().replace('-', '_')
    pd.DataFrame(rows).to_csv(results_dir / f'{slug}_forecasts.csv', index=False)
    pd.DataFrame(summaries).to_csv(results_dir / f'{slug}_scores.csv', index=False)
    np.savez_compressed(results_dir / f'{slug}_samples.npz', **arrays)
    return summaries


def draw_forecasts(frame, model_names, *, results_dir, figures_dir):
    results_dir, figures_dir = Path(results_dir), Path(figures_dir)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'pdf.fonttype': 42, 'ps.fonttype': 42})
    fig, axes = plt.subplots(len(model_names), 1, figsize=(6.0, 5.5), sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    origin = ORIGINS[0]
    history = frame.iloc[origin-96:origin]
    target = frame.iloc[origin:origin+HORIZON]
    for ax, name in zip(axes, model_names):
        path = results_dir/f"{name.lower().replace('-', '_')}_forecasts.csv"
        data = pd.read_csv(path, parse_dates=['ds'])
        data = data.loc[data.origin_index.eq(origin)]
        ax.plot(history.ds, history.y, color='#5a6268', lw=.9, label='Observed history')
        ax.plot(target.ds, target.y, color='#111111', lw=1.0, label='Held-out actual')
        ax.fill_between(data.ds, data.lo95, data.hi95, color='#147d92', alpha=.13, label='95% central interval')
        ax.fill_between(data.ds, data.lo80, data.hi80, color='#147d92', alpha=.25, label='80% central interval')
        ax.plot(data.ds, data.point, color='#b34b32', lw=1.15, label='Predictive median')
        ax.axvline(target.ds.iloc[0], color='#666666', ls='--', lw=.7)
        ax.set_title(name, loc='left', fontsize=9, fontweight='bold')
        ax.set_ylabel('OT (deg C)')
        ax.grid(axis='y', alpha=.15)
    axes[0].legend(ncol=2, fontsize=6.3, loc='upper left', frameon=False)
    locator = mdates.AutoDateLocator(minticks=3, maxticks=5)
    axes[-1].xaxis.set_major_locator(locator)
    axes[-1].xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    axes[-1].set_xlabel('Timestamp; dashed line starts the 96-hour forecast')
    fig.autofmt_xdate(rotation=0, ha='center')
    fig.tight_layout()
    for extension in ('pdf', 'png'):
        fig.savefig(figures_dir/f'zoo_forecasts.{extension}', dpi=180, bbox_inches='tight')
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', default='all', choices=['all', 'arima', 'nhits', 'chronos'])
    parser.add_argument('--max-steps', type=int, default=200)
    parser.add_argument('--chronos-sample-batch', type=int, default=N_SAMPLES,
                        help='Optional reduced-memory generation; exact samples can differ')
    parser.add_argument('--output-dir', type=Path,
                        help='New or empty run folder (default: a unique folder under runs/)')
    args = parser.parse_args(argv)
    if not 1 <= args.chronos_sample_batch <= N_SAMPLES:
        parser.error('--chronos-sample-batch must be between 1 and 128')
    if args.max_steps < 1:
        parser.error('--max-steps must be positive')
    try:
        output = fresh_output_dir(args.output_dir, root=ROOT, prefix='zoo')
    except ValueError as exc:
        parser.error(str(exc))
    results_dir, figures_dir = output / 'results', output / 'figures'
    results_dir.mkdir()
    figures_dir.mkdir()
    frame = load_data()
    available = [('AutoARIMA', lambda: arima_forecasts(frame)),
                 ('NHITS-Normal', lambda: nhits_forecasts(frame, args.max_steps)),
                 ('Chronos-T5-Small', lambda: chronos_forecasts(frame, args.chronos_sample_batch))]
    aliases = {'arima': 'AutoARIMA', 'nhits': 'NHITS-Normal', 'chronos': 'Chronos-T5-Small'}
    selected = [(name, run) for name, run in available
                if args.models == 'all' or name == aliases[args.models]]
    manifest_path = results_dir / 'manifest.json'
    manifest = {'status': 'running', 'started_utc': utc_now(),
                'requested_models': [name for name, _ in selected],
                'successful_models': [], 'selected_models_complete': False,
                'comparison_complete': False, 'models': {},
                'seed': SEED, 'n_samples': N_SAMPLES,
                'horizon': HORIZON, 'origins': list(ORIGINS),
                'origin_spacing': '96 hours, no target overlap',
                'python': platform.python_version(), 'device': 'CPU',
                'data_revision': DATA_REVISION, 'data_sha256': DATA_SHA256,
                'target': 'OT', 'target_units': 'degrees Celsius',
                'split_end_exclusive': {'train': TRAIN_END, 'validation': VAL_END, 'test': TEST_END},
                'timestamps': {str(i): str(frame.ds.iloc[i]) for i in (0, TRAIN_END, VAL_END, TEST_END-1)},
                'metric_verification': verify_scores(),
                'point_summary': 'Normal median=loc; Chronos sample median',
                'crps': 'empirical ensemble, diagonal pairs included; marginal average',
                'intervals': 'sample quantiles; no post-hoc calibration',
                'benchmark_scope': 'three fixed test origins; instructional bounded run',
                'pretraining_overlap': 'unknown',
                'packages': package_versions(('numpy', 'pandas', 'scipy', 'matplotlib', 'torch',
                    'statsforecast', 'neuralforecast', 'chronos-forecasting', 'transformers'))}
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    errors, successful, score_rows = [], [], []
    for name, run in selected:
        start = time.perf_counter()
        print(f'Starting {name}', flush=True)
        try:
            forecasts, config = run()
            rows = save_model(name, forecasts, frame, results_dir)
            score_rows.extend(rows)
            successful.append(name)
            manifest['models'][name] = {'status': 'executed', 'seconds': time.perf_counter()-start,
                                        'configuration': config}
        except Exception as exc:
            traceback.print_exc()
            manifest['models'][name] = {'status': 'failed', 'seconds': time.perf_counter()-start,
                                        'error': f'{type(exc).__name__}: {exc}'}
            errors.append(name)
        manifest['successful_models'] = list(successful)
        manifest_path.write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
        print(f'Finished {name}: {manifest["models"][name]["status"]}', flush=True)
    # Use only successful outputs from this invocation, never a directory glob.
    try:
        if score_rows:
            per_origin = pd.DataFrame(score_rows)
            per_origin.to_csv(results_dir / 'scores_by_origin.csv', index=False)
            metrics = per_origin.drop(columns='origin_index').groupby('model', sort=False).mean().reset_index()
            metrics.to_csv(results_dir / 'metrics.csv', index=False)
            (results_dir / 'metrics.json').write_text(metrics.to_json(orient='records', indent=2), encoding='utf-8')
            draw_forecasts(frame, successful, results_dir=results_dir, figures_dir=figures_dir)
            print(metrics.to_string(index=False), flush=True)
        manifest['postprocessing'] = {'status': 'completed' if score_rows else 'no successful forecasts'}
    except Exception as exc:
        traceback.print_exc()
        manifest['postprocessing'] = {'status': 'failed', 'error': f'{type(exc).__name__}: {exc}'}
        errors.append('postprocessing')
    manifest['selected_models_complete'] = len(successful) == len(selected)
    manifest['comparison_complete'] = len(successful) == len(available)
    manifest['status'] = 'executed' if not errors else ('partial' if successful else 'failed')
    manifest['finished_utc'] = utc_now()
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    print(f'Run artifacts: {output}', flush=True)
    if errors:
        raise SystemExit(f'Failures recorded in {manifest_path}: {errors}')
    return output


if __name__ == '__main__':
    main()

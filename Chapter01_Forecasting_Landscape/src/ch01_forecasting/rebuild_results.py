"""Verify saved laboratory artifacts; optionally regenerate into a fresh folder.

Without --output-dir this command is read-only, offline, and runs no models.
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd

from .artifacts import fresh_output_dir, package_versions, utc_now
from .metrics import score_forecast, verify_scores
from .run_zoo import ROOT, ORIGINS, HORIZON, N_SAMPLES, load_data, draw_forecasts

MODEL_NAMES = ['AutoARIMA', 'NHITS-Normal', 'Chronos-T5-Small']


def _compare_table(saved, expected, keys, *, tolerance=1e-12):
    if set(saved.columns) != set(expected.columns):
        raise ValueError('Saved table has unexpected columns')
    if saved.duplicated(keys).any():
        raise ValueError('Saved table has duplicate model/origin keys')
    saved = saved[expected.columns].sort_values(keys).reset_index(drop=True)
    expected = expected.sort_values(keys).reset_index(drop=True)
    pd.testing.assert_frame_equal(saved, expected, check_dtype=False,
                                  rtol=tolerance, atol=tolerance)


def verify_saved_results():
    """Recompute saved scores, returning data and a new verification report.

    All inputs are frozen chapter artifacts. Their provenance is inspected and
    hashed, never replaced with the verification environment.
    """
    frame = load_data(download=False)
    results = ROOT / 'results'
    manifest_path = results / 'manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if (manifest['horizon'] != HORIZON or manifest['n_samples'] != N_SAMPLES
            or manifest['origins'] != list(ORIGINS)):
        raise ValueError('Saved manifest does not match this fixed chapter protocol')
    verified, all_rows, source_files = [], [], [manifest_path]
    for name in MODEL_NAMES:
        slug = name.lower().replace('-', '_')
        samples_path = results / f'{slug}_samples.npz'
        scores_path = results / f'{slug}_scores.csv'
        forecasts_path = results / f'{slug}_forecasts.csv'
        source_files.extend([samples_path, scores_path, forecasts_path])
        scores = pd.read_csv(scores_path)
        forecasts = pd.read_csv(forecasts_path, parse_dates=['ds'])
        if (len(scores) != len(ORIGINS) or scores.origin_index.duplicated().any()
                or set(scores.origin_index) != set(ORIGINS) or not scores.model.eq(name).all()):
            raise ValueError(f'Unexpected saved score rows for {name}')
        if len(forecasts) != len(ORIGINS) * HORIZON or not forecasts.model.eq(name).all():
            raise ValueError(f'Unexpected saved forecast rows for {name}')
        with np.load(samples_path, allow_pickle=False) as saved:
            for origin in ORIGINS:
                actual = saved[f'actual_{origin}']
                samples = saved[f'samples_{origin}']
                point = saved[f'point_{origin}']
                if samples.shape != (N_SAMPLES, HORIZON) or point.shape != actual.shape or point.shape != (HORIZON,):
                    raise ValueError(f'Unexpected saved sample shape for {name}, {origin}')
                if not (np.isfinite(samples).all() and np.isfinite(point).all()):
                    raise ValueError(f'Nonfinite saved forecasts for {name}, {origin}')
                np.testing.assert_array_equal(actual, frame.y.iloc[origin:origin+HORIZON])
                if name == 'Chronos-T5-Small':
                    np.testing.assert_allclose(point, np.median(samples, axis=0))
                recomputed = score_forecast(actual, samples, point)
                old = scores.loc[scores.origin_index.eq(origin)].iloc[0]
                for metric, value in recomputed.items():
                    np.testing.assert_allclose(value, old[metric], rtol=1e-12, atol=1e-12)
                shown = forecasts.loc[forecasts.origin_index.eq(origin)]
                if len(shown) != HORIZON:
                    raise ValueError(f'Unexpected forecast horizon for {name}, {origin}')
                np.testing.assert_array_equal(shown.horizon, np.arange(1, HORIZON+1))
                np.testing.assert_array_equal(shown.ds.to_numpy(), frame.ds.iloc[origin:origin+HORIZON].to_numpy())
                np.testing.assert_allclose(shown['point'], point)
                np.testing.assert_allclose(shown['actual'], actual)
                quantiles = np.quantile(samples, [.025, .1, .9, .975], axis=0)
                for key, values in zip(('lo95', 'lo80', 'hi80', 'hi95'), quantiles):
                    np.testing.assert_allclose(shown[key], values)
                all_rows.append({'model': name, 'origin_index': origin, **recomputed})
                verified.append({'model': name, 'origin_index': origin,
                                 'sample_shape': list(samples.shape),
                                 'saved_targets_scores_and_intervals': 'passed'})
    by_origin = pd.DataFrame(all_rows)
    metrics = by_origin.drop(columns='origin_index').groupby('model', sort=False).mean().reset_index()
    for filename, expected, keys in (
            ('scores_by_origin.csv', by_origin, ['model', 'origin_index']),
            ('metrics.csv', metrics, ['model'])):
        path = results / filename
        _compare_table(pd.read_csv(path), expected, keys)
        source_files.append(path)
    # pandas' original JSON export rounds to ten decimal places.
    json_path = results / 'metrics.json'
    _compare_table(pd.DataFrame(json.loads(json_path.read_text())), metrics, ['model'], tolerance=1e-9)
    source_files.append(json_path)
    report = {'status': 'passed', 'verified_utc': utc_now(),
              'operation': 'Recomputed scores from saved samples; models not rerun',
              'python': platform.python_version(),
              'packages': package_versions(('numpy', 'pandas', 'matplotlib')),
              'metric_checks': verify_scores(), 'model_origin_checks': verified,
              'aggregate_tables': 'scores_by_origin.csv, metrics.csv and metrics.json verified',
              'source_sha256': {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                                for path in source_files}}
    return frame, metrics, by_origin, report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path,
                        help='Optional new/empty folder for regenerated artifacts; otherwise verify read-only')
    args = parser.parse_args(argv)
    frame, metrics, by_origin, report = verify_saved_results()
    if args.output_dir is not None:
        try:
            output = fresh_output_dir(args.output_dir, root=ROOT, prefix='verified')
        except ValueError as exc:
            parser.error(str(exc))
        results, figures = output / 'results', output / 'figures'
        results.mkdir()
        figures.mkdir()
        by_origin.to_csv(results / 'scores_by_origin.csv', index=False)
        metrics.to_csv(results / 'metrics.csv', index=False)
        (results / 'metrics.json').write_text(metrics.to_json(orient='records', indent=2), encoding='utf-8')
        draw_forecasts(frame, MODEL_NAMES, results_dir=ROOT / 'results', figures_dir=figures)
        write_table(metrics, results / 'zoo_table.tex')
        report['regenerated_outputs'] = ['results/scores_by_origin.csv', 'results/metrics.csv',
                                         'results/metrics.json', 'results/zoo_table.tex',
                                         'figures/zoo_forecasts.pdf', 'figures/zoo_forecasts.png']
        (results / 'verification.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
        print(f'Regenerated artifacts: {output}')
    print(metrics.to_string(index=False))
    print('Verified all 9 model-origin sample arrays and aggregate tables; reference files unchanged.')
    return report


def write_table(metrics, path):
    lines = [r'\begin{table}[htbp]', r'\centering\small',
             r'\caption{Measured ETTh1 results over three fixed 96-hour windows (288 targets). '
             r'MAE, CRPS and interval width are in $^\circ$C; sMAPE and coverage are percentages. '
             r'Intervals use 128-sample quantiles, without calibration.}',
             r'\label{tab:ch01-zoo-results}',
             r'\begin{tabular}{lrrrrr}', r'\toprule',
             r'Model & MAE & sMAPE & CRPS & \shortstack{80\%\\coverage} & \shortstack{80\%\\width} \\',
             r'\midrule']
    for row in metrics.itertuples(index=False):
        label = {'AutoARIMA': 'AutoARIMA', 'NHITS-Normal': 'N-HiTS--Normal',
                 'Chronos-T5-Small': 'Chronos-T5-Small'}[row.model]
        lines.append(f'{label} & {row.MAE:.3f} & {row.sMAPE_percent:.2f} & {row.CRPS:.3f} & {100*row.coverage_80:.2f} & {row.width_80:.2f} '+r'\\')
    lines.extend([r'\bottomrule', r'\end{tabular}', r'\end{table}'])
    Path(path).write_text('\n'.join(lines)+'\n', encoding='utf-8')


if __name__ == '__main__':
    main()

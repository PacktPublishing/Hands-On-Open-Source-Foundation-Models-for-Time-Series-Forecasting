"""Small, explicit forecasting scores used in the Chapter 1 laboratory.

Arrays are (samples, horizon). The CRPS is for the empirical ensemble
(including diagonal pairs), not the unbiased fair-ensemble estimator.
"""
import numpy as np


def ensemble_crps(y, samples):
    """Return marginal empirical-ensemble CRPS for every target.

    E|X-y| - 0.5 E|X-X'|, computed without an O(M**2) pair matrix.
    """
    y = np.asarray(y, dtype=float)
    x = np.asarray(samples, dtype=float)
    if x.ndim != 2 or x.shape[1:] != y.shape:
        raise ValueError('Expected y:(H,), samples:(M,H)')
    if x.shape[0] == 0 or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('Scores require nonempty finite forecasts and targets')
    ordered = np.sort(x, axis=0)
    m = len(ordered)
    weights = 2 * np.arange(1, m + 1) - m - 1
    pair_term = (weights[:, None] * ordered).sum(axis=0) / (m * m)
    return np.abs(x - y).mean(axis=0) - pair_term


def smape(y, point):
    """sMAPE in percent; both zero contributes zero, range [0,200]."""
    y, point = np.asarray(y), np.asarray(point)
    denominator = np.abs(y) + np.abs(point)
    ratio = np.divide(np.abs(y - point), denominator,
                      out=np.zeros_like(denominator, dtype=float),
                      where=denominator > 0)
    return 200 * ratio.mean()


def score_forecast(y, samples, point=None):
    """Common point summaries and marginal distribution diagnostics."""
    x = np.asarray(samples)
    p = np.median(x, axis=0) if point is None else np.asarray(point)
    out = {'MAE': float(np.abs(y - p).mean()),
           'sMAPE_percent': float(smape(y, p)),
           'CRPS': float(ensemble_crps(y, x).mean())}
    for level in (80, 95):
        q = (1 - level / 100) / 2
        lo, hi = np.quantile(x, [q, 1-q], axis=0)
        out[f'coverage_{level}'] = float(((y >= lo) & (y <= hi)).mean())
        out[f'width_{level}'] = float((hi - lo).mean())
    return out


def verify_scores():
    """Check the optimized CRPS independently, plus exact edge cases."""
    y = np.array([1., -2., 0.])
    x = np.array([[0., -1., 0.], [2., 3., 0.], [1., -2., 0.]])
    brute = np.abs(x-y).mean(0) - 0.5 * np.abs(x[:, None]-x[None, :]).mean((0, 1))
    np.testing.assert_allclose(ensemble_crps(y, x), brute, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(ensemble_crps(y, y[None]), np.zeros(3))
    assert smape(np.array([0.]), np.array([0.])) == 0
    assert smape(np.array([1.]), np.array([-1.])) == 200
    np.testing.assert_allclose(smape(np.array([2.]), np.array([1.])), 200/3)
    return {'optimized_crps_vs_pairwise': 'passed', 'perfect_forecast': 'passed',
            'smape_zero_and_opposite_sign': 'passed'}


if __name__ == '__main__':
    print(verify_scores())

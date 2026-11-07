"""Download and parse the primary sources (Notebook 00 only)."""
# --- cell: download ---
import hashlib
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

HEADERS = {"User-Agent": "Packt-TSFM-book-chapter3/1.0 (research; contact via GitHub)"}


def download(url, dest, log, retries=4, timeout=60):
    """Download url to dest (bytes), retrying with back-off; record a log entry.

    The log entry keeps what a reader needs to audit the snapshot: URL, time
    (UTC), HTTP status, size and SHA-256 of the raw file.
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    last_error = None
    for attempt in range(retries):
        try:
            r = requests.get(url, headers=HEADERS, timeout=timeout)
            r.raise_for_status()
            dest.write_bytes(r.content)
            log.append({
                "url": url, "file": dest.name,
                "downloaded_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "http_status": r.status_code, "bytes": len(r.content),
                "sha256": hashlib.sha256(r.content).hexdigest(),
            })
            return dest
        except Exception as err:  # network errors and HTTP errors
            last_error = err
            time.sleep(2 ** (attempt + 1))
    raise RuntimeError(f"Could not download {url}: {last_error}")


def find_link(page_url, pattern):
    """First link on a web page whose address matches a regular expression, made absolute.

    Publishers rename files between releases (HadCET v2.0 to v2.1, a new Pink
    Sheet every month), so the link is read from the landing page each time.
    """
    html = requests.get(page_url, headers=HEADERS, timeout=60).text
    for href in re.findall(r'href=["\']([^"\']+)["\']', html):
        if re.search(pattern, href):
            return requests.compat.urljoin(page_url, href)
    raise ValueError(f"No link matching {pattern!r} on {page_url}")


# --- cell: parsers ---
def read_fred_csv(path, series_id):
    """FRED fredgraph.csv: first column is the date, second the value ('.' or empty = missing)."""
    df = pd.read_csv(path)
    s = pd.to_numeric(df[df.columns[1]], errors="coerce")
    s.index = pd.to_datetime(df[df.columns[0]])
    s.name = series_id
    return s


def read_hadcet_daily(path):
    """HadCET daily file -> Series of daily mean temperature (deg C).

    Handles the two layouts the Met Office has used: one row per day
    ('YYYY-MM-DD value' or 'YYYY MM DD value'), and the legacy matrix layout
    (year, day of month, then 12 monthly columns in tenths of a degree).
    Missing values (-99.9 or below) become NaN.
    """
    text = Path(path).read_text(errors="replace")
    rows = []
    for line in text.splitlines():
        tok = line.replace(",", " ").split()
        if not tok:
            continue
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", tok[0]) and len(tok) >= 2:
            rows.append((tok[0], tok[1]))
        elif (len(tok) in (4, 5) and len(tok[0]) == 4
              and all(re.fullmatch(r"\d+", t) for t in tok[:3])):
            rows.append((f"{tok[0]}-{int(tok[1]):02d}-{int(tok[2]):02d}", tok[3]))
    if rows:
        s = pd.Series([float(v) for _, v in rows], index=pd.to_datetime([d for d, _ in rows]))
    else:  # legacy matrix: year, day, Jan..Dec (tenths of a degree)
        recs = []
        for line in text.splitlines():
            tok = line.split()
            if len(tok) == 14 and tok[0].isdigit() and len(tok[0]) == 4:
                year, day = int(tok[0]), int(tok[1])
                for m, v in enumerate(tok[2:], start=1):
                    try:
                        recs.append((pd.Timestamp(year, m, day), float(v) / 10))
                    except ValueError:  # e.g. 30 February
                        pass
        s = pd.Series(dict(recs)).sort_index()
    s = s.where(s > -99).sort_index()
    s.name = "temp"
    return s[~s.index.duplicated()]


def read_hadcet_monthly(path):
    """HadCET monthly file -> Series of monthly means (deg C), indexed by month start.

    Rows are a year followed by twelve monthly values (and an annual value).
    """
    recs = {}
    for line in Path(path).read_text(errors="replace").splitlines():
        tok = line.replace(",", " ").split()
        if len(tok) >= 13 and re.fullmatch(r"\d{4}", tok[0]):
            for m, v in enumerate(tok[1:13], start=1):
                try:
                    recs[pd.Timestamp(int(tok[0]), m, 1)] = float(v)
                except ValueError:
                    pass
    s = pd.Series(recs).sort_index()
    return s.where(s > -99)


def read_cpc_daily(path, name):
    """NOAA CPC daily teleconnection index: year, month, day, value (CSV or whitespace text)."""
    df = pd.read_csv(path)
    if df.shape[1] < 4:  # the older plain-text layout
        df = pd.read_csv(path, sep=r"\s+", header=None)
    df = df.apply(pd.to_numeric, errors="coerce").dropna(subset=df.columns[:3])
    idx = pd.to_datetime(dict(year=df.iloc[:, 0], month=df.iloc[:, 1], day=df.iloc[:, 2]))
    s = pd.Series(df.iloc[:, 3].to_numpy(), index=idx, name=name)
    return s.where(s > -99).sort_index()


def read_pink_sheet(path, columns):
    """World Bank Pink Sheet monthly prices: sheet 'Monthly Prices', dates like 1960M01."""
    raw = pd.read_excel(path, sheet_name="Monthly Prices", header=None)
    header_row = next(i for i in range(15) if raw.iloc[i].astype(str).str.contains("Copper").any())
    header = raw.iloc[header_row].astype(str).str.strip()
    data = raw.iloc[header_row + 1:].copy()
    data.columns = header
    first = data.columns[0]
    data = data[data[first].astype(str).str.fullmatch(r"\d{4}M\d{2}")]
    idx = pd.to_datetime(data[first].str.replace("M", "-") + "-01")
    out = {}
    for col, wb_name in columns.items():
        match = [c for c in data.columns if c.lower().startswith(wb_name.lower())]
        if match:
            out[col] = pd.to_numeric(data[match[0]], errors="coerce").to_numpy()
    return pd.DataFrame(out, index=idx)


# --- cell: snapshot-helpers ---
def last_complete_month_end(today=None):
    """Last day of the previous calendar month (the vintage cut)."""
    today = pd.Timestamp(today or datetime.now(timezone.utc).date())
    return (today.to_period("M") - 1).to_timestamp(how="end").normalize()


def flag_outliers(s, window, z_max=6.0):
    """Flag unusual *changes*: robust z = (d_t - median) / (1.4826 MAD) over a centred window.

    d_t is the first difference (log difference for prices passed as logs).
    Flags are for explaining forecast errors later; values are never removed.
    """
    d = s.diff()
    med = d.rolling(window, center=True, min_periods=window // 2).median()
    mad = (d - med).abs().rolling(window, center=True, min_periods=window // 2).median()
    z = (d - med) / (1.4826 * mad.replace(0, np.nan))
    return z[z.abs() > z_max]


def compare_overlap(a, b):
    """Correlation and gaps between two versions of a series over their common dates."""
    common = a.dropna().index.intersection(b.dropna().index)
    if len(common) < 3:
        return {"n_common": len(common), "corr": np.nan, "mean_gap": np.nan, "mean_abs_rel_gap_pct": np.nan}
    x, y = a.loc[common], b.loc[common]
    return {"n_common": len(common), "corr": float(np.corrcoef(x, y)[0, 1]),
            "mean_gap": float((x - y).mean()),
            "mean_abs_rel_gap_pct": float((100 * (x - y).abs() / y.abs()).mean())}


def write_manifest(path, manifest):
    """snapshot_manifest.json: vintage, downloads, datasets, covariate classes and choices made."""
    Path(path).write_text(json.dumps(manifest, indent=2, default=str))

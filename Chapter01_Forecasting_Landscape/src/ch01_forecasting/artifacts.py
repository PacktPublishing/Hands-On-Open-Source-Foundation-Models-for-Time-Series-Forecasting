"""Create isolated output directories without overwriting chapter references."""
from datetime import datetime, timezone
import importlib.metadata
from pathlib import Path
import uuid


def utc_now():
    """Return a timezone-aware timestamp for a new operation."""
    return datetime.now(timezone.utc).isoformat()


def package_versions(names):
    """Record optional distributions as null when they are not installed."""
    versions = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def fresh_output_dir(output_dir=None, *, root=None, prefix='run'):
    """Create an empty run folder, rejecting overlap with bundled artifacts.

    Explicit relative destinations are resolved from the current working
    directory. Existing empty directories are accepted; nonempty ones are never
    reused. Resolve symlinks before checking protected directory boundaries.
    """
    root = Path(root if root is not None else Path(__file__).resolve().parents[2]).resolve()
    if output_dir is None:
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        output_dir = root / 'runs' / f'{prefix}_{stamp}_{uuid.uuid4().hex[:8]}'
    path = Path(output_dir).expanduser().resolve()
    for name in ('data', 'results', 'figures'):
        protected = (root / name).resolve()
        if path == protected or path in protected.parents or protected in path.parents:
            raise ValueError(f'Output overlaps bundled {name}: {path}')
    if path.exists():
        if not path.is_dir() or any(path.iterdir()):
            raise ValueError(f'Output directory must be new or empty: {path}')
    else:
        path.mkdir(parents=True)
    return path

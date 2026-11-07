"""Shared wording for the supervised baselines (notebook-only figures)."""
import sys
from pathlib import Path

from ..explainer import GREY, SIZE_H, FanV, SeriesRowsStacked, Text

COLORS = ["#2f2f2f", "#2e8a5f", "#9a7b33"]


def inputs(window, roles=("Target", "Known ahead", "Past only"), note=None):
    rows = {"Target": ("Target", "temperature", False), "Known ahead": ("Known ahead", "day length", True),
            "Past only": ("Past only", "NAO index", False)}
    cols = {"Target": COLORS[0], "Known ahead": COLORS[1], "Past only": COLORS[2]}
    parts = [Text(f"Each forecast reads a window of the last {window}. A ? is a value not known yet.",
                  space_after=0.08),
             SeriesRowsStacked([rows[r] for r in roles], colors=[cols[r] for r in roles], space_after=0.1),
             Text("Prices go in as log differences, oil and yields as differences; all inputs are min-max scaled "
                  "on the training rows.", size=SIZE_H["small"], color=GREY, space_after=0.06 if note else 0.0)]
    if note:
        parts.append(Text(note, size=SIZE_H["small"], color=GREY, space_after=0.0))
    return parts


def trained(extra):
    return [Text("Trained once per dataset on the rows before the first test window (daily: the last 4,096), then "
                 "every window is forecast from its own origin with no retraining.", space_after=0.08),
            FanV(h=0.55, bands=False, labels=("point forecast",)),
            Text(extra, size=SIZE_H["small"], color=GREY, space_after=0.0)]


NOTE_TUNE = "Optuna tries a few settings on validation windows inside the training rows; the first is the default."

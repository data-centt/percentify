"""Diagnostic profiling for Percentify.

Unlike descriptive profilers that dump statistics for every column, ``profile``
is a *diagnostician*: it runs a registry of checks, surfaces the problems that
actually matter, ranks them worst-first, and tells you how to fix each one.

    from percentify import profile

    report = profile(df)                 # pandas or polars in, report out
    report                               # pretty summary (repr / notebook HTML)
    report.errors                        # just the blocking issues
    assert not report.errors             # drop straight into CI

Design notes
------------
* Backend is detected from the input type, never a flag: pandas, polars, or
  anything convertible all work. Analysis currently runs on a pandas view; the
  returned report is backend-agnostic. Native-polars checks can replace the
  conversion later without changing the public API.
* Every check is a plain function ``fn(df, target) -> list[Finding]`` registered
  in ``CHECKS``. Adding a diagnostic is a one-function change — a natural
  good-first-issue.
* No heavy dependencies: numpy + pandas only, matching the rest of the library.
  Notebook rendering uses ``_repr_html_``; the terminal uses plain unicode.
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import pandas as pd

__all__ = ["profile", "ProfileReport", "Finding"]

# Severity ordering and the penalty each level deducts from the health score.
_SEVERITY_ORDER = {"error": 0, "warning": 1, "info": 2}
_SEVERITY_PENALTY = {"error": 12, "warning": 4, "info": 1}
_SEVERITY_MARK = {"error": "\u2717", "warning": "\u26a0", "info": "\u2139"}  # ✗ ⚠ ℹ

_SPARK = "\u2581\u2582\u2583\u2584\u2585\u2586\u2587\u2588"  # ▁▂▃▄▅▆▇█

# "All-nines" style magic numbers that are almost always disguised missing data.
# Deliberately conservative so real values (0, -1, small negatives) don't trip it.
_SENTINELS = frozenset({-999, 999, -9999, 9999, -99999, 99999, -999999})


@dataclass(frozen=True)
class Finding:
    """A single diagnostic result."""

    column: str
    severity: str  # "error" | "warning" | "info"
    code: str  # short machine-readable id, e.g. "id_like"
    message: str  # human explanation of what's wrong
    suggestion: str = ""  # the recommended fix

    def __str__(self) -> str:
        mark = _SEVERITY_MARK.get(self.severity, "-")
        tail = f"  \u2192 {self.suggestion}" if self.suggestion else ""
        return f"{mark} {self.column:<22} {self.message}{tail}"


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def _to_pandas(data) -> pd.DataFrame:
    """Return a pandas DataFrame regardless of the input backend.

    Detects polars without importing it unless needed, so polars is never a
    hard dependency.
    """
    if isinstance(data, pd.DataFrame):
        return data
    module = type(data).__module__.split(".")[0]
    if module == "polars":
        return data.to_pandas()
    if isinstance(data, pd.Series):
        return data.to_frame()
    # Last resort: let pandas try (dict, records, numpy array, ...).
    return pd.DataFrame(data)


def _sparkline(series: pd.Series, bins: int = 8) -> str:
    """A tiny unicode histogram for a numeric column."""
    values = pd.to_numeric(series, errors="coerce").to_numpy()
    values = values[np.isfinite(values)]
    if values.size == 0:
        return " " * bins
    lo, hi = float(values.min()), float(values.max())
    if hi == lo:
        return _SPARK[0] * bins
    counts, _ = np.histogram(values, bins=bins, range=(lo, hi))
    if counts.max() == 0:
        return " " * bins
    idx = np.floor(counts / counts.max() * (len(_SPARK) - 1)).astype(int)
    return "".join(_SPARK[i] for i in idx)


def _numeric_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])
            and not pd.api.types.is_bool_dtype(df[c])]


def _is_textual_dtype(dt) -> bool:
    """True for object/categorical/string columns (incl. pandas 3.0 ``str``)."""
    if (pd.api.types.is_numeric_dtype(dt) or pd.api.types.is_bool_dtype(dt)
            or pd.api.types.is_datetime64_any_dtype(dt)
            or pd.api.types.is_timedelta64_dtype(dt)):
        return False
    return (isinstance(dt, pd.CategoricalDtype) or dt == object
            or pd.api.types.is_string_dtype(dt))


def _text_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if _is_textual_dtype(df[c].dtype)]


def _encode_target(target: pd.Series) -> tuple[np.ndarray, bool]:
    """Return (numeric codes, is_categorical)."""
    if pd.api.types.is_numeric_dtype(target) and not pd.api.types.is_bool_dtype(target):
        return target.to_numpy(dtype=float), False
    codes, _ = pd.factorize(target)
    return codes.astype(float), True


# --------------------------------------------------------------------------- #
# Checks — each takes (df, target_series_or_None) and returns a list[Finding]
# --------------------------------------------------------------------------- #
def check_missing(df: pd.DataFrame, target) -> list[Finding]:
    out: list[Finding] = []
    n = len(df)
    if n == 0:
        return out
    frac = df.isna().mean()
    for col, f in frac.items():
        if f >= 1.0:
            out.append(Finding(col, "error", "all_missing",
                               "column is entirely missing",
                               "drop the column"))
        elif f >= 0.4:
            out.append(Finding(col, "warning", "high_missing",
                               f"{f * 100:.0f}% missing",
                               "impute or drop"))
    return out


def check_constant(df: pd.DataFrame, target) -> list[Finding]:
    out: list[Finding] = []
    for col in df.columns:
        if df[col].isna().all():
            continue  # already reported by check_missing
        if df[col].nunique(dropna=True) <= 1:
            out.append(Finding(col, "warning", "constant",
                               "only one distinct value",
                               "drop — carries no information"))
    return out


def check_id_like(df: pd.DataFrame, target) -> list[Finding]:
    out: list[Finding] = []
    n = len(df)
    if n < 20:
        return out
    for col in df.columns:
        nun = df[col].nunique(dropna=True)
        if nun / n >= 0.95 and (pd.api.types.is_integer_dtype(df[col])
                                or _is_textual_dtype(df[col].dtype)):
            out.append(Finding(col, "warning", "id_like",
                               f"identifier-like ({nun}/{n} unique)",
                               "drop before modeling"))
    return out


def check_duplicate_rows(df: pd.DataFrame, target) -> list[Finding]:
    n = len(df)
    if n == 0:
        return []
    dups = int(df.duplicated().sum())
    if dups > 0:
        return [Finding("<rows>", "warning", "duplicate_rows",
                        f"{dups} duplicate rows ({dups / n * 100:.0f}%)",
                        "deduplicate")]
    return []


_SYMBOLS = re.compile(r"[,$%\s\u20ac\u00a3]")  # thousands sep, $, %, whitespace, € £
_DATEISH = re.compile(r"[-/:]")


def check_dtype_mismatch(df: pd.DataFrame, target) -> list[Finding]:
    # Regex runs in plain Python (not the pandas string accessor) so behaviour
    # is identical whether the column is object, pandas ``str`` or arrow-backed.
    out: list[Finding] = []
    for col in _text_columns(df):
        vals = [str(v) for v in df[col].dropna().head(200).tolist()]
        if len(vals) < 5:
            continue
        # numbers stored as text?
        stripped = [_SYMBOLS.sub("", v) for v in vals]
        numeric = pd.to_numeric(pd.Series(stripped), errors="coerce")
        if numeric.notna().mean() >= 0.9:
            out.append(Finding(col, "warning", "numeric_as_text",
                               "numbers stored as strings",
                               "cast to a numeric dtype"))
            continue
        # dates stored as text? Require a date-ish separator so bare ints
        # like "12345" aren't misread as timestamps.
        if np.mean([bool(_DATEISH.search(v)) for v in vals]) >= 0.8:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                parsed = pd.to_datetime(pd.Series(vals), errors="coerce",
                                        format="mixed")
            if parsed.notna().mean() >= 0.9:
                out.append(Finding(col, "warning", "date_as_text",
                                   "dates stored as strings",
                                   "cast to a datetime dtype"))
    return out


def check_categorical_consistency(df: pd.DataFrame, target) -> list[Finding]:
    """Detect values that are the same category under different spellings."""
    out: list[Finding] = []
    for col in _text_columns(df):
        raw = df[col].dropna().astype(str)
        if raw.empty:
            continue
        norm = raw.str.strip().str.lower().str.replace(r"\s+", " ", regex=True)
        groups = raw.groupby(norm).agg(lambda s: sorted(set(s)))
        offenders = groups[groups.map(len) > 1]
        if not offenders.empty:
            variants = offenders.iloc[0]
            example = " / ".join(f'"{v}"' for v in variants[:3])
            extra = f" (+{len(offenders) - 1} more)" if len(offenders) > 1 else ""
            out.append(Finding(col, "warning", "inconsistent_categories",
                               f"same category, different spellings: {example}{extra}",
                               "normalize casing/whitespace"))
    return out


def check_high_cardinality(df: pd.DataFrame, target) -> list[Finding]:
    out: list[Finding] = []
    n = len(df)
    if n < 20:
        return out
    for col in _text_columns(df):
        nun = df[col].nunique(dropna=True)
        if 50 <= nun < 0.95 * n:  # many levels, but not identifier-like
            out.append(Finding(col, "info", "high_cardinality",
                               f"{nun} categories",
                               "will expand heavily under one-hot encoding"))
    return out


def check_sentinel_missing(df: pd.DataFrame, target) -> list[Finding]:
    out: list[Finding] = []
    for col in _numeric_columns(df):
        s = df[col].dropna()
        if len(s) < 20:
            continue
        for sentinel in _SENTINELS:
            share = (s == sentinel).mean()
            if share >= 0.01:
                out.append(Finding(col, "warning", "sentinel_missing",
                                   f"value {sentinel:g} in {share * 100:.0f}% of rows "
                                   "looks like a hidden missing code",
                                   "replace with NaN before analysis"))
                break
    return out


def check_collinearity(df: pd.DataFrame, target) -> list[Finding]:
    """Flag pairs of near-duplicate numeric columns (|r| >= 0.95)."""
    cols = _numeric_columns(df)
    if len(cols) < 2:
        return []
    corr = df[cols].corr(numeric_only=True).abs()
    out: list[Finding] = []
    seen: set[frozenset] = set()
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            r = corr.loc[a, b]
            if pd.notna(r) and r >= 0.95 and frozenset((a, b)) not in seen:
                seen.add(frozenset((a, b)))
                out.append(Finding(f"{a} \u27f7 {b}", "warning", "collinear",
                                   f"correlation {r:.2f} — redundant pair",
                                   "drop one, or check vif()"))
    return out


def check_leakage(df: pd.DataFrame, target) -> list[Finding]:
    """Flag features that predict the target almost perfectly."""
    if target is None:
        return []
    y, y_is_cat = _encode_target(target)
    out: list[Finding] = []
    n = len(df)

    for col in df.columns:
        s = df[col]
        # Numeric feature: near-perfect linear association with the target.
        if col in _numeric_columns(df):
            mask = s.notna().to_numpy() & np.isfinite(y)
            if mask.sum() < 10:
                continue
            x = s.to_numpy(dtype=float)[mask]
            yy = y[mask]
            if np.std(x) == 0 or np.std(yy) == 0:
                continue
            r = abs(np.corrcoef(x, yy)[0, 1])
            if np.isfinite(r) and r > 0.98:
                out.append(Finding(col, "error", "leakage",
                                   f"predicts the target (|r|={r:.2f})",
                                   "likely leakage — confirm it is known at predict time"))
        # Categorical feature: does it almost perfectly determine the target?
        elif y_is_cat and col in _text_columns(df):
            nun = s.nunique(dropna=True)
            if nun <= 1 or nun >= 0.5 * n:
                continue  # constant or id-like handled elsewhere
            tmp = pd.DataFrame({"f": s.astype("object"), "y": y})
            tmp = tmp.dropna()
            if len(tmp) < 10:
                continue
            purity = (tmp.groupby("f")["y"]
                      .agg(lambda g: g.value_counts(normalize=True).max())
                      * tmp.groupby("f").size() / len(tmp)).sum()
            if purity > 0.995:
                out.append(Finding(col, "error", "leakage",
                                   f"almost perfectly determines the target "
                                   f"(purity {purity:.2f})",
                                   "likely leakage — confirm it is known at predict time"))
    return out


def check_target_imbalance(df: pd.DataFrame, target) -> list[Finding]:
    if target is None:
        return []
    if pd.api.types.is_numeric_dtype(target) and target.nunique(dropna=True) > 20:
        return []  # treat as continuous
    counts = target.value_counts(normalize=True, dropna=True)
    if len(counts) >= 2 and counts.min() < 0.05:
        smallest = counts.idxmin()
        return [Finding("<target>", "info", "imbalance",
                        f"class '{smallest}' is only {counts.min() * 100:.1f}% of rows",
                        "consider resampling or class weights")]
    return []


# Order here is only the registry order; findings are re-sorted by severity.
CHECKS: list[Callable[[pd.DataFrame, Optional[pd.Series]], list[Finding]]] = [
    check_missing,
    check_constant,
    check_id_like,
    check_duplicate_rows,
    check_dtype_mismatch,
    check_categorical_consistency,
    check_high_cardinality,
    check_sentinel_missing,
    check_collinearity,
    check_leakage,
    check_target_imbalance,
]


# --------------------------------------------------------------------------- #
# Report object
# --------------------------------------------------------------------------- #
@dataclass
class ProfileReport:
    """The result of :func:`profile`. Also a plain object you can act on."""

    n_rows: int
    n_cols: int
    findings: list[Finding]
    summary: pd.DataFrame
    target: Optional[str] = None
    _sparklines: dict = field(default_factory=dict, repr=False)

    # -- filtered views ---------------------------------------------------- #
    @property
    def issues(self) -> list[Finding]:
        return self.findings

    @property
    def errors(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == "error"]

    @property
    def warnings(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == "warning"]

    @property
    def infos(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == "info"]

    @property
    def health(self) -> int:
        penalty = sum(_SEVERITY_PENALTY[f.severity] for f in self.findings)
        return int(max(0, 100 - penalty))

    def to_frame(self) -> pd.DataFrame:
        """Findings as a tidy DataFrame (empty if the data is clean)."""
        return pd.DataFrame(
            [(f.severity, f.code, f.column, f.message, f.suggestion)
             for f in self.findings],
            columns=["severity", "code", "column", "message", "suggestion"],
        )

    # -- rendering --------------------------------------------------------- #
    def _header(self) -> str:
        e, w, i = len(self.errors), len(self.warnings), len(self.infos)
        return (f"{self.n_rows:,} rows \u00d7 {self.n_cols} cols  \u00b7  "
                f"health {self.health}/100  \u00b7  "
                f"\u2717 {e} errors  \u26a0 {w} warnings  \u2139 {i} info")

    def __str__(self) -> str:
        lines = [self._header(), ""]
        for level, label in (("error", "ERRORS"),
                             ("warning", "WARNINGS"),
                             ("info", "NOTES")):
            group = [f for f in self.findings if f.severity == level]
            if group:
                lines.append(label)
                lines.extend("  " + str(f) for f in group)
                lines.append("")
        lines.append("COLUMNS")
        for _, row in self.summary.iterrows():
            spark = self._sparklines.get(row["column"], "")
            lines.append(
                f"  {row['column'][:20]:<20} {row['dtype'][:7]:<7} "
                f"{spark:<8} miss {row['missing_pct']:>4.0f}%  "
                f"card {row['cardinality']:>6,}"
            )
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.__str__()

    def _repr_html_(self) -> str:
        colors = {"error": "#d33", "warning": "#e69500", "info": "#3a7"}
        rows = ""
        for f in self.findings:
            c = colors[f.severity]
            sug = f" &rarr; {f.suggestion}" if f.suggestion else ""
            rows += (
                f'<tr><td style="color:{c};font-weight:600;padding:2px 10px">'
                f'{f.severity}</td>'
                f'<td style="font-family:monospace;padding:2px 10px">{f.column}</td>'
                f'<td style="padding:2px 10px">{f.message}'
                f'<span style="color:#888">{sug}</span></td></tr>'
            )
        if not rows:
            rows = ('<tr><td colspan="3" style="padding:6px 10px;color:#3a7">'
                    'no issues found \u2713</td></tr>')
        cols = ""
        for _, r in self.summary.iterrows():
            spark = self._sparklines.get(r["column"], "")
            cols += (
                f'<tr><td style="font-family:monospace;padding:2px 10px">{r["column"]}</td>'
                f'<td style="padding:2px 10px;color:#888">{r["dtype"]}</td>'
                f'<td style="font-family:monospace;padding:2px 10px">{spark}</td>'
                f'<td style="padding:2px 10px">{r["missing_pct"]:.0f}%</td>'
                f'<td style="padding:2px 10px">{r["cardinality"]:,}</td></tr>'
            )
        return f"""
<div style="font-family:-apple-system,sans-serif;font-size:13px">
  <div style="font-weight:600;margin-bottom:6px">{self._header()}</div>
  <table style="border-collapse:collapse;margin-bottom:10px">{rows}</table>
  <div style="color:#888;font-size:11px;margin-bottom:2px">COLUMNS</div>
  <table style="border-collapse:collapse">{cols}</table>
</div>"""


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #
def profile(data, target: Optional[str] = None) -> ProfileReport:
    """Diagnose a DataFrame: rank its problems worst-first, with fixes.

    Parameters
    ----------
    data
        A pandas or polars DataFrame (or anything convertible to one).
    target
        Optional column name. When given, enables leakage and class-imbalance
        checks against it.

    Returns
    -------
    ProfileReport
        Renders as a compact summary in notebooks/terminals and exposes
        ``.errors``, ``.warnings``, ``.issues`` and ``.to_frame()`` for
        programmatic use.
    """
    df = _to_pandas(data)

    target_series = None
    if target is not None:
        if target in df.columns:
            target_series = df[target]
            df = df.drop(columns=[target])
        else:
            warnings.warn(f"target {target!r} not found in columns; ignoring",
                          stacklevel=2)

    findings: list[Finding] = []
    for check in CHECKS:
        try:
            findings.extend(check(df, target_series))
        except Exception as exc:  # a broken check must never sink the report
            warnings.warn(f"{check.__name__} failed: {exc}", stacklevel=2)

    findings.sort(key=lambda f: (_SEVERITY_ORDER[f.severity], f.column))

    # Per-column summary + sparklines for the numeric columns.
    n = len(df)
    summary_rows, sparks = [], {}
    for col in df.columns:
        summary_rows.append({
            "column": col,
            "dtype": str(df[col].dtype),
            "missing_pct": float(df[col].isna().mean() * 100) if n else 0.0,
            "cardinality": int(df[col].nunique(dropna=True)),
        })
        if pd.api.types.is_numeric_dtype(df[col]) and not pd.api.types.is_bool_dtype(df[col]):
            sparks[col] = _sparkline(df[col])
    summary = pd.DataFrame(summary_rows)

    return ProfileReport(
        n_rows=n, n_cols=df.shape[1], findings=findings,
        summary=summary, target=target, _sparklines=sparks,
    )

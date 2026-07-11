import pytest
import numpy as np
import pandas as pd
from percentify import profiler, ProfileReport, Finding, PercentifyWarning


@pytest.fixture
def messy_df():
    np.random.seed(0)
    base = np.random.randn(200)
    return pd.DataFrame({
        "id": range(200),                       # id-like
        "empty": [None] * 200,                   # entirely missing
        "const": [7] * 200,                      # constant
        "x": base,
        "x_dup": base * 1.0,                     # collinear with x
        "churn": ["No"] * 195 + ["Yes"] * 5,     # 2.5% minority target
    })


def test_profiler_returns_report(messy_df):
    assert isinstance(profiler(messy_df), ProfileReport)


def test_profiler_all_missing_is_error(messy_df):
    codes = {(f.column, f.code) for f in profiler(messy_df).errors}
    assert ("empty", "all_missing") in codes


def test_profiler_flags_constant_and_id(messy_df):
    warn_codes = {f.code for f in profiler(messy_df).warnings}
    assert "constant" in warn_codes
    assert "id_like" in warn_codes
    assert "collinear" in warn_codes


def test_profiler_target_imbalance(messy_df):
    infos = {f.code for f in profiler(messy_df, target="churn").infos}
    assert "imbalance" in infos


def test_profiler_to_frame(messy_df):
    frame = profiler(messy_df).to_frame()
    assert isinstance(frame, pd.DataFrame)
    assert list(frame.columns) == ["severity", "code", "column", "message", "suggestion"]


def test_profiler_findings_are_findings(messy_df):
    assert all(isinstance(f, Finding) for f in profiler(messy_df).findings)


def test_profiler_clean_data_full_health():
    clean = pd.DataFrame({
        "a": [x % 10 for x in range(30)],
        "b": [x % 7 for x in range(30)],
    })
    report = profiler(clean)
    assert report.findings == []
    assert report.health == 100


def test_profiler_health_drops_with_issues(messy_df):
    assert profiler(messy_df).health < 100


def test_profiler_target_not_found_warns(messy_df):
    with pytest.warns(PercentifyWarning):
        profiler(messy_df, target="does_not_exist")


def test_profiler_html_escapes_column_names():
    evil = pd.DataFrame({"<script>": [None] * 200, "ok": range(200)})
    rendered = profiler(evil)._repr_html_()
    assert "&lt;script&gt;" in rendered
    assert "<script>" not in rendered


def test_profiler_composes_missing_high_warning():
    # 60% missing should surface as a high_missing warning (via missing()).
    df = pd.DataFrame({"col": [1.0, 2.0] + [None] * 3})
    codes = {f.code for f in profiler(df).warnings}
    assert "high_missing" in codes


def test_profiler_polars_input(messy_df):
    pl = pytest.importorskip("polars")
    report = profiler(pl.from_pandas(messy_df), target="churn")
    assert isinstance(report, ProfileReport)
    assert any(f.code == "all_missing" for f in report.errors)

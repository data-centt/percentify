import pytest
import numpy as np
import pandas as pd
from percentify import vif


@pytest.fixture
def independent_df():
    np.random.seed(42)
    return pd.DataFrame({
        "a": np.random.randn(100),
        "b": np.random.randn(100),
        "c": np.random.randn(100),
    })


@pytest.fixture
def collinear_df():
    np.random.seed(42)
    x = np.random.randn(100)
    return pd.DataFrame({
        "x": x,
        "y": x * 2 + np.random.randn(100) * 0.01,
        "z": np.random.randn(100),
    })


def test_vif_returns_all_columns(independent_df):
    result = vif(independent_df)
    assert set(result.keys()) == {"a", "b", "c"}


def test_vif_independent_features_low(independent_df):
    result = vif(independent_df)
    for val in result.values():
        assert val < 5.0


def test_vif_collinear_features_high(collinear_df):
    result = vif(collinear_df)
    assert result["x"] > 10
    assert result["y"] > 10


def test_vif_flag_filters(independent_df):
    result = vif(independent_df, flag=5.0)
    assert len(result) == 0


def test_vif_flag_returns_collinear(collinear_df):
    result = vif(collinear_df, flag=5.0)
    assert "x" in result
    assert "y" in result
    assert "z" not in result


def test_vif_ignores_non_numeric():
    df = pd.DataFrame({
        "a": np.random.randn(50),
        "b": np.random.randn(50),
        "name": ["foo"] * 50,
    })
    result = vif(df)
    assert "name" not in result
    assert set(result.keys()) == {"a", "b"}


def test_vif_too_few_columns():
    df = pd.DataFrame({"a": [1, 2, 3]})
    with pytest.raises(ValueError):
        vif(df)


def test_vif_custom_decimals(independent_df):
    result = vif(independent_df, decimals=4)
    for val in result.values():
        str_val = str(val)
        if "." in str_val:
            assert len(str_val.split(".")[1]) <= 4

import warnings
from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd


class PercentifyWarning(UserWarning):
    """Raised when percentify handles a situation gracefully instead of erroring."""


def _warn(message: str) -> None:
    warnings.warn(message, PercentifyWarning, stacklevel=3)


def _round(value: float, decimals: Optional[int]) -> float:
    if decimals is None:
        return value
    return round(value, decimals)


def vif(df: pd.DataFrame, decimals: Optional[int] = 2, flag: Optional[float] = None) -> pd.DataFrame:
    """
    Calculate the Variance Inflation Factor for each numeric column in a DataFrame.

    VIF = 1 / (1 - R²), where R² comes from regressing each feature
    against all other features. VIF > 5 suggests moderate multicollinearity,
    VIF > 10 suggests severe multicollinearity.

    Non-numeric columns are ignored. If fewer than 2 numeric columns are
    available, a warning is raised and an empty DataFrame is returned instead
    of an error — encode categoricals first.

    Args:
        df: DataFrame with numeric columns.
        decimals: Number of decimal places to round to. If None, raw floats.
        flag: If set, only return rows with VIF above this threshold.

    Returns:
        DataFrame with columns ["feature", "VIF"], sorted highest VIF first.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"vif expects a pandas DataFrame, got {type(df).__name__}.")

    empty = pd.DataFrame(columns=["feature", "VIF"])
    numeric = df.select_dtypes(include=[np.number])
    n_numeric = numeric.shape[1]

    if n_numeric < 2:
        if n_numeric == 0:
            _warn("Numeric columns required: no numeric columns found. VIF measures "
                  "multicollinearity between numeric features - encode any "
                  "categorical/text columns first.")
        else:
            _warn("Numeric columns required: VIF needs at least 2 numeric columns to "
                  f"measure multicollinearity, but found only {n_numeric}.")
        return empty

    numeric = numeric.dropna()
    if numeric.shape[0] < 2:
        _warn("Not enough data: VIF needs at least 2 complete (non-null) rows.")
        return empty

    cols = numeric.columns.tolist()
    X = numeric.values.astype(float)

    vifs = []
    for i in range(len(cols)):
        y = X[:, i]
        others = np.delete(X, i, axis=1)
        others_with_intercept = np.hstack([np.ones((others.shape[0], 1)), others])

        coeffs, _, _, _ = np.linalg.lstsq(others_with_intercept, y, rcond=None)

        y_pred = others_with_intercept @ coeffs
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)

        r_sq = 0.0 if ss_tot == 0 else 1.0 - (ss_res / ss_tot)
        vif_val = float("inf") if r_sq >= 1.0 else 1.0 / (1.0 - r_sq)

        if decimals is not None:
            vif_val = round(vif_val, decimals)
        vifs.append(vif_val)

    result = pd.DataFrame({"feature": cols, "VIF": vifs})
    result = result.sort_values("VIF", ascending=False).reset_index(drop=True)

    if flag is not None:
        result = result[result["VIF"] > flag].reset_index(drop=True)

    return result


def missing(df: pd.DataFrame, decimals: Optional[int] = 2) -> pd.DataFrame:
    """
    Calculate the percentage of missing values for each column.

    Works on every column (numeric and text), so nothing is dropped.

    Args:
        df: DataFrame to profile.
        decimals: Number of decimal places to round to.

    Returns:
        DataFrame with columns ["column", "missing_pct"], sorted highest first.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"missing expects a pandas DataFrame, got {type(df).__name__}.")

    total = len(df)
    rows = []
    for col in df.columns:
        pct = 0.0 if total == 0 else df[col].isnull().sum() / total * 100.0
        rows.append((col, _round(pct, decimals)))

    result = pd.DataFrame(rows, columns=["column", "missing_pct"])
    return result.sort_values("missing_pct", ascending=False).reset_index(drop=True)


def cv(data: Union[pd.Series, pd.DataFrame], decimals: Optional[int] = 2) -> Union[float, pd.DataFrame]:
    """
    Calculate the coefficient of variation (CV = std / mean * 100).

    Args:
        data: A Series (returns a single float) or DataFrame (returns a DataFrame
            of all numeric columns).
        decimals: Number of decimal places to round to.

    Returns:
        float for a Series, or a DataFrame with columns ["feature", "cv"]
        sorted highest first. Non-numeric input is handled with a warning.
    """
    if isinstance(data, pd.Series):
        if not pd.api.types.is_numeric_dtype(data):
            _warn(f"cv expects numeric data, but got a non-numeric Series (dtype: "
                  f"{data.dtype}). Returning NaN. Encode or select a numeric column.")
            return float("nan")
        mean = data.mean()
        if mean == 0:
            _warn("Coefficient of variation is undefined when the mean is zero. "
                  "Returning inf.")
            return float("inf")
        return _round(data.std() / abs(mean) * 100.0, decimals)

    if not isinstance(data, pd.DataFrame):
        raise TypeError(f"cv expects a Series or DataFrame, got {type(data).__name__}.")

    numeric = data.select_dtypes(include=[np.number])
    if numeric.shape[1] == 0:
        _warn("Numeric columns required: no numeric columns found for cv.")
        return pd.DataFrame(columns=["feature", "cv"])

    rows = []
    for col in numeric.columns:
        mean = numeric[col].mean()
        if mean == 0:
            rows.append((col, float("inf")))
        else:
            rows.append((col, _round(numeric[col].std() / abs(mean) * 100.0, decimals)))

    result = pd.DataFrame(rows, columns=["feature", "cv"])
    return result.sort_values("cv", ascending=False).reset_index(drop=True)


def outliers(data: Union[pd.Series, pd.DataFrame], decimals: Optional[int] = 2, multiplier: float = 1.5) -> Union[float, pd.DataFrame]:
    """
    Calculate the percentage of outliers using the IQR method.

    An outlier is any value below Q1 - multiplier*IQR or above Q3 + multiplier*IQR.

    Args:
        data: A Series (returns a single float) or DataFrame (returns a DataFrame
            of all numeric columns).
        decimals: Number of decimal places to round to.
        multiplier: IQR multiplier for the outlier bounds (default: 1.5).

    Returns:
        float for a Series, or a DataFrame with columns ["feature", "outlier_pct"]
        sorted highest first. Non-numeric input is handled with a warning.
    """
    def _calc(s: pd.Series) -> float:
        s = s.dropna()
        if len(s) == 0:
            return 0.0
        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - multiplier * iqr
        upper = q3 + multiplier * iqr
        count = ((s < lower) | (s > upper)).sum()
        return count / len(s) * 100.0

    if isinstance(data, pd.Series):
        if not pd.api.types.is_numeric_dtype(data):
            _warn(f"outliers expects numeric data, but got a non-numeric Series "
                  f"(dtype: {data.dtype}). Returning NaN. Encode or select a numeric column.")
            return float("nan")
        return _round(_calc(data), decimals)

    if not isinstance(data, pd.DataFrame):
        raise TypeError(f"outliers expects a Series or DataFrame, got {type(data).__name__}.")

    numeric = data.select_dtypes(include=[np.number])
    if numeric.shape[1] == 0:
        _warn("Numeric columns required: no numeric columns found for outliers.")
        return pd.DataFrame(columns=["feature", "outlier_pct"])

    rows = [(col, _round(_calc(numeric[col]), decimals)) for col in numeric.columns]
    result = pd.DataFrame(rows, columns=["feature", "outlier_pct"])
    return result.sort_values("outlier_pct", ascending=False).reset_index(drop=True)


def r_squared(y_true: Union[pd.Series, Sequence, np.ndarray], y_pred: Union[pd.Series, Sequence, np.ndarray], decimals: Optional[int] = 2) -> float:
    """
    Calculate R-squared (coefficient of determination).

    R² = 1 - (SS_res / SS_tot), expressed as a percentage.

    Args:
        y_true: Actual values.
        y_pred: Predicted values.
        decimals: Number of decimal places to round to.

    Returns:
        float: R-squared as a percentage (e.g. 87.3 means 87.3%).

    Raises:
        ValueError: If inputs are non-numeric, differ in length, or have < 2 values.
    """
    try:
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
    except (ValueError, TypeError):
        raise ValueError("r_squared expects numeric values, but got non-numeric input.")

    if len(y_true) != len(y_pred):
        raise ValueError("`y_true` and `y_pred` must have the same length.")

    if len(y_true) < 2:
        raise ValueError("Need at least 2 values to compute R-squared.")

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

    if ss_tot == 0:
        return 0.0

    val = (1.0 - ss_res / ss_tot) * 100.0
    return _round(val, decimals)


def pca_variance(df: pd.DataFrame, decimals: Optional[int] = 2, n_components: Optional[int] = None, standardize: bool = True) -> pd.DataFrame:
    """
    Calculate the percentage of variance explained by each principal component.

    Performs PCA via eigendecomposition. By default every column is standardized
    to unit variance first (correlation-based PCA), so that a column measured in
    large units (e.g. dollars) cannot dominate the result purely because of its
    scale. Set ``standardize=False`` for covariance-based PCA on the raw values.

    Non-numeric columns are ignored. If fewer than 2 usable numeric columns are
    available, a warning is raised and an empty DataFrame is returned.

    Args:
        df: DataFrame with numeric columns.
        decimals: Number of decimal places to round to.
        n_components: Number of components to return. If None, returns all.
        standardize: If True (default), scale each column to unit variance
            before the decomposition. Constant (zero-variance) columns are
            dropped when standardizing, since they cannot be scaled.

    Returns:
        DataFrame with columns ["component", "variance_explained", "cumulative"].
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"pca_variance expects a pandas DataFrame, got {type(df).__name__}.")

    empty = pd.DataFrame(columns=["component", "variance_explained", "cumulative"])
    numeric = df.select_dtypes(include=[np.number]).dropna()

    if numeric.shape[1] < 2:
        _warn("Numeric columns required: pca_variance needs at least "
              "2 numeric columns.")
        return empty

    if numeric.shape[0] < 2:
        _warn("Not enough data: need at least 2 complete (non-null) rows.")
        return empty

    X = numeric.values.astype(float)

    if standardize:
        keep = X.std(axis=0, ddof=1) > 0
        if keep.sum() < 2:
            _warn("Numeric columns required: after dropping constant (zero-variance) "
                  "columns, fewer than 2 columns remain to standardize for PCA.")
            return empty
        X = X[:, keep]
        X = (X - X.mean(axis=0)) / X.std(axis=0, ddof=1)

    cov_matrix = np.cov(X, rowvar=False)
    eigenvalues, _ = np.linalg.eigh(cov_matrix)

    eigenvalues = eigenvalues[::-1]
    total = eigenvalues.sum()

    if total == 0:
        _warn("All numeric columns are constant - there is no variance to explain.")
        return empty

    ratios = eigenvalues / total * 100.0
    if n_components is not None:
        ratios = ratios[:n_components]

    cumulative = np.cumsum(ratios)
    return pd.DataFrame({
        "component": [f"PC{i + 1}" for i in range(len(ratios))],
        "variance_explained": [_round(r, decimals) for r in ratios],
        "cumulative": [_round(c, decimals) for c in cumulative],
    })

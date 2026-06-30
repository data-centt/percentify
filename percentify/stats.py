from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd


def vif(df: pd.DataFrame, decimals: Optional[int] = 2, flag: Optional[float] = None) -> Dict[str, float]:
    """
    Calculate the Variance Inflation Factor for each numeric column in a DataFrame.

    VIF = 1 / (1 - R²), where R² comes from regressing each feature
    against all other features. VIF > 5 suggests moderate multicollinearity,
    VIF > 10 suggests severe multicollinearity.

    Args:
        df: DataFrame with numeric columns.
        decimals: Number of decimal places to round to.
            If None, raw floats are returned.
        flag: If set, only return columns with VIF above this threshold.

    Returns:
        dict: Column names mapped to their VIF values.

    Raises:
        ValueError: If DataFrame has fewer than 2 numeric columns.
    """
    numeric = df.select_dtypes(include=[np.number])

    if numeric.shape[1] < 2:
        raise ValueError("DataFrame must have at least 2 numeric columns.")

    numeric = numeric.dropna()

    if numeric.shape[0] < 2:
        raise ValueError("DataFrame must have at least 2 non-null rows.")

    cols = numeric.columns.tolist()
    X = numeric.values.astype(float)

    result = {}
    for i, col in enumerate(cols):
        y = X[:, i]
        others = np.delete(X, i, axis=1)

        ones = np.ones((others.shape[0], 1))
        others_with_intercept = np.hstack([ones, others])

        coeffs, residuals, _, _ = np.linalg.lstsq(others_with_intercept, y, rcond=None)

        y_pred = others_with_intercept @ coeffs
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)

        if ss_tot == 0:
            r_sq = 0.0
        else:
            r_sq = 1.0 - (ss_res / ss_tot)

        if r_sq >= 1.0:
            vif_val = float("inf")
        else:
            vif_val = 1.0 / (1.0 - r_sq)

        if decimals is not None:
            vif_val = round(vif_val, decimals)

        result[col] = vif_val

    if flag is not None:
        result = {k: v for k, v in result.items() if v > flag}

    return result


def _round(value: float, decimals: Optional[int]) -> float:
    if decimals is None:
        return value
    return round(value, decimals)


def missing(df: pd.DataFrame, decimals: Optional[int] = 2) -> Dict[str, float]:
    """
    Calculate the percentage of missing values for each column.

    Args:
        df: DataFrame to profile.
        decimals: Number of decimal places to round to.

    Returns:
        dict: Column names mapped to their missing percentage,
            sorted from highest to lowest.
    """
    total = len(df)
    if total == 0:
        return {col: 0.0 for col in df.columns}

    result = {}
    for col in df.columns:
        pct = df[col].isnull().sum() / total * 100.0
        result[col] = _round(pct, decimals)

    return dict(sorted(result.items(), key=lambda x: x[1], reverse=True))


def cv(data: Union[pd.Series, pd.DataFrame], decimals: Optional[int] = 2) -> Union[float, Dict[str, float]]:
    """
    Calculate the coefficient of variation (CV = std / mean * 100).

    Args:
        data: A Series (single result) or DataFrame (all numeric columns).
        decimals: Number of decimal places to round to.

    Returns:
        float if Series, dict if DataFrame.

    Raises:
        ValueError: If the mean is zero (CV is undefined).
    """
    if isinstance(data, pd.Series):
        mean = data.mean()
        if mean == 0:
            raise ValueError("Coefficient of variation is undefined when mean is zero.")
        val = data.std() / abs(mean) * 100.0
        return _round(val, decimals)

    numeric = data.select_dtypes(include=[np.number])
    result = {}
    for col in numeric.columns:
        mean = numeric[col].mean()
        if mean == 0:
            result[col] = float("inf")
        else:
            val = numeric[col].std() / abs(mean) * 100.0
            result[col] = _round(val, decimals)
    return result


def outliers(data: Union[pd.Series, pd.DataFrame], decimals: Optional[int] = 2, multiplier: float = 1.5) -> Union[float, Dict[str, float]]:
    """
    Calculate the percentage of outliers using the IQR method.

    An outlier is any value below Q1 - multiplier*IQR or above Q3 + multiplier*IQR.

    Args:
        data: A Series (single result) or DataFrame (all numeric columns).
        decimals: Number of decimal places to round to.
        multiplier: IQR multiplier for defining outlier bounds (default: 1.5).

    Returns:
        float if Series, dict if DataFrame.
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
        return _round(_calc(data), decimals)

    numeric = data.select_dtypes(include=[np.number])
    result = {}
    for col in numeric.columns:
        result[col] = _round(_calc(numeric[col]), decimals)
    return result


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
        ValueError: If inputs have different lengths or fewer than 2 values.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

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


def variance_explained(df: pd.DataFrame, decimals: Optional[int] = 2, n_components: Optional[int] = None) -> Dict[str, float]:
    """
    Calculate the percentage of variance explained by each principal component.

    Performs PCA using eigendecomposition of the covariance matrix.

    Args:
        df: DataFrame with numeric columns.
        decimals: Number of decimal places to round to.
        n_components: Number of components to return. If None, returns all.

    Returns:
        dict: Component names mapped to their explained variance percentage.
            e.g. {"PC1": 45.2, "PC2": 23.1, ...}

    Raises:
        ValueError: If DataFrame has fewer than 2 numeric columns or rows.
    """
    numeric = df.select_dtypes(include=[np.number]).dropna()

    if numeric.shape[1] < 2:
        raise ValueError("DataFrame must have at least 2 numeric columns.")

    if numeric.shape[0] < 2:
        raise ValueError("DataFrame must have at least 2 non-null rows.")

    X = numeric.values.astype(float)
    X_centered = X - X.mean(axis=0)

    cov_matrix = np.cov(X_centered, rowvar=False)
    eigenvalues, _ = np.linalg.eigh(cov_matrix)

    eigenvalues = eigenvalues[::-1]
    total = eigenvalues.sum()

    if total == 0:
        return {}

    ratios = eigenvalues / total * 100.0

    if n_components is not None:
        ratios = ratios[:n_components]

    result = {}
    for i, r in enumerate(ratios):
        result[f"PC{i + 1}"] = _round(r, decimals)

    return result

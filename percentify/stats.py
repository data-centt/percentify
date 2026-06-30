from typing import Dict, List, Optional, Union

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

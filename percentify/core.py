from typing import SupportsFloat, Optional

def percent(part: SupportsFloat, whole: SupportsFloat, decimals: Optional[int] = 2) -> float:
    """
    Calculate what percentage `part` is of the `whole`.

    Args:
        part: The numerator.
        whole: The denominator.
        decimals: Number of decimal places to round to.
            If None, the raw percentage (unrounded float) is returned.

    Returns:
        float: Percentage value. If `whole` is 0, returns 0.0.
    """
    whole = float(whole)
    if whole == 0:
        return 0.0

    value = float(part) / whole * 100.0

    if decimals is None:
        return value

    if decimals < 0:
        raise ValueError("`decimals` must be non-negative or None.")

    return round(value, decimals)

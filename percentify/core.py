from typing import List, SupportsFloat, Optional


def _round_value(value: float, decimals: Optional[int]) -> float:
    if decimals is None:
        return value
    if decimals < 0:
        raise ValueError("`decimals` must be non-negative or None.")
    return round(value, decimals)


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
    return _round_value(value, decimals)


def percent_change(old: SupportsFloat, new: SupportsFloat, decimals: Optional[int] = 2) -> float:
    """
    Calculate the percentage change from `old` to `new`.

    Args:
        old: The original value.
        new: The new value.
        decimals: Number of decimal places to round to.
            If None, the raw percentage (unrounded float) is returned.

    Returns:
        float: Percentage change. Positive means increase, negative means decrease.
            If `old` is 0, returns 0.0.
    """
    old = float(old)
    if old == 0:
        return 0.0

    value = (float(new) - old) / abs(old) * 100.0
    return _round_value(value, decimals)


def percent_diff(a: SupportsFloat, b: SupportsFloat, decimals: Optional[int] = 2) -> float:
    """
    Calculate the percentage difference between two values.

    Uses the average of the two values as the denominator.

    Args:
        a: First value.
        b: Second value.
        decimals: Number of decimal places to round to.
            If None, the raw percentage (unrounded float) is returned.

    Returns:
        float: Percentage difference (always non-negative).
            If both values are 0, returns 0.0.
    """
    a, b = float(a), float(b)
    avg = (abs(a) + abs(b)) / 2.0
    if avg == 0:
        return 0.0

    value = abs(a - b) / avg * 100.0
    return _round_value(value, decimals)


def percent_distribute(total: SupportsFloat, weights: List[SupportsFloat], decimals: Optional[int] = 2) -> List[float]:
    """
    Distribute a total into percentage-based shares according to weights.

    Args:
        total: The total value to distribute.
        weights: A list of weights determining each share's proportion.
        decimals: Number of decimal places to round to.
            If None, the raw values (unrounded floats) are returned.

    Returns:
        list[float]: Each weight's share of the total.

    Raises:
        ValueError: If weights is empty or all weights are zero.
    """
    if not weights:
        raise ValueError("`weights` must not be empty.")

    float_weights = [float(w) for w in weights]
    weight_sum = sum(float_weights)

    if weight_sum == 0:
        raise ValueError("Sum of `weights` must not be zero.")

    total = float(total)
    return [_round_value(w / weight_sum * total, decimals) for w in float_weights]


def percent_format(value: SupportsFloat, decimals: Optional[int] = 2, suffix: str = "%") -> str:
    """
    Format a numeric value as a percentage string.

    Args:
        value: The percentage value to format.
        decimals: Number of decimal places to round to.
            If None, the raw float is used without rounding.
        suffix: The suffix to append (default: "%").

    Returns:
        str: Formatted percentage string, e.g. "25.0%".
    """
    rounded = _round_value(float(value), decimals)
    return f"{rounded}{suffix}"

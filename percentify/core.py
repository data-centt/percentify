def percent(part: float, whole: float, decimals: int | None = 2) -> float:
    """
    Easily calculate what percentage `part` is of the `whole`.

    Args:
        part (float): The numerator.
        whole (float): The denominator.
        decimals (int | None): Number of decimal places to round.
            If None, no rounding is applied.

    Returns:
        float: Percentage value.
    """
    if whole == 0:
        return 0.0
    value = (part / whole) * 100
    return round(value, decimals) if decimals is not None else value

def percent(part: float, whole: float, decimals: int = 2) -> float:
    """
    Easily calculate what percentage `part` is of the `whole`.
    
    Args:
        part (float): The numerator.
        whole (float): The denominator.
        decimals (int): Number of decimal places to round.
        
    Returns:
        float: Percentage value.
    """
    if whole == 0:
        return 0.0
    return round((part / whole) * 100, decimals)

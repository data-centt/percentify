import pytest
from percentify import percent, change, difference, split, display


# --- percent ---

def test_percent_normal():
    assert percent(50, 200) == 25.0

def test_percent_fraction():
    assert percent(1, 3) == 33.33

def test_percent_zero_division():
    assert percent(5, 0) == 0.0

def test_percent_no_rounding():
    assert percent(1, 3, None) == pytest.approx(33.33333333333333)

def test_percent_custom_decimals():
    assert percent(7, 9, 4) == 77.7778

def test_percent_negative_decimals():
    with pytest.raises(ValueError):
        percent(1, 3, -1)


# --- change ---

def test_change_increase():
    assert change(100, 150) == 50.0

def test_change_decrease():
    assert change(200, 150) == -25.0

def test_change_no_change():
    assert change(100, 100) == 0.0

def test_change_zero_old():
    assert change(0, 100) == 0.0

def test_change_negative_old():
    assert change(-100, -50) == 50.0

def test_change_custom_decimals():
    assert change(3, 7, 4) == 133.3333


# --- difference ---

def test_difference_basic():
    assert difference(10, 20) == 66.67

def test_difference_same():
    assert difference(50, 50) == 0.0

def test_difference_both_zero():
    assert difference(0, 0) == 0.0

def test_difference_order_independent():
    assert difference(10, 20) == difference(20, 10)

def test_difference_custom_decimals():
    assert difference(3, 7, 4) == 80.0


# --- split ---

def test_split_equal():
    assert split(100, [1, 1, 1]) == [pytest.approx(33.33), pytest.approx(33.33), pytest.approx(33.33)]

def test_split_weighted():
    assert split(200, [1, 3]) == [50.0, 150.0]

def test_split_single():
    assert split(100, [5]) == [100.0]

def test_split_empty():
    with pytest.raises(ValueError):
        split(100, [])

def test_split_zero_weights():
    with pytest.raises(ValueError):
        split(100, [0, 0])

def test_split_custom_decimals():
    result = split(100, [1, 1, 1], 4)
    assert result == [pytest.approx(33.3333), pytest.approx(33.3333), pytest.approx(33.3333)]


# --- display ---

def test_display_basic():
    assert display(25.0) == "25.0%"

def test_display_custom_decimals():
    assert display(33.3333, 1) == "33.3%"

def test_display_custom_suffix():
    assert display(50, suffix=" percent") == "50.0 percent"

def test_display_no_rounding():
    result = display(33.3333, None)
    assert result == "33.3333%"

def test_display_zero():
    assert display(0) == "0.0%"

def test_display_multiply():
    assert display(0.45, multiply=True) == "45.0%"

def test_display_multiply_small():
    assert display(0.0725, multiply=True) == "7.25%"

def test_display_multiply_false():
    assert display(0.45) == "0.45%"

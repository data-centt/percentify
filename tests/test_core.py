import pytest
from percentify import percent, percent_change, percent_diff, percent_distribute, percent_format


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


# --- percent_change ---

def test_percent_change_increase():
    assert percent_change(100, 150) == 50.0

def test_percent_change_decrease():
    assert percent_change(200, 150) == -25.0

def test_percent_change_no_change():
    assert percent_change(100, 100) == 0.0

def test_percent_change_zero_old():
    assert percent_change(0, 100) == 0.0

def test_percent_change_negative_old():
    assert percent_change(-100, -50) == 50.0

def test_percent_change_custom_decimals():
    assert percent_change(3, 7, 4) == 133.3333


# --- percent_diff ---

def test_percent_diff_basic():
    assert percent_diff(10, 20) == 66.67

def test_percent_diff_same():
    assert percent_diff(50, 50) == 0.0

def test_percent_diff_both_zero():
    assert percent_diff(0, 0) == 0.0

def test_percent_diff_order_independent():
    assert percent_diff(10, 20) == percent_diff(20, 10)

def test_percent_diff_custom_decimals():
    assert percent_diff(3, 7, 4) == 80.0


# --- percent_distribute ---

def test_distribute_equal():
    assert percent_distribute(100, [1, 1, 1]) == [pytest.approx(33.33), pytest.approx(33.33), pytest.approx(33.33)]

def test_distribute_weighted():
    assert percent_distribute(200, [1, 3]) == [50.0, 150.0]

def test_distribute_single():
    assert percent_distribute(100, [5]) == [100.0]

def test_distribute_empty():
    with pytest.raises(ValueError):
        percent_distribute(100, [])

def test_distribute_zero_weights():
    with pytest.raises(ValueError):
        percent_distribute(100, [0, 0])

def test_distribute_custom_decimals():
    result = percent_distribute(100, [1, 1, 1], 4)
    assert result == [pytest.approx(33.3333), pytest.approx(33.3333), pytest.approx(33.3333)]


# --- percent_format ---

def test_format_basic():
    assert percent_format(25.0) == "25.0%"

def test_format_custom_decimals():
    assert percent_format(33.3333, 1) == "33.3%"

def test_format_custom_suffix():
    assert percent_format(50, suffix=" percent") == "50.0 percent"

def test_format_no_rounding():
    result = percent_format(33.3333, None)
    assert result == "33.3333%"

def test_format_zero():
    assert percent_format(0) == "0.0%"

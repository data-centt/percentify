from percentify import percent

def test_percent_normal():
    assert percent(50, 200) == 25.0

def test_percent_fraction():
    assert percent(1, 3) == 33.33

def test_percent_zero_division():
    assert percent(5, 0) == 0.0

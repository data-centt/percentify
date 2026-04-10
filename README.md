#                                              % Percentify %
[![PyPI Downloads](https://static.pepy.tech/personalized-badge/percentify?period=total&units=INTERNATIONAL_SYSTEM&left_color=BLACK&right_color=GREEN&left_text=downloads)](https://pepy.tech/projects/percentify)
[![PyPI version](https://img.shields.io/pypi/v/percentify.svg?style=flat&color=blue)](https://pypi.org/project/percentify/)
[![Python Versions](https://img.shields.io/pypi/pyversions/percentify.svg?style=flat&color=green)](https://pypi.org/project/percentify/)
[![License](https://img.shields.io/pypi/l/percentify.svg?style=flat&color=orange)](LICENSE)
[![Build Status](https://github.com/data-centt/percentify/actions/workflows/python-app.yml/badge.svg)](https://github.com/data-centt/percentify/actions/workflows/python-app.yml)

**Percentify** — a tiny Python helper that turns *"part of a whole"* into a clean percentage.  
Stop typing `(part / whole) * 100` and worrying about division by zero.

---

## ✨ What It Does

A tiny, zero-dependency Python toolkit for all things percentages:

- **`percent`** — what percentage is `part` of `whole`?
- **`percent_change`** — how much did a value increase or decrease?
- **`percent_diff`** — how far apart are two values?
- **`percent_distribute`** — split a total into weighted shares.
- **`percent_format`** — turn any number into a clean `"25.0%"` string.

All functions handle edge cases (division by zero, negative values) safely and let you control decimal precision.

## 📦 Installation
```
pip install percentify
```

## Usage

### `percent` — Part of a Whole
```python
from percentify import percent

percent(50, 200)          # → 25.0
percent(1, 3)             # → 33.33
percent(5, 0)             # → 0.0  (safe division by zero)
percent(7, 9, 4)          # → 77.7778  (custom decimals)
```

### `percent_change` — Increase or Decrease
```python
from percentify import percent_change

percent_change(100, 150)  # → 50.0   (50% increase)
percent_change(200, 150)  # → -25.0  (25% decrease)
percent_change(0, 100)    # → 0.0    (safe when old is zero)
```

### `percent_diff` — Difference Between Two Values
```python
from percentify import percent_diff

percent_diff(10, 20)      # → 66.67
percent_diff(50, 50)      # → 0.0
```

### `percent_distribute` — Split a Total by Weights
```python
from percentify import percent_distribute

percent_distribute(200, [1, 3])       # → [50.0, 150.0]
percent_distribute(100, [1, 1, 1])    # → [33.33, 33.33, 33.33]
```

### `percent_format` — Format as a String
```python
from percentify import percent_format

percent_format(25.0)                  # → "25.0%"
percent_format(33.3333, 1)            # → "33.3%"
percent_format(50, suffix=" percent") # → "50.0 percent"
```

# 🤝 Contributing

Contributions are welcome!
- If you have an idea (extra helpers, bug fixes or an idea):
- Fork this repo
- Create a branch
- Commit your changes
- Open a pull request

I try to keep it tiny on purpose, to discuss big new features first.

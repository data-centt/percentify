#                                              % Percentify %

[![PyPI version](https://img.shields.io/pypi/v/percentify.svg?style=flat&color=blue)](https://pypi.org/project/percentify/)
[![Python Versions](https://img.shields.io/pypi/pyversions/percentify.svg?style=flat&color=green)](https://pypi.org/project/percentify/)
[![License](https://img.shields.io/pypi/l/percentify.svg?style=flat&color=orange)](LICENSE)
[![Build Status](https://github.com/data-centt/percentify/actions/workflows/python-app.yml/badge.svg)](https://github.com/data-centt/percentify/actions/workflows/python-app.yml)

**Percentify** — a tiny Python helper that turns *"part of a whole"* into a clean percentage.  
Stop typing `(part / whole) * 100` and worrying about division by zero.

---

## ✨ What It Does

Percentify gives you a single function:

- Calculates what percentage one number is of another.
- Handles divide-by-zero safely (returns 0.0 instead of crashing).
- Lets you choose how many decimal places to round to.
- Has zero dependencies — just pure Python.

## 📦 Installation
```
pip install percentify
```

### Usage
```
from percentify import percent

# Basic usage
percent(50, 200)          # → 25.0

# Handles fractions
percent(1, 3)             # → 33.33

# Safe when dividing by zero
percent(5, 0)             # → 0.0

# Custom decimals
percent(7, 9, 4)          # → 77.7778
```

### 🛠️ How It Works

The library is intentionally simple — just one function:
```
def percent(part: float, whole: float, decimals: int = 2) -> float:
    if whole == 0:
        return 0.0
    return round((part / whole) * 100, decimals)
```
That’s it. Clean, safe, and ready to use anywhere you need percentages in your code.

# 🤝 Contributing

Contributions are welcome!
- If you have an idea (extra helpers, bug fixes or an idea):
- Fork this repo
- Create a branch
- Commit your changes
- Open a pull request

I try to keep it tiny on purpose, to discuss big new features first.

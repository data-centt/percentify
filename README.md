
<p align="center">
  <img src="asset/log.png" alt="Percentify logo" height="150">
</p>

[![Downloads](https://static.pepy.tech/badge/percentify)](https://pepy.tech/project/percentify)
[![PyPI version](https://img.shields.io/pypi/v/percentify.svg?style=flat&color=blue)](https://pypi.org/project/percentify/)
[![Python Versions](https://img.shields.io/pypi/pyversions/percentify.svg?style=flat&color=green)](https://pypi.org/project/percentify/)
[![License](https://img.shields.io/pypi/l/percentify.svg?style=flat&color=orange)](LICENSE)
[![Docs](https://img.shields.io/badge/docs-percentify-14b8a6)](https://data-centt.github.io/percentify/)
[![Build Status](https://github.com/data-centt/percentify/actions/workflows/python-app.yml/badge.svg)](https://github.com/data-centt/percentify/actions/workflows/python-app.yml)
[![Polars](https://img.shields.io/badge/Polars-supported-cd792c?style=flat)](https://data-centt.github.io/percentify/)

**Percentify is a data science library that turns the 20% of data science operations behind 80% of daily work into single, readable function calls.**

**⚡ Full Polars DataFrame support:** pass a polars DataFrame or Series and get the same kind back, no flag needed.


Built on pandas and numpy, it pairs everyday hard to reach tools with lesser-known ones. Where a function wraps an existing library (pandas, scipy, statsmodels, scikit-learn), it names it, so you always know where to dig deeper.

## 📖 Documentation

**Full guide, every function, and live examples → [data-centt.github.io/percentify](https://data-centt.github.io/percentify/)**

## 📦 Installation

```bash
pip install percentify
```

Requires `numpy` and `pandas`.

## Quick example

```python
import pandas as pd
from percentify import missing

df = pd.DataFrame({
    "salary": [50000, None, 60000, None],
    "age":    [25, 30, None, 40],
    "city":   ["NY", "LA", "SF", "LA"],
})

missing(df)
#    column  missing_pct
# 0  salary         50.0
# 1     age         25.0
# 2    city          0.0
```

One import, one line. A clean, sorted DataFrame you can read or feed into the next step.

## What's inside

| Function | What it answers |
|---|---|
| `profile` | What is wrong with this dataset, and how do I fix it? |
| `change` | Growth as numbers, columns, or a whole series |
| `vif` | Which features are collinear? |
| `missing` | How much of each column is missing? |
| `cv` | How variable is each column, relative to its mean? |
| `outliers` | What percentage of each column are outliers? |
| `r_squared` | How well do predictions fit? |
| `pca_variance` | How much variance does each principal component explain? |
| `imbalance` | How skewed are the classes in a target column? |
| `difference` | How far apart are two values or columns? |
| `split` | How does a total divide across weights or groups? |
| `display` | Format numbers or a column as clean "%" strings |

→ See the **[documentation](https://data-centt.github.io/percentify/)** for a worked, real-output example of every function.

## 🛟 Friendly by design

- **No cryptic tracebacks**; Hand a function a text column where numbers are needed and you get a clear PercentifyWarning, not an Arrow/NumPy stack trace.
- **Sensible defaults**; Results come back sorted worst-first, and PCA is standardized out of the box.
- **DataFrames everywhere**; so the output drops straight into your notebook, your next filter, or your model.
- **Pandas or polars**; pass either a pandas or polars object and you get the same kind back, no flag needed.


## 🤝 Contributing

Contributions are welcome but they must follow the repo's guiding principle:
> Keep each method as direct-to-output as possible. A percentify function should return the single most common answer in one line, and point users to the underlying library (pandas, scipy, statsmodels, scikit-learn) for the full, configurable version when the simplest output isn't what they're after.

**It must support polars.** Every function accepts both pandas and polars objects (via the `@_backend_aware` decorator) and returns the same kind, so any new contribution must keep that parity.

If your idea keeps things that simple and direct:
- Open an issue first to discuss it
- Fork the repo
- Create a branch
- Commit your changes
- Open a pull request

> Anything that adds knobs and options for their own sake, or duplicates what the parent libraries already do well, is out of scope; those cases should point to the source library instead.
I try to keep it compact on purpose, to discuss big new features first.

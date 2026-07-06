#                                               Percentify
[![PyPI Downloads](https://static.pepy.tech/personalized-badge/percentify?period=total&units=INTERNATIONAL_SYSTEM&left_color=BLACK&right_color=GREEN&left_text=downloads)](https://pepy.tech/projects/percentify)
[![PyPI version](https://img.shields.io/pypi/v/percentify.svg?style=flat&color=blue)](https://pypi.org/project/percentify/)
[![Python Versions](https://img.shields.io/pypi/pyversions/percentify.svg?style=flat&color=green)](https://pypi.org/project/percentify/)
[![License](https://img.shields.io/pypi/l/percentify.svg?style=flat&color=orange)](LICENSE)
[![Build Status](https://github.com/data-centt/percentify/actions/workflows/python-app.yml/badge.svg)](https://github.com/data-centt/percentify/actions/workflows/python-app.yml)

**Percentify** brings the everyday data-science checks to your doorstep — one import, one line.

Stop digging through scipy, statsmodels, and sklearn for the operations you run on every dataset. Percentify surfaces the common answer — the one 80% of practitioners actually want — as a single readable function call. Need to go deeper? The underlying libraries are still right there.

Every function takes a pandas `DataFrame` (or `Series`) and hands back a clean `DataFrame` you can read, sort, or feed straight into the next step.

---

## 📦 Installation
```
pip install percentify
```

Requires `numpy` and `pandas`.

---

## 📊 The Toolkit

### `change` — Percentage Change
Two numbers, two columns, or a whole series at once.
```python
from percentify import change

change(100, 150)                   # → 50.0   (a 50% increase)

change(df["forecast"], df["actual"])   # element-wise % change between two columns

change(df["revenue"])              # period-over-period % change down the column
# 0     NaN
# 1    50.0
# 2   -40.0

change(df)                         # every numeric column at once
```

### `vif` — Variance Inflation Factor (Multicollinearity)
Buried in `statsmodels.stats.outliers_influence` behind a six-line loop. One call here.
```python
from percentify import vif

vif(df)
#   feature   VIF
# 0  income  8.40
# 1    debt  7.90
# 2     age  1.20

vif(df, flag=5.0)   # only rows above the threshold (your problem columns)
```

### `missing` — Missing Data Profiling
No more `df.isnull().sum() / len(df) * 100`.
```python
from percentify import missing

missing(df)
#    column  missing_pct
# 0  salary        12.40
# 1     age         3.10
# 2    name         0.00
```

### `cv` — Coefficient of Variation
Not built in anywhere — instead of `df.std() / df.mean() * 100`.
```python
from percentify import cv

cv(df["salary"])   # → 34.2   (a single Series returns a number)
cv(df)             # → DataFrame of every numeric column, most variable first
```

### `outliers` — Percentage of Outliers (IQR Method)
Stop rewriting the IQR bounds from scratch.
```python
from percentify import outliers

outliers(df["salary"])   # → 4.7
outliers(df)             # → DataFrame of every numeric column
```

### `r_squared` — R-Squared
```python
from percentify import r_squared

r_squared(y_true, y_pred)   # → 87.3
```

### `pca_variance` — PCA Variance Breakdown
Columns are standardized by default, so a feature measured in large units (e.g.
dollars) can't dominate the result just because of its scale. Pass
`standardize=False` for covariance-based PCA on the raw values.
```python
from percentify import pca_variance

pca_variance(df)
#   component  variance_explained  cumulative
# 0       PC1               45.2        45.2
# 1       PC2               23.1        68.3
# 2       PC3               12.8        81.1

pca_variance(df, standardize=False)   # covariance-based (scale-sensitive)
```

---

## 🛟 Friendly by design

Built for early-career analysts as much as seasoned ones:

- **No cryptic tracebacks.** Hand it a text column where numbers are needed and you get a clear `PercentifyWarning` ("numeric columns required") plus an empty result — not an Arrow/NumPy stack trace.
- **Sensible defaults.** Results come back sorted worst-first (most collinear, most missing, most variable), and PCA is standardized out of the box.
- **DataFrames everywhere**, so the output drops straight into your notebook, your next filter, or your model.

Catch or silence the warnings like any other:
```python
import warnings
from percentify import PercentifyWarning

warnings.filterwarnings("ignore", category=PercentifyWarning)
```

---

## 🤝 Contributing

Contributions are welcome!
- If you have an idea (extra helpers, bug fixes or an idea):
- Fork this repo
- Create a branch
- Commit your changes
- Open a pull request

I try to keep it within scope, so please open an issue to discuss big new features first.

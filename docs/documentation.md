# Documentation

Every example on this page is real — the output shown is exactly what the function returns.

## Installation

```bash
pip install percentify
```

Percentify requires `numpy` and `pandas`.

```python
from percentify import change, vif, missing, cv, outliers, r_squared, pca_variance
```

---

## Conventions

A few rules hold across the whole library, so you always know what to expect:

- **DataFrame in → DataFrame out.** Pass a DataFrame and you get a clean DataFrame back. Pass a single `Series` (where it makes sense) and you get a single number.
- **Sorted worst-first.** Results come back ordered by what usually matters most — most collinear, most missing, most variable, most outliers.
- **`decimals` everywhere.** Every function accepts a `decimals` argument (default `2`). Pass `decimals=None` for full precision.
- **Numeric-only functions ignore text columns.** They quietly skip non-numeric columns instead of crashing.
- **Friendly warnings, not tracebacks.** Hand a function something it can't use (e.g. an all-text DataFrame) and you get a `PercentifyWarning` explaining why, plus an empty result — never a raw NumPy/pandas stack trace. See [Warnings](#warnings-and-non-numeric-data).

---

## Polars support

Every function accepts **polars** DataFrames and Series as well as pandas, and hands back the same kind you passed in: polars in, polars out. Detection is automatic from the input type, so there is nothing to configure.

```python
import polars as pl
from percentify import missing

df = pl.DataFrame({"salary": [50000, None, 60000, None], "age": [25, 30, None, 40]})

missing(df)   # returns a polars DataFrame
```

polars stays optional: it is only imported when you actually pass a polars object, so pandas-only users pay nothing. Conversion goes through Arrow, so `pyarrow` must be installed for the polars path (it usually already is in a polars setup).

---

## `change`

Percentage change — as two numbers, between two columns, or down a whole series.

!!! tip "Similar concept"
    `pandas.DataFrame.pct_change`

**Signature**

```python
change(old, new=None, decimals=2)
```

**Two numbers**

```python
from percentify import change

change(100, 150)
```

```text
50.0
```

**Between two columns** (element-wise) — perfect for a new column:

```python
import pandas as pd

df = pd.DataFrame({
    "forecast": [100, 200, 50, 80],
    "actual":   [150, 150, 100, 80],
})

df["change_pct"] = change(df["forecast"], df["actual"])
df
```

```text
   forecast  actual  change_pct
0       100     150        50.0
1       200     150       -25.0
2        50     100       100.0
3        80      80         0.0
```

**Down one column** (period-over-period):

```python
change(pd.Series([100, 150, 90, 135]))
```

```text
0     NaN
1    50.0
2   -40.0
3    50.0
dtype: float64
```

The first value is `NaN` because there is no prior period to compare against. Passing a whole `DataFrame` applies this to every numeric column at once.

!!! note
    Where `old` is `0`, the result is `0.0` (safe division). Two Series are aligned by **index**, not position — fine when both columns come from the same DataFrame.

---

## `vif`

Variance Inflation Factor — the classic multicollinearity check, without the six-line loop.

!!! tip "Similar concept"
    `statsmodels.stats.outliers_influence.variance_inflation_factor`

**Signature**

```python
vif(df, decimals=2, flag=None)
```

**Example**

```python
import numpy as np, pandas as pd
from percentify import vif

np.random.seed(7)
base = np.random.randn(200)
df = pd.DataFrame({
    "age":    base + np.random.randn(200) * 0.1,
    "income": base * 2 + np.random.randn(200) * 0.1,   # closely tracks age
    "score":  np.random.randn(200),
})

vif(df)
```

```text
  feature    VIF
0  income  83.54
1     age  83.53
2   score   1.01
```

`age` and `income` have sky-high VIFs (they carry the same signal), while `score` is independent.

**Flag only the problem columns** with `flag`:

```python
vif(df, flag=5.0)
```

```text
  feature    VIF
0  income  83.54
1     age  83.53
```

!!! info "Rules of thumb"
    VIF > 5 suggests moderate multicollinearity; VIF > 10 suggests severe. A perfectly collinear column returns `inf`.

---

## `missing`

The percentage of missing values in each column, sorted highest first.

!!! tip "Similar concept"
    `pandas.DataFrame.isna`

**Signature**

```python
missing(df, decimals=2)
```

**Example**

```python
import pandas as pd
from percentify import missing

df = pd.DataFrame({
    "salary": [50000, None, 60000, None],
    "age":    [25, 30, None, 40],
    "city":   ["NY", "LA", "SF", "LA"],
})

missing(df)
```

```text
   column  missing_pct
0  salary         50.0
1     age         25.0
2    city          0.0
```

Unlike the numeric-only functions, `missing` reports on **every** column, text included.

---

## `cv`

Coefficient of variation — relative variability (`std ÷ mean`), as a percentage. Handy for comparing spread across columns on different scales.

!!! tip "Similar concept"
    `scipy.stats.variation`

**Signature**

```python
cv(data, decimals=2)
```

**A single column returns a number**

```python
import pandas as pd
from percentify import cv

cv(pd.Series([50000, 52000, 48000, 90000, 51000]))
```

```text
30.65
```

**A DataFrame returns every numeric column, most variable first**

```python
df = pd.DataFrame({
    "salary": [50000, 52000, 48000, 90000, 51000],
    "age":    [25, 30, 35, 40, 45],
})

cv(df)
```

```text
  feature     cv
0  salary  30.65
1     age  22.59
```

---

## `outliers`

The percentage of values that are outliers by the IQR method (below `Q1 − 1.5·IQR` or above `Q3 + 1.5·IQR`).

!!! tip "Similar concept"
    `scipy.stats.iqr`

**Signature**

```python
outliers(data, decimals=2, multiplier=1.5)
```

**A single column returns a number**

```python
import pandas as pd
from percentify import outliers

outliers(pd.Series([10, 11, 12, 13, 14, 15, 200]))
```

```text
14.29
```

One value out of seven (`200`) is an outlier → 14.29%.

**A DataFrame returns every numeric column**

```python
df = pd.DataFrame({
    "salary": [10, 11, 12, 13, 14, 15, 200],
    "age":    [25, 26, 27, 28, 29, 30, 31],
})

outliers(df)
```

```text
  feature  outlier_pct
0  salary        14.29
1     age         0.00
```

Tune the sensitivity with `multiplier` (e.g. `multiplier=3.0` for a looser bound).

---

## `r_squared`

R-squared (coefficient of determination), as a percentage.

!!! tip "Similar concept"
    `sklearn.metrics.r2_score`

**Signature**

```python
r_squared(y_true, y_pred, decimals=2)
```

**Example**

```python
from percentify import r_squared

r_squared([1, 2, 3, 4, 5], [1.1, 1.9, 3.2, 3.8, 5.1])
```

```text
98.9
```

The predictions explain 98.9% of the variance in the true values. Accepts lists, numpy arrays, or pandas Series.

---

## `pca_variance`

The percentage of variance explained by each principal component, plus a running cumulative total.

!!! tip "Similar concept"
    `sklearn.decomposition.PCA` (`.explained_variance_ratio_`)

**Signature**

```python
pca_variance(df, decimals=2, n_components=None, standardize=True)
```

**Example**

```python
import numpy as np, pandas as pd
from percentify import pca_variance

np.random.seed(7)
base = np.random.randn(200)
df = pd.DataFrame({
    "height": base + np.random.randn(200) * 0.3,
    "weight": base + np.random.randn(200) * 0.3,   # shares a signal with height
    "noise":  np.random.randn(200),
})

pca_variance(df)
```

```text
  component  variance_explained  cumulative
0       PC1               64.04       64.04
1       PC2               33.34       97.38
2       PC3                2.62      100.00
```

Read the `cumulative` column to decide how many components to keep — here PC1 + PC2 already capture 97.4% of the variance.

!!! warning "Standardization matters"
    By default `standardize=True`, so each column is scaled to unit variance first. This stops a column measured in large units (e.g. dollars) from dominating purely because of its scale. Pass `standardize=False` for covariance-based PCA on the raw values.

---

## `imbalance`

Summarize class imbalance in a categorical target: the per-class breakdown plus the headline metrics you actually report.

!!! tip "Similar concept"
    `pandas.Series.value_counts`

**Signature**

```python
imbalance(data, decimals=2)
```

**Example**

```python
import pandas as pd
from percentify import imbalance

df = pd.DataFrame({"churn": ["No"] * 850 + ["Yes"] * 150})

result = imbalance(df["churn"])
result
```

```text
  class  count   pct
0    No    850  85.0
1   Yes    150  15.0
```

The headline metrics come attached on `.attrs["summary"]`, so the return stays a clean DataFrame:

```python
result.attrs["summary"]
```

```text
{'n_classes': 2,
 'majority_class': 'No',
 'minority_class': 'Yes',
 'imbalance_ratio': 5.67,
 'entropy_pct': 60.98}
```

- **`imbalance_ratio`**: the majority count divided by the minority count (`5.67` means "No" is 5.7x more common than "Yes").
- **`entropy_pct`**: `100` for a perfectly balanced target, approaching `0` as one class dominates.

Pass a single column (`df["target"]`), not the whole DataFrame. Nulls are dropped before counting.

---

## `difference`

Symmetric percentage difference between two values or two columns — how *far apart* they are, regardless of direction. (Reach for `change` when direction matters.)

!!! tip "Similar concept"
    `numpy` / `pandas` element-wise arithmetic

**Signature**

```python
difference(a, b, decimals=2)
```

**Two numbers**

```python
from percentify import difference

difference(10, 20)
```

```text
66.67
```

**Two columns** (element-wise) — the average of the two values is the denominator, so `difference(a, b) == difference(b, a)`:

```python
import pandas as pd

sensors = pd.DataFrame({
    "sensor_a": [10.0, 50.0, 100.0],
    "sensor_b": [12.0, 50.0, 130.0],
})

sensors["pct_gap"] = difference(sensors["sensor_a"], sensors["sensor_b"])
sensors
```

```text
   sensor_a  sensor_b  pct_gap
0      10.0      12.0    18.18
1      50.0      50.0     0.00
2     100.0     130.0    26.09
```

---

## `split`

Distribute a total across weights, proportionally — allocation for budgets, quotas, or apportioning a sum by group.

!!! tip "Similar concept"
    `numpy` / `pandas` weighted arithmetic

**Signature**

```python
split(total, weights, decimals=2)
```

**A list of weights returns a list**

```python
from percentify import split

split(10000, [2, 3, 5])
```

```text
[2000.0, 3000.0, 5000.0]
```

**A column of weights returns an aligned Series** — allocate a budget by population:

```python
import pandas as pd

budget = pd.DataFrame({
    "region": ["North", "South", "East"],
    "population": [200, 300, 500],
})

budget["budget"] = split(10000, budget["population"])
budget
```

```text
  region  population  budget
0  North         200  2000.0
1  South         300  3000.0
2   East         500  5000.0
```

Raises `ValueError` if `weights` is empty or sums to zero.

---

## `display`

Format a number — or a whole column — as clean percentage strings. The "last mile" for reports, dashboards, and exports.

!!! tip "Similar concept"
    `pandas.Series.map` + Python string formatting

**Signature**

```python
display(value, decimals=2, suffix="%", multiply=False)
```

**A single number returns a string**

```python
from percentify import display

display(0.45, multiply=True)
```

```text
'45.0%'
```

**A column returns a Series of strings** — turn ratios into report-ready text:

```python
import pandas as pd

rates = pd.DataFrame({
    "campaign": ["A", "B", "C"],
    "conv_rate": [0.045, 0.12, 0.083],
})

rates["display"] = display(rates["conv_rate"], multiply=True)
rates
```

```text
  campaign  conv_rate display
0        A      0.045    4.5%
1        B      0.120   12.0%
2        C      0.083    8.3%
```

Use `multiply=True` when your values are ratios (`0.45` → `"45.0%"`); leave it off when they're already percentages (`45` → `"45.0%"`). Customize the trailing text with `suffix`.

---

## Warnings and non-numeric data

Percentify never hands you a raw traceback for a predictable mistake. Give a function data it can't use and it raises a `PercentifyWarning` and returns an empty result:

```python
import pandas as pd
from percentify import vif

vif(pd.DataFrame({"city": ["NY", "LA"], "team": ["A", "B"]}))
```

```text
PercentifyWarning: Numeric columns required: no numeric columns found. VIF
measures multicollinearity between numeric features - encode any
categorical/text columns first.
```

The warning is a normal Python warning, so you can catch or silence it:

```python
import warnings
from percentify import PercentifyWarning

warnings.filterwarnings("ignore", category=PercentifyWarning)
```

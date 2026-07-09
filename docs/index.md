# Percentify

**A niche data science library for practitioners and learners alike** — drawing its main dependencies from pandas and numpy, and including everyday statistics.

Following the **Pareto principle**, Percentify brings the 20% of operations that make up 80% of daily data work to the forefront, each as a single, readable function call. No more digging through six-line recipes and hard-to-remember import paths for the checks you run on every dataset.

Percentify **does not aim to compete** with pandas, scipy, statsmodels, or scikit-learn — it stands on their shoulders and works alongside them. Every function names the underlying library it draws from, so the moment you need the full, configurable version, you know exactly where to go.

---

## Install

```bash
pip install percentify
```

Requires `numpy` and `pandas`.

## A quick taste

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

One import, one line — a clean, sorted DataFrame you can read or feed straight into the next step.

[Read the full documentation →](documentation.md){ .md-button .md-button--primary }

---

## What's inside

| Function | What it answers |
|---|---|
| `change` | How much did a value grow — as numbers, columns, or a whole series? |
| `vif` | Which features are collinear? |
| `missing` | How much of each column is missing? |
| `cv` | How variable is each column, relative to its mean? |
| `outliers` | What percentage of each column are outliers? |
| `r_squared` | How well do predictions fit? |
| `pca_variance` | How much variance does each principal component explain? |

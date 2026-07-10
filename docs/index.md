<p align="center">
  <img src="assets/logo.png" alt="Percentify" width="480">
</p>

# Percentify

*The 20% of data science operations behind 80% of daily work: each a single, readable function call.*

Following the *Pareto principle*, Percentify brings the checks you run on every dataset to the forefront, one call at a time. No more digging through six-line recipes and hard-to-remember import paths.

Built on pandas and numpy, it pairs the everyday tools you reach for constantly with lesser-known ones worth knowing. Where a function wraps an existing library (pandas, scipy, statsmodels, scikit-learn), it names it, so you always know where to dig deeper.

---

## Install

bash
pip install percentify


Requires numpy and pandas.

## A quick taste

python
import pandas as pd
from percentify import missing

df = pd.DataFrame({
    "salary": [50000, None, 60000, None],
    "age":    [25, 30, None, 40],
    "city":   ["NY", "LA", "SF", "LA"],
})

missing(df)


text
   column  missing_pct
0  salary         50.0
1     age         25.0
2    city          0.0


One import, one line. A clean, sorted DataFrame you can read or feed straight into the next step.

[Read the full documentation →](documentation.md){ .md-button .md-button--primary }

---

## What's inside

| Function | What it answers |
|---|---|
| change | How much did a value grow (as numbers, columns, or a whole series)? |
| vif | Which features are collinear? |
| missing | How much of each column is missing? |
| cv | How variable is each column, relative to its mean? |
| outliers | What percentage of each column are outliers? |
| r_squared | How well do predictions fit? |
| pca_variance | How much variance does each principal component explain? |
| difference | How far apart are two values or columns, regardless of direction? |
| split | How does a total divide across weights or groups? |
| display | Format numbers or a column as clean "%" strings for reports. |

# Texas Residential Real Estate Market Analysis (2026)

An exploratory data analysis (EDA) and machine learning study of the residential real estate market in Texas, based on a corpus of 12,137 active MLS listings. The project quantifies the structural characteristics of the Texas housing stock and evaluates the predictive power of structural features (square footage, room counts, garage capacity, construction year) on listing price using linear, tree-based, and ensemble regression models.

## 1. Overview

Residential property valuation is a canonical regression problem in applied data science: price is a function of a mix of physical attributes (size, room count, age) and location-driven signals that are not always captured in tabular data. This project has two goals:

1. **Descriptive** — characterize the distribution of property types, prices, sizes, and physical attributes across the Texas market.
2. **Predictive** — test how well listing price can be recovered from structural features alone (i.e., without location, text, or image data), using Linear Regression, a single Decision Tree, and a Random Forest ensemble as a progression of model complexity.

## 2. Dataset

- **Source:** [Texas Residential Real Estate Intelligence 2026](https://www.kaggle.com/datasets/jahnavikachhia23/texas-residential-real-estate-intelligence-2026) (Kaggle, jahnavikachhia23)
- **Size:** 12,137 listings, 13 raw columns (property type, subtype, free-text description, price, square footage, stories, beds, baths, full baths, calculated full baths, garage capacity, year built, price per square foot)
- **Target variable:** `listPrice` (active MLS listing price, USD)

## 3. Repository Structure

```
.
├── data_analysis.py       # Data loading, cleaning, EDA, feature engineering
│── machine_learning.py    # Preprocessing, model training, evaluation
├── eda/                   # Exploratory data analysis figures
│── ml/                    # Model diagnostic plots
├── datasets/              # (gitignored) raw and processed CSV files
└── README.md
```

## 4. Methodology

### 4.1 Data Cleaning

Initial inspection revealed substantial missingness in several columns, most notably `sub_type` (10,388 / 12,137 missing, ~86%) and `year_built` (2,288 missing, ~19%). The cleaning pipeline:

- Dropped `sub_type`, `text` (free-form description), and the derived `Price_Per_SqFt` column (redundant with `listPrice` / `sqft`, and a source of target leakage).
- Removed 63 exact duplicate rows.
- Row-wise dropped listings missing `listPrice`, `sqft`, `stories`, `beds`, `baths`, `baths_full`, or `baths_full_calc` (the "core" structural fields).
- Median-imputed `garage` and `year_built`, since missingness in these fields is plausibly non-random (e.g., listings without a garage) and median imputation is more robust to the strong right-skew observed in `listPrice` and `sqft`.

### 4.2 Feature Engineering

Two derived features were added on top of the raw MLS fields:

- `house_age = 2026 - year_built`
- `baths_per_bedroom = baths / beds` — a proxy for interior "quality density" that is independent of raw house size.

Property `type` was one-hot encoded for downstream modeling.

### 4.3 Modeling

Three regressors of increasing complexity were trained on the structural feature set (`sqft`, `stories`, `beds`, `baths`, `baths_full`, `baths_full_calc`, `garage`, `year_built`, `house_age`, `baths_per_bedroom`), all features standardized with `StandardScaler`:

| Model | Notes |
|---|---|
| Linear Regression | Baseline linear model |
| Decision Tree Regressor | Single unconstrained tree (no depth limit) |
| Random Forest Regressor | 200 trees, `max_depth=10`, `min_samples_split=5` |

For the modeling stage, the dataset was additionally filtered to a more homogeneous segment — single-family homes under 5,000 sqft, built in 2023, 2024, or 2026, with a 2- or 3-car garage — to reduce heterogeneity in the price-generating process before fitting. Models were evaluated on a held-out 10% test split (`random_state=42`).

## 5. Results

### 5.1 Market Composition

Single-family homes dominate the sample (11,188 of 12,137 listings, ~92%), followed by condos (340), multi-family properties (302), and townhomes (293); apartments and coop-style listings are negligible (14 combined). This composition means the dataset is, in practice, primarily a single-family-home price model rather than a balanced cross-segment view of the Texas market.

![Primary Property Types](data%20analysis%20plots/analysis_of_primary_property_types_of_houses_in_Texas.png)

### 5.2 Price Distribution

Listing prices are heavily right-skewed: mean price is **$499,879**, but the median is **$374,900** — a gap of roughly $125,000 driven by a long tail of high-end listings (max: $11,995,000; std: $540,944). The 25th–75th percentile range ($288,092–$539,000) is a more representative band for the "typical" Texas listing in this sample than the mean.

![Price Distribution](data%20analysis%20plots/analysis_of_prices_of_houses_in_Texas.png)

### 5.3 Size and Structural Attributes

Interior square footage is likewise right-skewed (mean 2,286 sqft, median 2,085 sqft, max 19,600 sqft), with the bulk of listings concentrated between roughly 1,600 and 2,700 sqft.

![Square Footage Distribution](data%20analysis%20plots/analysis_of_total_interior_square_footage_of_houses_in_Texas.png)

Bedroom and bathroom counts cluster tightly around conventional single-family configurations — most listings report 3–4 bedrooms and 2–3 bathrooms — while garage capacity is dominated by 2-car garages.

![Bedroom Count Distribution](data%20analysis%20plots/analysis_of_total_bedroom_of_houses_in_Texas.png)
![Bathroom Count Distribution](data%20analysis%20plots/analysis_of_total_bathroom_of_houses_in_Texas.png)

### 5.4 Correlation Structure

![Price Distribution](data%20analysis%20plots/correlation_heatmap_analysis.png)
A Pearson correlation analysis of the numeric fields shows that `listPrice` correlates most strongly with `sqft` (r = 0.627), followed by `baths` (r = 0.505), `baths_full` (r = 0.476), and `garage` (r = 0.393). Notably, `year_built` is essentially uncorrelated with price (r = 0.036), suggesting that, in this dataset, construction age has little direct linear relationship with list price once the effect of size is not controlled for — a signal that Texas price levels are driven substantially more by scale than by vintage. `beds` and `baths_full`/`baths_full_calc` are highly collinear with each other (r > 0.88), as expected.

![Sqft vs Price](data%20analysis%20plots/total_interior_square_footage_vs_Price.png)

The `sqft` vs. `listPrice` scatter plot shows a positive but non-linear, heteroscedastic relationship: price variance increases sharply as square footage grows, consistent with the right-skewed price distribution and the presence of luxury outliers.

### 5.5 Predictive Modeling

| Model | R² (test) | MAE (test) |
|---|---|---|
| Linear Regression | 0.656 | — |
| Decision Tree Regressor | 0.162 | $94,164.82 |
| **Random Forest Regressor** | **0.717** | — |

The Random Forest ensemble achieved the best fit (**R² = 0.717**), outperforming both the linear baseline (R² = 0.656) and the single, unconstrained Decision Tree (R² = 0.162). The large gap between the single tree and the forest is a textbook illustration of variance reduction through bagging: an unconstrained decision tree overfits the training data and generalizes poorly, while averaging over 200 depth-limited trees substantially improves generalization on unseen listings. The fact that Random Forest only moderately outperforms plain Linear Regression (+0.06 R²) suggests that most of the predictive signal in the structural feature set is close to linear (dominated by `sqft`), with non-linear interactions (e.g., between size, baths, and garage) contributing a secondary, but non-trivial, improvement.

![Random Forest: Predicted vs Actual](ML%20plots/random_forest_plot.png)

The predicted-vs-actual scatter plot for the Random Forest model shows tight clustering along the diagonal for mid-range prices, with increasing dispersion at the high end — the model underfits luxury properties, which is consistent with these being driven by unobserved factors (location, finishes, lot size) not present in this structural feature set.

### 5.6 Interpretation

Taken together, these results indicate that:

- **Size is the dominant structural price driver** in this dataset — no other single structural variable comes close to `sqft`'s correlation with price.
- **Construction year carries little standalone predictive value**, at least linearly, which may reflect the fact that much of the sampled inventory is relatively recently built or renovated (median `year_built` in the raw data is 2023).
- **Roughly 28–30% of price variance is left unexplained** by the best model (Random Forest, R² = 0.717), which is expected given that the feature set omits the two variables classically dominant in real estate pricing: **location** (ZIP code / neighborhood / school district) and **lot size / land value**. This is the natural ceiling for a purely structural feature set and is consistent with results reported in comparable tabular real-estate regression studies.

## 6. Reproducing the Analysis

```bash
# clone the repo
git clone <your-repo-url>
cd <repo-name>

# install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn

# 1. place the raw Kaggle CSV at:
#    datasets/Texas_Residential_Real Estate_Intelligenc_ 2026.csv

# 2. run the EDA / cleaning / feature engineering pipeline
python src/data_analysis.py     # outputs plots/eda/*.png and datasets/ml_df.csv

# 3. run the modeling pipeline
python src/machine_learning.py  # outputs plots/ml/*.png and prints R² / MAE to stdout
```

## 7. Data Availability

The raw dataset is **not included** in this repository. It is distributed by its original author on Kaggle: [Texas Residential Real Estate Intelligence 2026](https://www.kaggle.com/datasets/jahnavikachhia23/texas-residential-real-estate-intelligence-2026). Kaggle datasets carry their own, dataset-specific license set by the uploader (visible on the dataset page under "Usage"); check that page before redistributing the raw or processed CSV files.

## 8. Citation

If you build on this analysis, please cite the original dataset:

> jahnavikachhia23, *Texas Residential Real Estate Intelligence 2026*, Kaggle. https://www.kaggle.com/datasets/jahnavikachhia23/texas-residential-real-estate-intelligence-2026

## 9. Limitations

- The dataset excludes location (ZIP code, coordinates, neighborhood) and lot size, both of which are typically primary drivers of real estate price — the ~0.72 R² ceiling reflects this.
- The listing-price target is the **active list price**, not the closing/sale price, so it reflects seller pricing strategy rather than realized market-clearing value.
- The modeling subset (single-family, <5,000 sqft, `year_built` ∈ {2023, 2024, 2026}, 2–3 car garage) is a deliberately narrowed slice of the full dataset chosen for homogeneity; reported model metrics should not be read as representative of the full 12,137-listing sample.
- No hyperparameter search (e.g., grid/random search or cross-validation) was performed; Random Forest and Decision Tree parameters were set manually.

## 10. Tech Stack

`Python` · `pandas` · `numpy` · `matplotlib` · `seaborn` · `scikit-learn`

# Medical Insurance Cost Prediction

A machine learning project that predicts medical insurance charges
based on personal and lifestyle factors, with a focus on feature
engineering, categorical encoding, outlier analysis, and model
comparison.

## Dataset

The Medical Cost Personal Dataset from Kaggle, containing 1,338
samples and 7 features.

| Feature | Type | Description |
|---|---|---|
| `age` | Numerical | Age of the insured person |
| `sex` | Categorical | Gender (male/female) |
| `bmi` | Numerical | Body Mass Index |
| `children` | Numerical | Number of dependent children |
| `smoker` | Categorical | Smoking status (yes/no) |
| `region` | Categorical | Residential region in the US |
| `charges` | **Target** | Medical costs billed by insurance |

## Project Structure
├── Medical_Cost.ipynb   # Main notebook
└── README.md

## Workflow

1. Data loading and exploration
2. EDA — distributions, scatter plots, boxplots, count plots
3. Data cleaning — duplicate removal
4. Outlier analysis
5. Categorical encoding
6. Correlation matrix analysis
7. Model training and evaluation without feature engineering
8. Feature engineering
9. Model training and evaluation after feature engineering
10. Log transformation experiment on target variable
11. Cross-validation for all experiments

## Data Cleaning

- Found and removed **1 duplicate row** → final dataset: 1,337 rows
- No missing values in any column

## Outlier Analysis

Outlier detection using IQR method revealed:
- `age`: 0 outliers
- `bmi`: 9 outliers
- `children`: 0 outliers
- `charges`: 139 outliers

**Decision: kept all outliers.** The 9 BMI outliers represent
individuals with genuinely high BMI — a real medical condition,
not a data entry error. The 139 charge outliers represent real
high-cost patients (smokers, obese individuals) whose patterns
are essential for the model to learn. Removing them would
hurt model performance.

An additional scatter plot of Age vs BMI confirmed there is no
relationship between the two (correlation = 0.04), ruling out
age as a cause of the BMI outliers.

## Categorical Encoding

- `sex` and `smoker` → binary encoding using `map()`
  (male=0/female=1, no=0/yes=1)
- `region` → one-hot encoding using `get_dummies()` with
  `drop_first=True` to avoid multicollinearity

## Key EDA Findings

- `smoker` is the strongest predictor of charges with a
  correlation of **0.79**
- `age` vs `charges` scatter plot revealed **3 distinct bands**
  — caused by the smoker variable splitting the population
- `bmi` vs `charges` scatter plot revealed **2 clusters**
  — smokers with high BMI form a separate high-cost cluster
- `charges` is right-skewed — most people have low costs
  while a small group has very high costs
- `children` shows no meaningful pattern with charges
- Age and children have no relationship (correlation = 0.04)

## Feature Engineering

Based on EDA findings, the following interaction features
were created:

| Feature | Formula | Reason |
|---|---|---|
| `smoker_bmi` | smoker × bmi | Captures combined smoking + BMI effect |
| `bmi_category` | 1 if bmi ≥ 30, else 0 | Binary obesity flag |
| `obese_smoker` | bmi_category × smoker | Obese smokers charged dramatically more |

**Features tested and dropped:**
- `age_smoker` (age × smoker) — highly correlated with
  `smoker_bmi` (0.91), caused multicollinearity with no
  improvement in R²
- `age_children` (age × children) — no relationship between
  age and children confirmed (correlation = 0.04), no impact
  on results

## Results

### Before Feature Engineering

| Metric | LinearRegression | SGDRegressor |
|---|---|---|
| R² | 0.8069 | 0.8076 |
| MAE | 4177.05 | 4177.21 |
| RMSE | 5956.34 | 5946.47 |
| CV Mean R² | 0.7258 | 0.7257 |
| CV Std | 0.0253 | 0.0249 |

### After Feature Engineering

| Metric | LinearRegression | SGDRegressor |
|---|---|---|
| R² | 0.9092 | 0.9094 |
| MAE | 2276.45 | 2287.81 |
| RMSE | 4085.44 | 4080.97 |
| CV Mean R² | 0.8511 | 0.8506 |
| CV Std | 0.0246 | 0.0248 |

Feature engineering improved R² by **~10%** and reduced MAE
by nearly **$1,900 per prediction**.

## Experiments

**LinearRegression vs SGDRegressor** — both models produced
nearly identical results across all experiments. LinearRegression
was selected as the final model for simplicity and interpretability.

**Log transformation on charges** — applied `log1p` to the
target variable and reversed with `expm1` after prediction.
This resulted in R² = 0.6567, significantly worse than 0.9092
without transformation. The reason is that the feature
engineering already captured the non-linearity in the data,
making the transformation unnecessary. Additionally, errors
in log space amplify unevenly when converted back to dollar
values, hurting high-charge predictions the most.

**smoker_bmi only vs age_smoker only vs both** — `smoker_bmi`
alone gave R² = 0.88, `age_smoker` alone gave R² = 0.80,
and both together gave R² = 0.88 with no improvement.
This confirmed multicollinearity between the two features,
so `age_smoker` was dropped.

## Why Cross-Validation Mattered

Without cross-validation, the test R² before feature engineering
looked strong at **0.8069**, which could give the impression that
the model was already performing well. Cross-validation revealed
the truth — the real generalization performance was only **0.7258**,
meaning the single test split gave an overly optimistic result
by pure luck of how the data was divided.

This 8% gap between test R² and CV Mean R² was the key signal
that feature engineering was necessary. After feature engineering,
the gap closed significantly — test R² of 0.9092 vs CV Mean R²
of 0.8511 — confirming that the model genuinely improved and
was not just getting lucky on a favorable split.

This is exactly why cross-validation should always be used
alongside a single train/test split — it gives you the honest
picture of your model's real performance.


## Key Findings

- `smoker` is by far the most important feature — smoking
  status alone accounts for most of the variance in charges
- The engineered feature `smoker_bmi` became the strongest
  predictor after feature engineering
- An obese smoker is charged dramatically more than either
  an obese non-smoker or a thin smoker — the interaction
  effect is non-linear and cannot be captured without
  feature engineering
- Cross-validation std of ~0.025 is stable and confirms
  no overfitting

## Technologies

- Python 3
- pandas, numpy
- scikit-learn
- matplotlib, seaborn

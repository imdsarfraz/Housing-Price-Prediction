# House Price Prediction – King County, USA

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/imdsarfraz/Housing-Price-Prediction/blob/main/Housing_Price.ipynb)

Predicting house sale prices in King County, Washington (the Seattle area) with **linear**, **polynomial** and **Ridge** regression, using pandas, seaborn and scikit-learn.

The notebook walks through a complete regression workflow: data cleaning → exploratory analysis → model building → evaluation on a held-out test set → regularization.

## Results

All three models use the same 11 features and the same 80/20 train/test split (`random_state=42`).

| Model | Train R² | Test R² | Test RMSE (USD) |
|---|:---:|:---:|:---:|
| Multiple linear regression | 0.656 | 0.661 | 226,248 |
| **Polynomial regression pipeline (degree 2)** | 0.757 | **0.711** | **208,962** |
| Ridge regression (α = 0.1) | 0.656 | 0.661 | – |

- The polynomial pipeline (`StandardScaler → PolynomialFeatures → LinearRegression`) performs best: it explains **~71% of the price variation on unseen houses** and has a **~7.6% lower RMSE** than plain linear regression.
- Ridge regularization doesn't change the linear model's scores, because that model doesn't overfit (train R² ≈ test R²).

**Features used:** `floors`, `waterfront`, `lat`, `bedrooms`, `sqft_basement`, `view`, `bathrooms`, `sqft_living15`, `sqft_above`, `grade`, `sqft_living`

> Scores were produced with scikit-learn 1.9. With older scikit-learn versions the polynomial model's scores can differ slightly (third decimal place): some features are perfectly collinear (see below), which makes its least-squares problem rank-deficient.

## Key insights

- `sqft_living` (r = 0.70), `grade` (r = 0.67) and `sqft_above` (r = 0.61) are the features most correlated with price; `sqft_living` alone explains ~49% of the price variation.
- Waterfront homes are rare (163 of 21,613) but sell for a median of **~$1.4M**, vs. **~$450K** for all other homes.
- Location matters along the north–south axis (`lat`, r = 0.31) far more than east–west (`long`, r = 0.02).
- `sqft_living` is exactly `sqft_above + sqft_basement` for every house, so these three features are perfectly collinear.
- Prices are right-skewed (median $450K, max $7.7M) and spread out more for bigger houses.

## Dataset

[`housing.csv`](housing.csv) is the [House Sales in King County, USA](https://www.kaggle.com/datasets/harlfoxem/housesalesprediction) dataset: **21,613 sales** between **May 2014 and May 2015**, with prices from $75,000 to $7.7M (median $450,000).

| Column | Description |
|---|---|
| `id` | Unique ID of the sale |
| `date` | Date of the sale |
| `price` | Sale price in USD — **the target variable** |
| `bedrooms` | Number of bedrooms |
| `bathrooms` | Number of bathrooms (0.5 = a room with a toilet but no shower) |
| `sqft_living` | Interior living space (sq ft) |
| `sqft_lot` | Size of the land lot (sq ft) |
| `floors` | Number of floors |
| `waterfront` | 1 if the house overlooks the waterfront, otherwise 0 |
| `view` | How good the view is (0–4) |
| `condition` | Overall condition (1–5) |
| `grade` | Construction and design quality (1–13; 7 = average) |
| `sqft_above` | Interior space above ground level (sq ft) |
| `sqft_basement` | Interior space below ground level (sq ft) |
| `yr_built` | Year built |
| `yr_renovated` | Year of the last renovation (0 = never renovated) |
| `zipcode` | ZIP code (70 different ones) |
| `lat`, `long` | Latitude and longitude |
| `sqft_living15` | Interior living space of the 15 nearest neighbors (sq ft) |
| `sqft_lot15` | Lot size of the 15 nearest neighbors (sq ft) |

The file also has an unnamed row-index column, and 13 `bedrooms` and 10 `bathrooms` values are missing. The notebook drops the index and `id` columns and fills the missing values with the column mean.

## Notebook walkthrough

[`Housing_Price.ipynb`](Housing_Price.ipynb) is organized into these sections:

1. **Load the data** – read the CSV straight from this GitHub repo with pandas.
2. **Explore** – data types, summary statistics, missing-value counts.
3. **Clean** – drop `id` and the index column, fill missing `bedrooms`/`bathrooms` with the mean.
4. **Exploratory analysis** – floor counts, waterfront box plot, `sqft_living` vs. price regression plot, correlation of every numeric feature with price.
5. **Model 1: Linear regression** – one feature (`long`, `sqft_living`), then 11 features.
6. **Model 2: Polynomial regression** – `Pipeline(StandardScaler → PolynomialFeatures → LinearRegression)`.
7. **Train/test evaluation** – 80/20 split, R² and RMSE for both models.
8. **Ridge regression** – Linear vs. Ridge comparison and the effect of the regularization strength α.
9. **Results summary**

## How to run

**In Google Colab (no setup):** click the *Open in Colab* badge at the top, then *Runtime → Run all*. The notebook downloads the dataset from this repo.

**Locally** (Python 3.9+):

```bash
git clone https://github.com/imdsarfraz/Housing-Price-Prediction.git
cd Housing-Price-Prediction
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook Housing_Price.ipynb
```

The notebook reads `housing.csv` from GitHub, so it needs an internet connection. To work offline, change the loading cell to `pd.read_csv("housing.csv")`.

## Repository structure

```
Housing-Price-Prediction/
├── Housing_Price.ipynb   # analysis and models
├── housing.csv           # dataset (21,613 house sales)
├── requirements.txt      # Python dependencies
└── README.md
```

## Possible next steps

- Predict `log(price)` instead of `price`: prices are right-skewed and the errors grow with house size.
- Use more of the data: `zipcode` (one-hot encoded), `yr_built`, `yr_renovated` and the sale date.
- Drop one of `sqft_living` / `sqft_above` / `sqft_basement` to remove the perfect collinearity and make the coefficients interpretable.
- Tune the polynomial degree and the Ridge α with cross-validation (`GridSearchCV`, `RidgeCV`) instead of a single train/test split.
- Try tree-based models such as Random Forest or Gradient Boosting.

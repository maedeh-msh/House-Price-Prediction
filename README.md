# House Price Prediction

A machine learning project that predicts residential house sale prices based on property
characteristics such as size, quality, location, and age. Built with scikit-learn on the
classic Ames Housing dataset (Kaggle "House Prices - Advanced Regression Techniques").

## Problem

The goal is to predict `SalePrice` for each house in the test set. Model performance is
evaluated using Root Mean Squared Error (RMSE) between the logarithm of predicted and
actual sale prices — a lower score is better.

## Project Structure

```
.
├── data/
│   ├── train.csv
│   └── test.csv
├── house-price-prediction.ipynb   # Main notebook: EDA, preprocessing, modeling
├── submission.csv                 # Final predictions
├── requirements.txt
└── README.md
```

## Approach

1. **Exploratory Data Analysis** — inspect shape, types, missing values, target
   distribution, and correlations.
2. **Preprocessing** — a `ColumnTransformer` pipeline that:
   - Imputes and scales numerical features (median imputation + `StandardScaler`)
   - Handles `GarageYrBlt` missing values separately (constant fill)
   - One-hot encodes categorical features, distinguishing columns where a missing
     value genuinely means "None" from columns with a few truly missing entries
3. **Model Selection** — compares several regressors with 5-fold cross-validation:
   - Linear Regression (baseline)
   - Random Forest
   - Ridge Regression
   - Extra Trees
   - Gradient Boosting
   - Voting Regressor (RF + GB)
   - Stacking Regressor (RF + GB → Linear Regression)
4. **Hyperparameter Tuning** — `RandomizedSearchCV` for Random Forest and Gradient
   Boosting.
5. **Target Transformation** — training on `log1p(SalePrice)` and inverting predictions
   with `expm1` to better satisfy the competition's log-RMSE metric.
6. **Final Model** — a tuned Stacking Regressor trained on the log-transformed target,
   used to generate `submission.csv`.

## Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/maedeh-msh/House-Price-Prediction.git
cd House-Price-Prediction
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Add the data
Download `train.csv` and `test.csv` from the
[Kaggle competition page](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data)
and place them in a `data/` folder at the project root.

### 4. Run the notebook
```bash
jupyter notebook house-price-prediction.ipynb
```

## Output

Running the notebook end-to-end produces `submission.csv`, containing predicted
`SalePrice` values for each `Id` in the test set, ready for submission to Kaggle.

## Results

Submitted to the [Kaggle "House Prices - Advanced Regression Techniques"](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques) competition:

| Metric | Score |
|---|---|
| Public Leaderboard RMSE (log scale) | **0.21138** |

## License

This project is open source and available under the MIT License.

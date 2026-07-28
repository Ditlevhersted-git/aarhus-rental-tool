import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import LinearRegression
import xgboost as xgb

df = pd.read_csv("data/gold_listings_modelling.csv")
feature_cols = [c for c in df.columns if c not in
                ["rent_kr", "listed_date", "kilde", "price_per_m2", "floor", "housing_type"]]
X = df[feature_cols]
y = df["rent_kr"]

kf = KFold(n_splits=5, shuffle=True, random_state=42)

lr = LinearRegression()
xgb_model = xgb.XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.05,
                              subsample=0.8, colsample_bytree=0.8, random_state=42)

for name, model in [("Linear Regression", lr), ("XGBoost", xgb_model)]:
    rmse_scores = -cross_val_score(model, X, y, cv=kf, scoring="neg_root_mean_squared_error")
    r2_scores = cross_val_score(model, X, y, cv=kf, scoring="r2")
    print(f"{name}:")
    print(f"  RMSE per fold: {np.round(rmse_scores,0)}")
    print(f"  Mean RMSE: {rmse_scores.mean():.0f} (+/- {rmse_scores.std():.0f})")
    print(f"  Mean R2: {r2_scores.mean():.3f} (+/- {r2_scores.std():.3f})")
    print()

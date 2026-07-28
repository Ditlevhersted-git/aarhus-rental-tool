import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb

df = pd.read_csv("data/gold_listings_modelling.csv")

feature_cols = [c for c in df.columns if c not in
                ["rent_kr", "listed_date", "kilde", "price_per_m2", "floor", "housing_type"]]
X = df[feature_cols]
y = df["rent_kr"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lr = LinearRegression()
lr.fit(X_train, y_train)
pred_lr = lr.predict(X_test)
rmse_lr = np.sqrt(mean_squared_error(y_test, pred_lr))
mae_lr = mean_absolute_error(y_test, pred_lr)
r2_lr = r2_score(y_test, pred_lr)

xgb_model = xgb.XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.05,
                              subsample=0.8, colsample_bytree=0.8, random_state=42)
xgb_model.fit(X_train, y_train)
pred_xgb = xgb_model.predict(X_test)
rmse_xgb = np.sqrt(mean_squared_error(y_test, pred_xgb))
mae_xgb = mean_absolute_error(y_test, pred_xgb)
r2_xgb = r2_score(y_test, pred_xgb)

print("=== Model comparison (test set, n=%d, full gold dataset n=%d) ===" % (len(y_test), len(df)))
print(f"{'Model':<20}{'RMSE':>10}{'MAE':>10}{'R2':>8}")
print(f"{'Linear Regression':<20}{rmse_lr:>10.0f}{mae_lr:>10.0f}{r2_lr:>8.3f}")
print(f"{'XGBoost':<20}{rmse_xgb:>10.0f}{mae_xgb:>10.0f}{r2_xgb:>8.3f}")

print("\n=== XGBoost feature importance (top 10) ===")
importances = pd.Series(xgb_model.feature_importances_, index=feature_cols).sort_values(ascending=False)
print(importances.head(10))

results = pd.DataFrame({"Model": ["Linear Regression", "XGBoost"],
                         "RMSE": [rmse_lr, rmse_xgb], "MAE": [mae_lr, mae_xgb], "R2": [r2_lr, r2_xgb]})
results.to_csv("data/model_results.csv", index=False)
importances.to_csv("data/feature_importance.csv")

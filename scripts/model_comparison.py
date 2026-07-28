import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.dummy import DummyRegressor
import xgboost as xgb

df = pd.read_csv("data/gold_listings_modelling.csv")
feature_cols = [c for c in df.columns if c not in
                ["rent_kr", "listed_date", "kilde", "price_per_m2", "floor", "housing_type"]]
X = df[feature_cols]
y = df["rent_kr"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# KNN needs scaled features (distance-based)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

results = []

def evaluate(name, pred):
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    mae = mean_absolute_error(y_test, pred)
    r2 = r2_score(y_test, pred)
    results.append({"Model": name, "RMSE": rmse, "MAE": mae, "R2": r2})

# 1. Mean benchmark
dummy = DummyRegressor(strategy="mean")
dummy.fit(X_train, y_train)
evaluate("Mean benchmark", dummy.predict(X_test))

# 2. Linear regression
lr = LinearRegression()
lr.fit(X_train, y_train)
evaluate("Linear Regression", lr.predict(X_test))

# 3. Decision tree
dt = DecisionTreeRegressor(max_depth=6, random_state=42)
dt.fit(X_train, y_train)
evaluate("Decision Tree", dt.predict(X_test))

# 4. Random forest
rf = RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42)
rf.fit(X_train, y_train)
evaluate("Random Forest", rf.predict(X_test))

# 5. KNN (scaled)
knn = KNeighborsRegressor(n_neighbors=10)
knn.fit(X_train_scaled, y_train)
evaluate("K-Nearest Neighbours", knn.predict(X_test_scaled))

# 6. XGBoost
xgb_model = xgb.XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.05,
                              subsample=0.8, colsample_bytree=0.8, random_state=42)
xgb_model.fit(X_train, y_train)
evaluate("XGBoost", xgb_model.predict(X_test))

results_df = pd.DataFrame(results).sort_values("RMSE")
results_df[["RMSE","MAE","R2"]] = results_df[["RMSE","MAE","R2"]].round(3)
print(results_df.to_string(index=False))
results_df.to_csv("data/model_comparison_6.csv", index=False)

import pandas as pd
import numpy as np

silver = pd.read_csv("data/silver_listings_clean.csv")
context = pd.read_csv("data/dst_omraadestatistik.csv")

# --- Join area-level context data (population, student share, distance to centre) ---
gold = silver.merge(context, left_on="area", right_on="bydel", how="left").drop(columns=["bydel"])

# --- Feature engineering ---
gold["price_per_m2"] = (gold["rent_kr"] / gold["size_m2"]).round(1)
gold["listed_date"] = pd.to_datetime(gold["listed_date"])
gold["listed_month"] = gold["listed_date"].dt.month
gold["is_almen"] = (gold["housing_type"] == "almen").astype(int)

gold.to_csv("data/gold_listings.csv", index=False)

# Modelling-ready version (encoded)
gold_m = pd.get_dummies(gold, columns=["area"], prefix="area", drop_first=True)
floor_map = {"stuen": 0, "1": 1, "2": 2, "3": 3, "4+": 4, "ukendt": -1}
gold_m["floor_num"] = gold["floor"].map(floor_map)
gold_m.to_csv("data/gold_listings_modelling.csv", index=False)

print("=== Gold layer ===")
print("Shape:", gold.shape)
print("\nColumns:", gold.columns.tolist())
print("\nSample:")
print(gold[["size_m2","rooms","area","housing_type","rent_kr","price_per_m2",
            "befolkningstal","andel_studerende_pct","afstand_til_centrum_km"]].head())
print("\nMissing values after join:\n", gold.isna().sum())

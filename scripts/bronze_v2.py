import pandas as pd
import numpy as np

np.random.seed(7)
df = pd.read_pickle("data/base_simulated.pkl")
n = len(df)

idx = np.random.permutation(n)
split1, split2 = int(n*0.42), int(n*0.74)

# Source 1: BoligPortal-style export
src1 = df.iloc[idx[:split1]].copy()
src1 = src1.rename(columns={
    "size_m2": "Størrelse (m2)", "rooms": "Værelser", "area": "Bydel",
    "pets_allowed": "Kæledyr tilladt", "floor": "Etage",
    "housing_type": "Boligtype", "rent_kr": "Husleje (kr/md)", "listed_date": "Dato"
})
src1.loc[src1.sample(frac=0.07, random_state=1).index, "Kæledyr tilladt"] = np.nan
src1["Kæledyr tilladt"] = src1["Kæledyr tilladt"].map({1: "Ja", 0: "Nej"})
src1.to_csv("data/bronze_source1_boligportal.csv", index=False)

# Source 2: AarhusBolig-style scrape
src2 = df.iloc[idx[split1:split2]].copy()
src2 = src2.rename(columns={
    "size_m2": "size", "rooms": "num_rooms", "area": "location",
    "pets_allowed": "pet_friendly", "floor": "floor_level",
    "housing_type": "rental_type", "rent_kr": "monthly_rent", "listed_date": "date_posted"
})
src2["monthly_rent"] = src2["monthly_rent"].astype(int).astype(str) + " kr"
src2["pet_friendly"] = src2["pet_friendly"].map({1: "yes", 0: "no"})
dupes = src2.sample(frac=0.05, random_state=2)
src2 = pd.concat([src2, dupes], ignore_index=True)
src2.to_csv("data/bronze_source2_aarhusbolig.csv", index=False)

# Source 3: DBA-style minimal listing
src3 = df.iloc[idx[split2:]].copy()
src3 = src3.rename(columns={
    "size_m2": "m2", "rooms": "vaerelser", "area": "omraade",
    "pets_allowed": "kaeledyr", "housing_type": "type",
    "rent_kr": "pris", "listed_date": "opslaas_dato"
})
src3 = src3.drop(columns=["floor"])
src3.loc[src3.sample(frac=0.09, random_state=3).index, "m2"] = np.nan
src3.to_csv("data/bronze_source3_dba.csv", index=False)

print("Bronze layer:")
print(" source1:", src1.shape, " source2:", src2.shape, " source3:", src3.shape)

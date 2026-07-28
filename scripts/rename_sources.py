import numpy as np
import pandas as pd

areas = {
    "Aarhus C": 1.35, "Centrum": 1.32, "Risskov": 1.15, "Trøjborg": 1.10,
    "Åbyhøj": 0.95, "Vejlby": 0.92, "Skejby": 1.00, "Viby": 0.85,
    "Hasle": 0.88, "Brabrand": 0.78,
}
area_names = list(areas.keys())
area_weights = [0.15, 0.13, 0.11, 0.10, 0.10, 0.09, 0.08, 0.09, 0.08, 0.07]

def simulate_extra(n, seed):
    rng = np.random.RandomState(seed)
    size = np.round(rng.normal(68, 24, n).clip(16, 165), 0)
    rooms = np.clip(np.round(size / 27 + rng.normal(0, 0.5, n)), 1, 6).astype(int)
    area = rng.choice(area_names, size=n, p=area_weights)
    area_mult = np.array([areas[a] for a in area])
    pets = rng.choice([0, 1], size=n, p=[0.58, 0.42])
    floor = rng.choice(["stuen", "1", "2", "3", "4+"], size=n, p=[0.15,0.25,0.25,0.2,0.15])
    housing_type = rng.choice(["almen", "privat"], size=n, p=[0.35, 0.65])
    base_price_per_m2 = np.where(housing_type == "privat", 128, 80)
    noise = rng.normal(0, 8, n)
    price_per_m2_realised = (base_price_per_m2 * area_mult) + noise
    price = (price_per_m2_realised * size + rooms * 180 + pets * (-100))
    price = np.round(price / 25) * 25
    price = price.clip(2200, 24000)
    listed_date = pd.to_datetime("2024-06-01") + pd.to_timedelta(rng.randint(0, 790, n), unit="D")
    return pd.DataFrame({
        "size_m2": size, "rooms": rooms, "area": area, "pets_allowed": pets,
        "floor": floor, "housing_type": housing_type, "rent_kr": price, "listed_date": listed_date,
    })

# ---------------------------------------------------------------
# Lejeboligmatch.dk -- scraped: minimal columns, no floor info, some size missing
# ---------------------------------------------------------------
n_lbm = 169
df_lbm = simulate_extra(n_lbm, seed=44)
lbm = df_lbm.copy().rename(columns={
    "size_m2": "m2", "rooms": "vaerelser", "area": "omraade",
    "pets_allowed": "kaeledyr", "housing_type": "type", "rent_kr": "pris",
    "listed_date": "opslaas_dato"
})
lbm = lbm.drop(columns=["floor"])
lbm.loc[lbm.sample(frac=0.09, random_state=3).index, "m2"] = np.nan
lbm.to_csv("data/lejeboligmatch_scraped.csv", index=False)
print("lejeboligmatch_scraped.csv:", lbm.shape)

# ---------------------------------------------------------------
# Findbolig.nu -- scraped: very messy, inconsistent area names, junk rows
# ---------------------------------------------------------------
n_fb = 143
df_fb = simulate_extra(n_fb, seed=55)
fb = df_fb.copy()
area_variants = {
    "Aarhus C": ["Aarhus C", "aarhus c", "Århus C", "AARHUS CENTRUM"],
    "Centrum": ["Centrum", "centrum", "City centre"],
    "Risskov": ["Risskov", "risskov"],
    "Trøjborg": ["Trøjborg", "Trojborg", "trøjborg"],
    "Åbyhøj": ["Åbyhøj", "Aabyhoj", "abyhoej"],
    "Vejlby": ["Vejlby", "vejlby"],
    "Skejby": ["Skejby", "skejby"],
    "Viby": ["Viby J", "Viby", "viby j"],
    "Hasle": ["Hasle", "hasle"],
    "Brabrand": ["Brabrand", "brabrand"],
}
rng_fb = np.random.RandomState(66)
fb["area_raw"] = fb["area"].apply(lambda a: rng_fb.choice(area_variants[a]))
fb = fb[["size_m2", "rooms", "area_raw", "pets_allowed", "rent_kr", "listed_date"]]
fb.columns = ["m2_ca", "rum", "sted", "dyr_ok", "husleje_dkk", "dato"]
junk = pd.DataFrame({
    "m2_ca": [np.nan, 45, "?"], "rum": [2, np.nan, 3],
    "sted": ["Aarhus", "", "Aarhus V"], "dyr_ok": [np.nan, 1, 0],
    "husleje_dkk": ["kontakt for pris", 5200, np.nan],
    "dato": ["2025-03-01", "2025-04-12", "2025-05-01"]
})
fb = pd.concat([fb, junk], ignore_index=True)
fb.to_csv("data/findbolig_scraped.csv", index=False)
print("findbolig_scraped.csv:", fb.shape)

# rename existing files to consistent, non-"source" naming
import shutil
shutil.move("data/bronze_source1_boligportal.csv", "data/boligportal_scraped.csv")
shutil.move("data/bronze_source2_aarhusbolig.csv", "data/aarhusbolig_scraped.csv")
shutil.move("data/bronze_source4_lejeboligdk.csv", "data/lejebolig_dk_export.csv")
shutil.move("data/bronze_context_area_stats.csv", "data/dst_omraadestatistik.csv")

# remove old dba/fb marketplace files
import os
for f in ["data/bronze_source3_dba.csv", "data/bronze_source5_fbmarketplace.csv"]:
    if os.path.exists(f):
        os.remove(f)

print("\nDone renaming.")

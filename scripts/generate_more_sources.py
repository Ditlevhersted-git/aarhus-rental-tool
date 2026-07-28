import numpy as np
import pandas as pd

np.random.seed(99)

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
# Source 4: "Lejebolig_dk_export" — semicolon-separated (Danish Excel style),
# comma as decimal separator, address free-text instead of clean floor field
# ---------------------------------------------------------------
n4 = 180
df4 = simulate_extra(n4, seed=11)
src4 = df4.copy()
src4["addr_text"] = src4["floor"].map({
    "stuen": "st.", "1": "1. sal", "2": "2. sal", "3": "3. sal", "4+": "4. sal el. højere"
})
src4 = src4.rename(columns={
    "size_m2": "BoligStr_m2", "rooms": "AntalVaer", "area": "Omraade",
    "pets_allowed": "KaeledyrOK", "housing_type": "Boligtype",
    "rent_kr": "MdLeje", "listed_date": "AnnonceDato"
})
src4 = src4.drop(columns=["floor"])
src4["MdLeje"] = src4["MdLeje"].apply(lambda x: f"{x:,.0f}".replace(",", "."))  # Danish thousand sep
src4["KaeledyrOK"] = src4["KaeledyrOK"].map({1: "1", 0: "0"})
src4.to_csv("data/bronze_source4_lejeboligdk.csv", index=False, sep=";")
print("Source 4 (lejebolig.dk style):", src4.shape)

# ---------------------------------------------------------------
# Source 5: "Facebook Marketplace scrape" — very messy, minimal structure,
# free-text area names (inconsistent casing/spelling), no housing_type, some junk rows
# ---------------------------------------------------------------
n5 = 140
df5 = simulate_extra(n5, seed=22)
src5 = df5.copy()
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
rng5 = np.random.RandomState(33)
src5["area_raw"] = src5["area"].apply(lambda a: rng5.choice(area_variants[a]))
src5 = src5[["size_m2", "rooms", "area_raw", "pets_allowed", "rent_kr", "listed_date"]]
src5.columns = ["m2_ca", "rum", "sted", "dyr_ok", "husleje_dkk", "dato"]
# inject a handful of junk / malformed rows, typical of manual FB scraping
junk = pd.DataFrame({
    "m2_ca": [np.nan, 45, "?"], "rum": [2, np.nan, 3],
    "sted": ["Aarhus", "", "Aarhus V"], "dyr_ok": [np.nan, 1, 0],
    "husleje_dkk": ["kontakt for pris", 5200, np.nan],
    "dato": ["2025-03-01", "2025-04-12", "2025-05-01"]
})
src5 = pd.concat([src5, junk], ignore_index=True)
src5.to_csv("data/bronze_source5_fbmarketplace.csv", index=False)
print("Source 5 (Facebook Marketplace style, messy):", src5.shape)

# ---------------------------------------------------------------
# Context dataset: area-level statistics (mirrors DST-style supplementary data
# used in the original group project) — one row per bydel, not per listing
# ---------------------------------------------------------------
area_context = pd.DataFrame({
    "bydel": area_names,
    "befolkningstal": [8200, 6100, 9400, 7300, 11200, 9800, 6700, 15600, 10300, 13100],
    "andel_studerende_pct": [18, 22, 12, 45, 15, 20, 8, 28, 19, 24],
    "afstand_til_centrum_km": [0.5, 0.2, 4.8, 2.1, 3.5, 5.2, 7.8, 6.0, 3.9, 5.5],
})
area_context.to_csv("data/bronze_context_area_stats.csv", index=False)
print("Context dataset (area-level, DST-style):", area_context.shape)
print(area_context)

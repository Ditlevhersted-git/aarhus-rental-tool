import pandas as pd
import numpy as np

# =================================================================
# BRONZE -> SILVER: read all 6 raw sources, standardise, clean, merge
# =================================================================

# --- 1. Boligportal (clean-ish, Danish column names) ---
bp = pd.read_csv("data/boligportal_scraped.csv")
bp = pd.DataFrame({
    "size_m2": bp["Størrelse (m2)"], "rooms": bp["Værelser"], "area": bp["Bydel"],
    "pets_allowed": bp["Kæledyr tilladt"].map({"Ja": 1, "Nej": 0}),
    "floor": bp["Etage"], "housing_type": bp["Boligtype"],
    "rent_kr": bp["Husleje (kr/md)"], "listed_date": bp["Dato"], "kilde": "boligportal",
})

# --- 2. AarhusBolig (English columns, price as "X kr" string) ---
ab = pd.read_csv("data/aarhusbolig_scraped.csv")
ab = pd.DataFrame({
    "size_m2": ab["size"], "rooms": ab["num_rooms"], "area": ab["location"],
    "pets_allowed": ab["pet_friendly"].map({"yes": 1, "no": 0}),
    "floor": ab["floor_level"], "housing_type": ab["rental_type"],
    "rent_kr": ab["monthly_rent"].str.replace(" kr", "", regex=False).astype(int),
    "listed_date": ab["date_posted"], "kilde": "aarhusbolig",
})
ab = ab.drop_duplicates()  # scraped duplicates

# --- 3. Lejeboligmatch (minimal, no floor, some missing size) ---
lbm = pd.read_csv("data/lejeboligmatch_scraped.csv")
lbm = pd.DataFrame({
    "size_m2": lbm["m2"], "rooms": lbm["vaerelser"], "area": lbm["omraade"],
    "pets_allowed": lbm["kaeledyr"], "floor": np.nan, "housing_type": lbm["type"],
    "rent_kr": lbm["pris"], "listed_date": lbm["opslaas_dato"], "kilde": "lejeboligmatch",
})

# --- 4. Findbolig.nu (messy: inconsistent area spelling, junk rows) ---
fb = pd.read_csv("data/findbolig_scraped.csv")
# normalise inconsistent area names -> canonical
area_norm = {
    "aarhus c": "Aarhus C", "århus c": "Aarhus C", "aarhus centrum": "Aarhus C", "aarhus c": "Aarhus C",
    "centrum": "Centrum", "city centre": "Centrum",
    "risskov": "Risskov", "trøjborg": "Trøjborg", "trojborg": "Trøjborg",
    "åbyhøj": "Åbyhøj", "aabyhoj": "Åbyhøj", "abyhoej": "Åbyhøj",
    "vejlby": "Vejlby", "skejby": "Skejby",
    "viby j": "Viby", "viby": "Viby",
    "hasle": "Hasle", "brabrand": "Brabrand",
}
fb["area_clean"] = fb["sted"].astype(str).str.strip().str.lower().map(area_norm)
fb["m2_ca"] = pd.to_numeric(fb["m2_ca"], errors="coerce")   # "?" -> NaN
fb["husleje_dkk"] = pd.to_numeric(fb["husleje_dkk"], errors="coerce")  # "kontakt for pris" -> NaN
fb = pd.DataFrame({
    "size_m2": fb["m2_ca"], "rooms": fb["rum"], "area": fb["area_clean"],
    "pets_allowed": fb["dyr_ok"], "floor": np.nan, "housing_type": np.nan,
    "rent_kr": fb["husleje_dkk"], "listed_date": fb["dato"], "kilde": "findbolig",
})
fb = fb.dropna(subset=["area", "rent_kr"])  # drop unusable junk rows (no price/area = can't use)

# --- 5. Lejebolig.dk export (semicolon-sep, Danish decimal/thousand format) ---
ldk = pd.read_csv("data/lejebolig_dk_export.csv", sep=";", dtype={"MdLeje": str})
ldk["MdLeje"] = ldk["MdLeje"].str.replace(".", "", regex=False).astype(float)  # "13.050" -> 13050
floor_from_text = {"st.": "stuen", "1. sal": "1", "2. sal": "2", "3. sal": "3", "4. sal el. højere": "4+"}
ldk = pd.DataFrame({
    "size_m2": ldk["BoligStr_m2"], "rooms": ldk["AntalVaer"], "area": ldk["Omraade"],
    "pets_allowed": ldk["KaeledyrOK"], "floor": ldk["addr_text"].map(floor_from_text),
    "housing_type": ldk["Boligtype"], "rent_kr": ldk["MdLeje"],
    "listed_date": ldk["AnnonceDato"], "kilde": "lejebolig_dk",
})

# --- Parse dates PER SOURCE before merging (formats differ across sources,
# e.g. findbolig mixes "YYYY-MM-DD" and "YYYY-MM-DD HH:MM:SS"; parsing the
# combined column at once causes pandas' format auto-detection to misfire
# and silently turn valid dates into NaT) ---
for d in [bp, ab, lbm, fb, ldk]:
    d["listed_date"] = pd.to_datetime(d["listed_date"], errors="coerce")

# --- Merge all listing-level sources ---
silver = pd.concat([bp, ab, lbm, fb, ldk], ignore_index=True)

# --- Cleaning ---
silver["size_m2"] = silver.groupby("area")["size_m2"].transform(lambda x: x.fillna(x.median()))
silver["pets_allowed"] = silver["pets_allowed"].fillna(silver["pets_allowed"].mode()[0]).astype(int)
silver["floor"] = silver["floor"].fillna("ukendt")
silver["housing_type"] = silver["housing_type"].fillna(silver["housing_type"].mode()[0])
silver = silver.dropna(subset=["size_m2", "rooms", "rent_kr", "area", "listed_date"])
silver = silver[(silver["rent_kr"] > 1500) & (silver["size_m2"] > 12)]
silver["rooms"] = silver["rooms"].astype(int)
silver = silver.reset_index(drop=True)

silver.to_csv("data/silver_listings_clean.csv", index=False)

print("=== Silver layer summary ===")
print("Total listings after cleaning:", silver.shape)
print("\nBy source:")
print(silver["kilde"].value_counts())
print("\nMissing values:\n", silver.isna().sum())
print("\nSample:")
print(silver.head())

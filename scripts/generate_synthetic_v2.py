import numpy as np
import pandas as pd

np.random.seed(42)

# ---- Aarhus neighbourhoods, calibrated to realistic relative price levels ----
# Based on real market data: Aarhus avg ~124-140 kr/m2 private, Aarhus N ~183 kr/m2,
# Randers (far outskirt comparator) ~68 kr/m2 -> gives sense of relative spread
areas = {
    "Aarhus C":   1.35,   # city centre, most expensive
    "Centrum":    1.32,
    "Risskov":    1.15,   # attractive coastal/villa area
    "Trøjborg":   1.10,   # close to uni, popular w/ students
    "Åbyhøj":     0.95,
    "Vejlby":     0.92,
    "Skejby":     1.00,   # newer developments
    "Viby":       0.85,
    "Hasle":      0.88,
    "Brabrand":   0.78,   # cheaper western suburb
}
area_names = list(areas.keys())
area_weights = [0.15, 0.13, 0.11, 0.10, 0.10, 0.09, 0.08, 0.09, 0.08, 0.07]

n = 650  # total listings across 3 raw sources before cleaning/dedup

def simulate_base(n):
    size = np.round(np.random.normal(68, 24, n).clip(16, 165), 0)
    rooms = np.clip(np.round(size / 27 + np.random.normal(0, 0.5, n)), 1, 6).astype(int)
    area = np.random.choice(area_names, size=n, p=area_weights)
    area_mult = np.array([areas[a] for a in area])
    pets = np.random.choice([0, 1], size=n, p=[0.58, 0.42])
    floor = np.random.choice(["stuen", "1", "2", "3", "4+"], size=n, p=[0.15,0.25,0.25,0.2,0.15])

    # Housing type: almene (non-profit) vs privat (private rental) -- real, distinct price regimes
    housing_type = np.random.choice(["almen", "privat"], size=n, p=[0.35, 0.65])

    # Calibrated to real Aarhus market:
    # private ~ 124-140 kr/m2 baseline; almene ~ 78-85 kr/m2 baseline
    base_price_per_m2 = np.where(housing_type == "privat", 128, 80)
    noise = np.random.normal(0, 8, n)  # noise on per-m2 rate itself
    price_per_m2_realised = (base_price_per_m2 * area_mult) + noise

    price = (price_per_m2_realised * size
             + rooms * 180
             + pets * (-100))
    price = np.round(price / 25) * 25  # round to nearest 25 kr, like real listings
    # calibrate overall bounds to real market: ~1,787 - 29,500 kr
    price = price.clip(2200, 24000)

    listed_date = pd.to_datetime("2024-06-01") + pd.to_timedelta(
        np.random.randint(0, 790, n), unit="D"
    )

    return pd.DataFrame({
        "size_m2": size,
        "rooms": rooms,
        "area": area,
        "pets_allowed": pets,
        "floor": floor,
        "housing_type": housing_type,
        "rent_kr": price,
        "listed_date": listed_date,
    })

df = simulate_base(n)

print("=== Sanity check against real Aarhus market data ===")
print(f"Mean rent: {df['rent_kr'].mean():.0f} kr (real market: ~8,200-8,960 kr)")
print(f"Median rent: {df['rent_kr'].median():.0f} kr (real market: ~5,800-8,000 kr depending on area)")
print(f"Mean price/m2: {(df['rent_kr']/df['size_m2']).mean():.0f} kr/m2 (real market: ~116-140 kr/m2 private, ~80 kr/m2 almene)")
print(f"Min/Max rent: {df['rent_kr'].min():.0f} - {df['rent_kr'].max():.0f} kr (real market range: ~1,787-29,500 kr)")
print(f"\nBy housing type:")
print(df.groupby("housing_type").apply(lambda g: pd.Series({
    "mean_rent": g["rent_kr"].mean(),
    "mean_price_per_m2": (g["rent_kr"]/g["size_m2"]).mean()
})))

df.to_pickle("data/base_simulated.pkl")
print("\nShape:", df.shape)

# Is This Rent Fair? — Aarhus Rental Price Transparency Tool
Data Science Project (re-exam), Cand.merc. Business Intelligence, Aarhus University BSS.
## Project structure

bronze/       Raw listing sources, five simulated portals + one context dataset
silver/       Cleaned and merged listings (one row per listing)
gold/         Final modelling dataset, feature-engineered, plus model results
scripts/      Python scripts used to build the pipeline and train the models
dashboard/    The tenant-facing price transparency tool (index.html)
report/       Cover page for the written report

## Data pipeline (medallion architecture)
The dataset is synthetic, generated with realistic statistical relationships and
calibrated against real Aarhus market statistics (average rent, price per square
metre, and the price gap between private and non-profit housing). It is not
collected from live listings. This is documented transparently in the report
(Section 1.4 / 3.3.1).
Scripts were run in this order:
1. scripts/generate_synthetic_v2.py — generates the underlying calibrated
   listing data (650 base listings with realistic size/room/area/price
   relationships).
2. scripts/bronze_v2.py — splits the base data into the first raw,
   deliberately messy source files (different column names, formats, missing
   values).
3. scripts/generate_more_sources.py — adds two further raw sources with
   their own format quirks (semicolon separators, inconsistent area
   spelling, junk rows).
4. scripts/rename_sources.py — renames the raw files to their final,
   descriptive names (boligportal_scraped.csv, aarhusbolig_scraped.csv,
   etc.) and removes the earlier placeholder files.
5. scripts/build_silver.py — cleans and merges all five bronze sources
   into the single silver/silver_listings_clean.csv dataset (973 listings).
6. scripts/build_gold.py — joins the silver-layer listings with the
   neighbourhood context dataset (population, student share, distance to
   centre) and applies feature engineering to produce
   gold/gold_listings_modelling.csv.
## Modelling
7. scripts/model_final.py — trains the linear regression baseline and
   extracts the coefficients used in the client-side dashboard tool.
8. scripts/model_comparison.py — trains and compares six models (mean
   benchmark, linear regression, decision tree, random forest, k-nearest
   neighbours, XGBoost) on the gold-layer dataset. Produces
   gold/model_comparison_6.csv, reported in Section 3.6 of the report.
9. scripts/cv_check.py — runs 5-fold cross-validation on the linear
   regression and XGBoost models to check that results are stable across
   different splits, reported in Section 3.6.
## Dashboard
dashboard/index.html is a self-contained, client-side tool. It uses a
linear approximation of the trained model (see coefficients in
scripts/model_final.py) as an interpretable stand-in for the primary
XGBoost model discussed in the report, so that predictions can be computed
directly in the browser without a backend.
To host it, see the report Appendix for the live link, or deploy the file
yourself via GitHub Pages, Netlify, or Surge. lav den her bedre, og nævn ikke netlify og surge, skriv det meget kortereShow moreStreamlined README by removing deployment services, condensing content, renaming folderStreamlined README by removing deployment services, condensing content, renaming folder
Is This Rent Fair? — Aarhus Rental Price Transparency Tool
Data Science Project (re-exam), Cand.merc. Business Intelligence, Aarhus University BSS.
Structure
bronze/     Raw listing sources (5 simulated portals + area context data)
silver/     Cleaned, merged listings (973 rows)
gold/       Final modelling dataset + model results
scripts/    Pipeline and modelling scripts, run in numbered order below
docs/       The price transparency tool (index.html), served via GitHub Pages
report/     Cover page for the written report
Data
The dataset is partly synthetic, generated with realistic relationships and calibrated against real Aarhus market statistics (average rent, price per m², the private/non-profit price gap). It is not collected from live listings. Documented in the report, Sections 1.4 and 3.3.1.
Pipeline

generate_synthetic_v2.py — generates the base calibrated listing data
bronze_v2.py — splits into the first raw, messy source files
generate_more_sources.py — adds two more raw sources
rename_sources.py — renames sources to their final names
build_silver.py — cleans and merges all five sources into silver/
build_gold.py — adds neighbourhood context and feature engineering

Modelling

model_final.py — trains the linear baseline, exports coefficients used in the dashboard
model_comparison.py — compares 6 models (mean benchmark, linear, decision tree, random forest, KNN, XGBoost), see gold/model_comparison_6.csv
cv_check.py — 5-fold cross-validation on linear regression and XGBoost

Dashboard
docs/index.html is self-contained and client-side. It uses a linear approximation of the trained model (coefficients in model_final.py) as an interpretable stand-in for the primary XGBoost model, so predictions run directly in the browser.

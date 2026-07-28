#Is This Rent Fair? — Aarhus Rental Price Transparency Tool
##Data Science Project, Cand.merc. Business Intelligence, Aarhus University BSS. August 2026
Structure
bronze/     Raw listing sources
silver/     Cleaned, merged listings
gold/       Final modelling dataset + model results
scripts/    Pipeline and modelling scripts
docs/       The price transparency tool (index.html), served via GitHub Pages
report/     Cover page for the written report
Overview
The dataset combines historical listings with simulated observations calibrated using Aarhus rental-market statistics rather than live market data (see Sections 1.4 and 3.3.1).

The scripts/ folder contains the full pipeline in execution order, covering data generation, cleaning, feature engineering, model comparison, and cross-validation.

The final tool is located in docs/index.html. It is a self-contained, client-side price checker that uses a linear approximation of the trained model.

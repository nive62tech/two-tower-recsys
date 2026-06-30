# Phase 1 — Synthetic Data Pipeline

## What was built
- Synthetic data generator producing 2000 users, 1500 items, and 50000 interaction events (clicks, skips, dwell time)
- Preference-weighted click probability so generated data has realistic signal (users click more on items matching their preferred categories)
- Preprocessing script that joins users, items, and interactions into a single feature table
- Chronological train/val/test split (80/10/10) by timestamp, matching how real-world recsys data is split

## How to run
.\venv\Scripts\Activate.ps1
python data\generate_synthetic_data.py
python data\preprocess.py

## Key technical decisions
- Chronological split used instead of random split, since recommendation systems are evaluated on predicting future interactions from past ones
- Click probability weighted by category preference match (0.65 vs 0.15) to ensure the two-tower model has real signal to learn in Phase 2
- Raw and processed CSVs excluded from Git via .gitignore since they are fully regeneratable from the two scripts

## Files created
- data/generate_synthetic_data.py
- data/preprocess.py
- data/raw/users.csv (generated, not committed)
- data/raw/items.csv (generated, not committed)
- data/raw/interactions.csv (generated, not committed)
- data/processed/train.csv (generated, not committed)
- data/processed/val.csv (generated, not committed)
- data/processed/test.csv (generated, not committed)

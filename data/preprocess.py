import pandas as pd
from pathlib import Path

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_FRACTION = 0.8
VAL_FRACTION = 0.1
# remaining fraction goes to test


def main():
    users_df = pd.read_csv(RAW_DIR / "users.csv")
    items_df = pd.read_csv(RAW_DIR / "items.csv")
    interactions_df = pd.read_csv(RAW_DIR / "interactions.csv")

    merged_df = interactions_df.merge(users_df, on="user_id", how="left")
    merged_df = merged_df.merge(items_df, on="item_id", how="left")

    merged_df = merged_df.sort_values("timestamp").reset_index(drop=True)

    n = len(merged_df)
    train_end = int(n * TRAIN_FRACTION)
    val_end = int(n * (TRAIN_FRACTION + VAL_FRACTION))

    train_df = merged_df.iloc[:train_end]
    val_df = merged_df.iloc[train_end:val_end]
    test_df = merged_df.iloc[val_end:]

    train_df.to_csv(PROCESSED_DIR / "train.csv", index=False)
    val_df.to_csv(PROCESSED_DIR / "val.csv", index=False)
    test_df.to_csv(PROCESSED_DIR / "test.csv", index=False)

    print(f"Train: {len(train_df)} rows")
    print(f"Val: {len(val_df)} rows")
    print(f"Test: {len(test_df)} rows")


if __name__ == "__main__":
    main()

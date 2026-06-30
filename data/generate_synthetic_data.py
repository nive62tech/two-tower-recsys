import numpy as np
import pandas as pd
import random
from pathlib import Path

random.seed(42)
np.random.seed(42)

NUM_USERS = 2000
NUM_ITEMS = 1500
NUM_INTERACTIONS = 50000

CATEGORIES = ["music", "movies", "tech", "sports", "fashion", "gaming", "books", "food"]
LOCATIONS = ["US", "UK", "IN", "CA", "AU", "DE", "BR", "JP"]
AGE_GROUPS = ["13-17", "18-24", "25-34", "35-44", "45-54", "55+"]
GENDERS = ["male", "female", "other"]

OUTPUT_DIR = Path("data/raw")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def generate_users(n):
    rows = []
    for uid in range(1, n + 1):
        num_prefs = random.randint(1, 3)
        prefs = random.sample(CATEGORIES, num_prefs)
        rows.append({
            "user_id": uid,
            "age_group": random.choice(AGE_GROUPS),
            "gender": random.choice(GENDERS),
            "location": random.choice(LOCATIONS),
            "signup_days_ago": random.randint(1, 1000),
            "preferred_categories": "|".join(prefs),
        })
    return pd.DataFrame(rows)


def generate_items(n):
    rows = []
    for iid in range(1, n + 1):
        category = random.choice(CATEGORIES)
        rows.append({
            "item_id": iid,
            "category": category,
            "subcategory": f"{category}_sub_{random.randint(1, 5)}",
            "price": round(random.uniform(5, 500), 2),
            "popularity_score": round(random.uniform(0, 1), 4),
            "description": f"A {category} item, ID {iid}, curated for fans of {category}.",
        })
    return pd.DataFrame(rows)


def generate_interactions(users_df, items_df, n):
    user_ids = users_df["user_id"].tolist()
    item_ids = items_df["item_id"].tolist()
    item_category_map = dict(zip(items_df["item_id"], items_df["category"]))
    user_prefs_map = dict(zip(users_df["user_id"], users_df["preferred_categories"]))

    rows = []
    base_timestamp = 1_700_000_000

    for i in range(n):
        user_id = random.choice(user_ids)
        item_id = random.choice(item_ids)
        item_category = item_category_map[item_id]
        user_prefs = user_prefs_map[user_id].split("|")

        matches_preference = item_category in user_prefs
        click_prob = 0.65 if matches_preference else 0.15
        clicked = np.random.rand() < click_prob

        if clicked:
            dwell_seconds = round(np.random.gamma(shape=2.0, scale=15.0), 2)
            event_type = "click"
            label = 1
        else:
            dwell_seconds = round(np.random.uniform(0, 3), 2)
            event_type = "skip"
            label = 0

        rows.append({
            "interaction_id": i + 1,
            "user_id": user_id,
            "item_id": item_id,
            "timestamp": base_timestamp + i * random.randint(1, 30),
            "event_type": event_type,
            "dwell_seconds": dwell_seconds,
            "label": label,
        })

    return pd.DataFrame(rows)


def main():
    print("Generating users...")
    users_df = generate_users(NUM_USERS)

    print("Generating items...")
    items_df = generate_items(NUM_ITEMS)

    print("Generating interactions...")
    interactions_df = generate_interactions(users_df, items_df, NUM_INTERACTIONS)

    users_df.to_csv(OUTPUT_DIR / "users.csv", index=False)
    items_df.to_csv(OUTPUT_DIR / "items.csv", index=False)
    interactions_df.to_csv(OUTPUT_DIR / "interactions.csv", index=False)

    print(f"Wrote {len(users_df)} users, {len(items_df)} items, {len(interactions_df)} interactions to data/raw/")


if __name__ == "__main__":
    main()

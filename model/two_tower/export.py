import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from model.two_tower.architecture import TwoTowerModel
from model.two_tower.dataset import encode_categorical

DATA_DIR = Path("data/raw")
ARTIFACTS_DIR = Path("model/artifacts")
DEVICE = torch.device("cpu")


def main():
    items_df = pd.read_csv(DATA_DIR / "items.csv")
    train_df = pd.read_csv("data/processed/train.csv")

    with open(ARTIFACTS_DIR / "vocab.json", "r") as f:
        vocabs = json.load(f)

    model = TwoTowerModel(
        num_users=len(vocabs["user_id"]),
        num_items=len(vocabs["item_id"]),
        num_age_groups=len(vocabs["age_group"]),
        num_genders=len(vocabs["gender"]),
        num_locations=len(vocabs["location"]),
        num_categories=len(vocabs["category"]),
        num_subcategories=len(vocabs["subcategory"]),
    )
    model.load_state_dict(torch.load(ARTIFACTS_DIR / "two_tower_best.pt", map_location=DEVICE))
    model.eval()

    price_vals = train_df["price"].values.astype(np.float32)
    price_mean = price_vals.mean()
    price_std = price_vals.std() + 1e-6

    item_id_batch = []
    category_batch = []
    subcategory_batch = []
    price_batch = []
    popularity_batch = []

    for _, row in items_df.iterrows():
        item_id_batch.append(encode_categorical(row["item_id"], vocabs["item_id"]))
        category_batch.append(encode_categorical(row["category"], vocabs["category"]))
        subcategory_batch.append(encode_categorical(row["subcategory"], vocabs["subcategory"]))
        price_batch.append((float(row["price"]) - price_mean) / price_std)
        popularity_batch.append(float(row["popularity_score"]))

    item_batch = {
        "item_id": torch.tensor(item_id_batch, dtype=torch.long),
        "category": torch.tensor(category_batch, dtype=torch.long),
        "subcategory": torch.tensor(subcategory_batch, dtype=torch.long),
        "price": torch.tensor(price_batch, dtype=torch.float32),
        "popularity_score": torch.tensor(popularity_batch, dtype=torch.float32),
    }

    with torch.no_grad():
        item_embeddings = model.item_tower(item_batch).numpy().astype("float32")

    np.save(ARTIFACTS_DIR / "item_embeddings.npy", item_embeddings)

    item_ids = items_df["item_id"].tolist()
    with open(ARTIFACTS_DIR / "item_id_order.json", "w") as f:
        json.dump(item_ids, f)

    torch.save(model.state_dict(), ARTIFACTS_DIR / "two_tower_final.pt")

    print(f"Exported {item_embeddings.shape[0]} item embeddings of dim {item_embeddings.shape[1]}")
    print(f"Saved to {ARTIFACTS_DIR / 'item_embeddings.npy'} and {ARTIFACTS_DIR / 'item_id_order.json'}")


if __name__ == "__main__":
    main()

import json
import sqlite3
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from model.two_tower.architecture import TwoTowerModel
from model.two_tower.dataset import encode_categorical

ARTIFACTS_DIR = Path("model/artifacts")
SNAPSHOTS_DIR = ARTIFACTS_DIR / "snapshots"
SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)
DEVICE = torch.device("cpu")


def save_embedding_snapshot(model: TwoTowerModel, vocabs: dict, db_path: str):
    items_df = pd.read_csv("data/raw/items.csv")
    train_df = pd.read_csv("data/processed/train.csv")

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

    model.eval()
    with torch.no_grad():
        embeddings = model.item_tower(item_batch).numpy().astype("float32")

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    snapshot_filename = f"item_embeddings_{timestamp}.npy"
    snapshot_path = SNAPSHOTS_DIR / snapshot_filename
    np.save(snapshot_path, embeddings)

    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS embedding_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            snapshot_file TEXT NOT NULL,
            num_items INTEGER,
            embedding_dim INTEGER,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.execute(
        "INSERT INTO embedding_snapshots (timestamp, snapshot_file, num_items, embedding_dim) VALUES (?, ?, ?, ?)",
        (timestamp, snapshot_filename, embeddings.shape[0], embeddings.shape[1])
    )
    conn.commit()
    conn.close()

    print(f"[SnapshotLogger] Saved snapshot: {snapshot_filename} ({embeddings.shape[0]} items, dim={embeddings.shape[1]})")
    return snapshot_filename

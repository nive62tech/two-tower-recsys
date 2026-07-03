import json
import time
import sqlite3
from pathlib import Path

import numpy as np
import faiss

from model.online_learning.incremental_update import run_incremental_update
from streaming.snapshot_logger import save_embedding_snapshot

DB_PATH = "recsys.db"
ARTIFACTS_DIR = Path("model/artifacts")
UPDATE_INTERVAL_SECONDS = 30
INDEX_PATH = ARTIFACTS_DIR / "item_index.faiss"
EMBEDDINGS_PATH = ARTIFACTS_DIR / "item_embeddings.npy"
ITEM_ID_ORDER_PATH = ARTIFACTS_DIR / "item_id_order.json"


def rebuild_faiss_index():
    embeddings = np.load(EMBEDDINGS_PATH).astype("float32")
    with open(ITEM_ID_ORDER_PATH, "r") as f:
        item_ids = json.load(f)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    faiss.write_index(index, str(INDEX_PATH))
    print(f"[Worker] FAISS index rebuilt with {len(item_ids)} items.")


def export_updated_embeddings(model, vocabs):
    import pandas as pd
    import torch
    from model.two_tower.dataset import encode_categorical

    items_df = pd.read_csv("data/raw/items.csv")
    train_df = pd.read_csv("data/processed/train.csv")

    price_vals = train_df["price"].values.astype("float32")
    price_mean = price_vals.mean()
    price_std = price_vals.std() + 1e-6

    item_id_batch, category_batch, subcategory_batch, price_batch, popularity_batch = [], [], [], [], []

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
    import torch
    with torch.no_grad():
        embeddings = model.item_tower(item_batch).numpy().astype("float32")

    np.save(EMBEDDINGS_PATH, embeddings)

    item_ids = items_df["item_id"].tolist()
    with open(ITEM_ID_ORDER_PATH, "w") as f:
        json.dump(item_ids, f)

    print(f"[Worker] Exported {embeddings.shape[0]} updated item embeddings.")


def main():
    print(f"[Worker] Online trainer started. Update interval: {UPDATE_INTERVAL_SECONDS}s")
    since_id = 0

    with open(ARTIFACTS_DIR / "vocab.json", "r") as f:
        vocabs = json.load(f)

    while True:
        time.sleep(UPDATE_INTERVAL_SECONDS)
        print(f"[Worker] Triggering incremental update (since_id={since_id})...")

        try:
            new_since_id, model = run_incremental_update(DB_PATH, since_id=since_id)

            if model is not None:
                export_updated_embeddings(model, vocabs)
                rebuild_faiss_index()
                save_embedding_snapshot(model, vocabs, DB_PATH)
                since_id = new_since_id
                print(f"[Worker] Update cycle complete. since_id now={since_id}")
            else:
                print("[Worker] No update performed this cycle.")

        except Exception as e:
            print(f"[Worker] Error during update cycle: {e}")


if __name__ == "__main__":
    main()

import json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from model.two_tower.architecture import TwoTowerModel
from model.two_tower.dataset import InteractionDataset, build_or_load_vocabs, collate_fn
from model.two_tower.losses import in_batch_negative_loss

ARTIFACTS_DIR = Path("model/artifacts")
DEVICE = torch.device("cpu")
LEARNING_RATE = 0.0001
MIN_BATCH_SIZE = 8


def load_model(vocabs):
    model = TwoTowerModel(
        num_users=len(vocabs["user_id"]),
        num_items=len(vocabs["item_id"]),
        num_age_groups=len(vocabs["age_group"]),
        num_genders=len(vocabs["gender"]),
        num_locations=len(vocabs["location"]),
        num_categories=len(vocabs["category"]),
        num_subcategories=len(vocabs["subcategory"]),
    )
    weights_path = ARTIFACTS_DIR / "two_tower_final.pt"
    model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    model.train()
    return model


def fetch_recent_feedback(db_path: str, since_id: int, limit: int = 256):
    import sqlite3
    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        """
        SELECT f.user_id, f.item_id, f.label
        FROM feedback_events f
        WHERE f.id > ? AND f.label = 1
        ORDER BY f.id ASC
        LIMIT ?
        """,
        (since_id, limit)
    ).fetchall()
    conn.close()
    return rows


def build_dataframe_from_feedback(rows, users_df, items_df):
    if not rows:
        return None

    user_ids = [r[0] for r in rows]
    item_ids = [r[1] for r in rows]

    records = []
    for uid, iid, label in rows:
        if uid not in users_df.index or iid not in items_df.index:
            continue
        u = users_df.loc[uid]
        it = items_df.loc[iid]
        records.append({
            "user_id": uid,
            "item_id": iid,
            "label": label,
            "age_group": u["age_group"],
            "gender": u["gender"],
            "location": u["location"],
            "signup_days_ago": u["signup_days_ago"],
            "category": it["category"],
            "subcategory": it["subcategory"],
            "price": it["price"],
            "popularity_score": it["popularity_score"],
            "dwell_seconds": 0.0,
        })

    if not records:
        return None

    return pd.DataFrame(records)


def run_incremental_update(db_path: str, since_id: int):
    with open(ARTIFACTS_DIR / "vocab.json", "r") as f:
        vocabs = json.load(f)

    users_df = pd.read_csv("data/raw/users.csv").set_index("user_id")
    items_df = pd.read_csv("data/raw/items.csv").set_index("item_id")

    rows = fetch_recent_feedback(db_path, since_id=since_id)
    if not rows:
        print("[OnlineLearning] No new feedback events found. Skipping update.")
        return since_id, None

    df = build_dataframe_from_feedback(rows, users_df, items_df)
    if df is None or len(df) < MIN_BATCH_SIZE:
        print(f"[OnlineLearning] Not enough valid rows ({len(df) if df is not None else 0}). Skipping update.")
        return since_id, None

    model = load_model(vocabs)
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)

    dataset = InteractionDataset(df, vocabs)
    loader = DataLoader(dataset, batch_size=min(64, len(df)), shuffle=True, collate_fn=collate_fn)

    total_loss = 0.0
    for user_batch, item_batch, _ in loader:
        user_batch = {k: v.to(DEVICE) for k, v in user_batch.items()}
        item_batch = {k: v.to(DEVICE) for k, v in item_batch.items()}
        optimizer.zero_grad()
        user_emb, item_emb = model(user_batch, item_batch)
        loss = in_batch_negative_loss(user_emb, item_emb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    torch.save(model.state_dict(), ARTIFACTS_DIR / "two_tower_final.pt")

    max_id = max(r[0] for r in rows)
    new_since_id = max(r for r in [rows[-1][0]] if r) if rows else since_id

    import sqlite3
    conn = sqlite3.connect(db_path)
    new_since_id = conn.execute("SELECT MAX(id) FROM feedback_events WHERE label = 1").fetchone()[0] or since_id
    conn.close()

    print(f"[OnlineLearning] Updated model on {len(df)} events. Loss: {total_loss:.4f}. New since_id: {new_since_id}")
    return new_since_id, model

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

VOCAB_PATH = Path("model/artifacts/vocab.json")


def build_vocab(series):
    unique_vals = sorted(series.dropna().unique().tolist())
    return {val: idx + 1 for idx, val in enumerate(unique_vals)}  # 0 reserved for unknown


def build_or_load_vocabs(train_df):
    if VOCAB_PATH.exists():
        with open(VOCAB_PATH, "r") as f:
            return json.load(f)

    vocabs = {
        "age_group": build_vocab(train_df["age_group"]),
        "gender": build_vocab(train_df["gender"]),
        "location": build_vocab(train_df["location"]),
        "category": build_vocab(train_df["category"]),
        "subcategory": build_vocab(train_df["subcategory"]),
        "user_id": build_vocab(train_df["user_id"]),
        "item_id": build_vocab(train_df["item_id"]),
    }

    VOCAB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(VOCAB_PATH, "w") as f:
        json.dump(vocabs, f)

    return vocabs


def encode_categorical(value, vocab):
    return vocab.get(value, 0)


class InteractionDataset(Dataset):
    def __init__(self, df: pd.DataFrame, vocabs: dict):
        self.df = df.reset_index(drop=True)
        self.vocabs = vocabs

        price_vals = self.df["price"].values.astype(np.float32)
        self.price_mean = price_vals.mean()
        self.price_std = price_vals.std() + 1e-6

        dwell_vals = self.df["dwell_seconds"].values.astype(np.float32)
        self.dwell_mean = dwell_vals.mean()
        self.dwell_std = dwell_vals.std() + 1e-6

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        user_features = {
            "user_id": encode_categorical(row["user_id"], self.vocabs["user_id"]),
            "age_group": encode_categorical(row["age_group"], self.vocabs["age_group"]),
            "gender": encode_categorical(row["gender"], self.vocabs["gender"]),
            "location": encode_categorical(row["location"], self.vocabs["location"]),
            "signup_days_ago": float(row["signup_days_ago"]) / 1000.0,
        }

        item_features = {
            "item_id": encode_categorical(row["item_id"], self.vocabs["item_id"]),
            "category": encode_categorical(row["category"], self.vocabs["category"]),
            "subcategory": encode_categorical(row["subcategory"], self.vocabs["subcategory"]),
            "price": (float(row["price"]) - self.price_mean) / self.price_std,
            "popularity_score": float(row["popularity_score"]),
        }

        label = float(row["label"])

        return user_features, item_features, label


def collate_fn(batch):
    user_ids = torch.tensor([b[0]["user_id"] for b in batch], dtype=torch.long)
    age_groups = torch.tensor([b[0]["age_group"] for b in batch], dtype=torch.long)
    genders = torch.tensor([b[0]["gender"] for b in batch], dtype=torch.long)
    locations = torch.tensor([b[0]["location"] for b in batch], dtype=torch.long)
    signup_days = torch.tensor([b[0]["signup_days_ago"] for b in batch], dtype=torch.float32)

    item_ids = torch.tensor([b[1]["item_id"] for b in batch], dtype=torch.long)
    categories = torch.tensor([b[1]["category"] for b in batch], dtype=torch.long)
    subcategories = torch.tensor([b[1]["subcategory"] for b in batch], dtype=torch.long)
    prices = torch.tensor([b[1]["price"] for b in batch], dtype=torch.float32)
    popularity = torch.tensor([b[1]["popularity_score"] for b in batch], dtype=torch.float32)

    labels = torch.tensor([b[2] for b in batch], dtype=torch.float32)

    user_batch = {
        "user_id": user_ids,
        "age_group": age_groups,
        "gender": genders,
        "location": locations,
        "signup_days_ago": signup_days,
    }
    item_batch = {
        "item_id": item_ids,
        "category": categories,
        "subcategory": subcategories,
        "price": prices,
        "popularity_score": popularity,
    }

    return user_batch, item_batch, labels

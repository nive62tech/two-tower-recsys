import json
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from model.two_tower.architecture import TwoTowerModel
from model.two_tower.dataset import InteractionDataset, build_or_load_vocabs, collate_fn
from model.two_tower.losses import in_batch_negative_loss, recall_at_k

DATA_DIR = Path("data/processed")
ARTIFACTS_DIR = Path("model/artifacts")
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 256
NUM_EPOCHS = 5
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    print(f"Using device: {DEVICE}")

    train_df = pd.read_csv(DATA_DIR / "train.csv")
    val_df = pd.read_csv(DATA_DIR / "val.csv")

    # only positive (clicked) interactions are used as training pairs;
    # negatives come implicitly from in-batch sampling
    train_pos_df = train_df[train_df["label"] == 1].reset_index(drop=True)
    val_pos_df = val_df[val_df["label"] == 1].reset_index(drop=True)

    print(f"Train positive pairs: {len(train_pos_df)}")
    print(f"Val positive pairs: {len(val_pos_df)}")

    vocabs = build_or_load_vocabs(train_df)

    train_dataset = InteractionDataset(train_pos_df, vocabs)
    val_dataset = InteractionDataset(val_pos_df, vocabs)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    model = TwoTowerModel(
        num_users=len(vocabs["user_id"]),
        num_items=len(vocabs["item_id"]),
        num_age_groups=len(vocabs["age_group"]),
        num_genders=len(vocabs["gender"]),
        num_locations=len(vocabs["location"]),
        num_categories=len(vocabs["category"]),
        num_subcategories=len(vocabs["subcategory"]),
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_recall = 0.0
    history = []

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total_loss = 0.0
        num_batches = 0

        for user_batch, item_batch, _ in train_loader:
            user_batch = {k: v.to(DEVICE) for k, v in user_batch.items()}
            item_batch = {k: v.to(DEVICE) for k, v in item_batch.items()}

            optimizer.zero_grad()
            user_emb, item_emb = model(user_batch, item_batch)
            loss = in_batch_negative_loss(user_emb, item_emb)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_train_loss = total_loss / max(num_batches, 1)

        model.eval()
        recalls = []
        with torch.no_grad():
            for user_batch, item_batch, _ in val_loader:
                user_batch = {k: v.to(DEVICE) for k, v in user_batch.items()}
                item_batch = {k: v.to(DEVICE) for k, v in item_batch.items()}
                user_emb, item_emb = model(user_batch, item_batch)
                recalls.append(recall_at_k(user_emb, item_emb, k=10))

        avg_val_recall = sum(recalls) / max(len(recalls), 1)

        print(f"Epoch {epoch}/{NUM_EPOCHS} - train_loss: {avg_train_loss:.4f} - val_recall@10: {avg_val_recall:.4f}")
        history.append({"epoch": epoch, "train_loss": avg_train_loss, "val_recall_at_10": avg_val_recall})

        if avg_val_recall > best_recall:
            best_recall = avg_val_recall
            torch.save(model.state_dict(), ARTIFACTS_DIR / "two_tower_best.pt")
            print(f"  Saved new best checkpoint (recall@10 = {best_recall:.4f})")

    with open(ARTIFACTS_DIR / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nTraining complete. Best val_recall@10: {best_recall:.4f}")
    print(f"Best checkpoint saved to {ARTIFACTS_DIR / 'two_tower_best.pt'}")


if __name__ == "__main__":
    main()

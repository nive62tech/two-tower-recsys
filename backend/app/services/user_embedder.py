import json
from pathlib import Path

import pandas as pd
import torch

from model.two_tower.architecture import TwoTowerModel
from model.two_tower.dataset import encode_categorical

ARTIFACTS_DIR = Path("model/artifacts")
DEVICE = torch.device("cpu")


class UserEmbedderService:
    def __init__(self):
        with open(ARTIFACTS_DIR / "vocab.json", "r") as f:
            self.vocabs = json.load(f)

        self.model = TwoTowerModel(
            num_users=len(self.vocabs["user_id"]),
            num_items=len(self.vocabs["item_id"]),
            num_age_groups=len(self.vocabs["age_group"]),
            num_genders=len(self.vocabs["gender"]),
            num_locations=len(self.vocabs["location"]),
            num_categories=len(self.vocabs["category"]),
            num_subcategories=len(self.vocabs["subcategory"]),
        )
        self.model.load_state_dict(torch.load(ARTIFACTS_DIR / "two_tower_final.pt", map_location=DEVICE))
        self.model.eval()

        self.users_df = pd.read_csv("data/raw/users.csv").set_index("user_id")

    def get_user_embedding(self, user_id: int):
        if user_id not in self.users_df.index:
            raise ValueError(f"user_id {user_id} not found")

        row = self.users_df.loc[user_id]

        user_batch = {
            "user_id": torch.tensor([encode_categorical(user_id, self.vocabs["user_id"])], dtype=torch.long),
            "age_group": torch.tensor([encode_categorical(row["age_group"], self.vocabs["age_group"])], dtype=torch.long),
            "gender": torch.tensor([encode_categorical(row["gender"], self.vocabs["gender"])], dtype=torch.long),
            "location": torch.tensor([encode_categorical(row["location"], self.vocabs["location"])], dtype=torch.long),
            "signup_days_ago": torch.tensor([float(row["signup_days_ago"]) / 1000.0], dtype=torch.float32),
        }

        with torch.no_grad():
            embedding = self.model.user_tower(user_batch).numpy()[0]

        return embedding


user_embedder_service = UserEmbedderService()

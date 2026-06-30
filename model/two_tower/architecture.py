import torch
import torch.nn as nn

EMBEDDING_DIM = 64


class UserTower(nn.Module):
    def __init__(self, num_users, num_age_groups, num_genders, num_locations, embedding_dim=EMBEDDING_DIM):
        super().__init__()
        self.user_id_emb = nn.Embedding(num_users + 1, 32, padding_idx=0)
        self.age_group_emb = nn.Embedding(num_age_groups + 1, 8, padding_idx=0)
        self.gender_emb = nn.Embedding(num_genders + 1, 4, padding_idx=0)
        self.location_emb = nn.Embedding(num_locations + 1, 8, padding_idx=0)

        input_dim = 32 + 8 + 4 + 8 + 1  # + signup_days_ago scalar

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, embedding_dim),
        )

    def forward(self, user_batch):
        u_id = self.user_id_emb(user_batch["user_id"])
        age = self.age_group_emb(user_batch["age_group"])
        gender = self.gender_emb(user_batch["gender"])
        loc = self.location_emb(user_batch["location"])
        signup = user_batch["signup_days_ago"].unsqueeze(1)

        x = torch.cat([u_id, age, gender, loc, signup], dim=1)
        embedding = self.mlp(x)
        return nn.functional.normalize(embedding, p=2, dim=1)


class ItemTower(nn.Module):
    def __init__(self, num_items, num_categories, num_subcategories, embedding_dim=EMBEDDING_DIM):
        super().__init__()
        self.item_id_emb = nn.Embedding(num_items + 1, 32, padding_idx=0)
        self.category_emb = nn.Embedding(num_categories + 1, 8, padding_idx=0)
        self.subcategory_emb = nn.Embedding(num_subcategories + 1, 8, padding_idx=0)

        input_dim = 32 + 8 + 8 + 1 + 1  # + price + popularity_score scalars

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, embedding_dim),
        )

    def forward(self, item_batch):
        i_id = self.item_id_emb(item_batch["item_id"])
        cat = self.category_emb(item_batch["category"])
        subcat = self.subcategory_emb(item_batch["subcategory"])
        price = item_batch["price"].unsqueeze(1)
        popularity = item_batch["popularity_score"].unsqueeze(1)

        x = torch.cat([i_id, cat, subcat, price, popularity], dim=1)
        embedding = self.mlp(x)
        return nn.functional.normalize(embedding, p=2, dim=1)


class TwoTowerModel(nn.Module):
    def __init__(self, num_users, num_items, num_age_groups, num_genders, num_locations,
                 num_categories, num_subcategories, embedding_dim=EMBEDDING_DIM):
        super().__init__()
        self.user_tower = UserTower(num_users, num_age_groups, num_genders, num_locations, embedding_dim)
        self.item_tower = ItemTower(num_items, num_categories, num_subcategories, embedding_dim)

    def forward(self, user_batch, item_batch):
        user_emb = self.user_tower(user_batch)
        item_emb = self.item_tower(item_batch)
        return user_emb, item_emb

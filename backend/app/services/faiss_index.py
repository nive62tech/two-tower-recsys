import json
from pathlib import Path

import faiss
import numpy as np

ARTIFACTS_DIR = Path("model/artifacts")
INDEX_PATH = ARTIFACTS_DIR / "item_index.faiss"
EMBEDDINGS_PATH = ARTIFACTS_DIR / "item_embeddings.npy"
ITEM_ID_ORDER_PATH = ARTIFACTS_DIR / "item_id_order.json"


class FaissIndexService:
    def __init__(self):
        self.index = None
        self.item_ids = None
        self._load_or_build()

    def _load_or_build(self):
        with open(ITEM_ID_ORDER_PATH, "r") as f:
            self.item_ids = json.load(f)

        if INDEX_PATH.exists():
            self.index = faiss.read_index(str(INDEX_PATH))
        else:
            embeddings = np.load(EMBEDDINGS_PATH).astype("float32")
            dim = embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dim)
            self.index.add(embeddings)
            faiss.write_index(self.index, str(INDEX_PATH))

    def search(self, user_embedding: np.ndarray, top_k: int = 20):
        user_embedding = user_embedding.reshape(1, -1).astype("float32")
        scores, indices = self.index.search(user_embedding, top_k)
        result_item_ids = [self.item_ids[i] for i in indices[0] if i != -1]
        result_scores = scores[0][: len(result_item_ids)].tolist()
        return result_item_ids, result_scores

    def rebuild(self):
        embeddings = np.load(EMBEDDINGS_PATH).astype("float32")
        with open(ITEM_ID_ORDER_PATH, "r") as f:
            self.item_ids = json.load(f)
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)
        faiss.write_index(self.index, str(INDEX_PATH))


faiss_index_service = FaissIndexService()

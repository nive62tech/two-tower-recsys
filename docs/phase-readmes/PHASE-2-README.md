# Phase 2 — Two-Tower Model Training

## What was built
- UserTower and ItemTower neural networks: each embeds its categorical features, concatenates with numerical features, and passes through an MLP to produce a normalized 64-dim embedding
- In-batch negative sampling loss, using every other item in a training batch as an implicit negative for each user
- Training loop tracking train loss and val recall@10 per epoch, saving the best checkpoint automatically
- Export script that computes embeddings for the full item catalog and saves them as a numpy array for Phase 3's FAISS index
- Optional Colab notebook mirroring the local training script for future GPU-accelerated runs at larger scale

## How to run
.\venv\Scripts\Activate.ps1
python -m model.two_tower.train
python -m model.two_tower.export

## Key technical decisions
- In-batch negative sampling chosen over explicit negative mining since it scales to large batches without extra sampling cost, the same approach used at YouTube and Pinterest
- Embeddings L2-normalized so dot product similarity is equivalent to cosine similarity, which keeps the downstream FAISS index simple (flat inner product index)
- Trained locally on CPU since the synthetic dataset is small (50k rows); Colab GPU path documented for when the dataset is scaled up
- Only positive (clicked) interactions used as training pairs; negatives are implicit from in-batch sampling rather than skip events directly

## Files created
- model/two_tower/__init__.py
- model/two_tower/architecture.py
- model/two_tower/dataset.py
- model/two_tower/losses.py
- model/two_tower/train.py
- model/two_tower/export.py
- model/notebooks/train_two_tower_colab.ipynb
- model/artifacts/two_tower_best.pt (generated, not committed)
- model/artifacts/two_tower_final.pt (generated, not committed)
- model/artifacts/vocab.json (generated, not committed)
- model/artifacts/item_embeddings.npy (generated, not committed)
- model/artifacts/item_id_order.json (generated, not committed)
- model/artifacts/training_history.json (generated, not committed)

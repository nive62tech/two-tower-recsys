import torch
import torch.nn.functional as F


def in_batch_negative_loss(user_emb, item_emb, temperature=0.1):
    logits = torch.matmul(user_emb, item_emb.T) / temperature
    labels = torch.arange(user_emb.size(0), device=user_emb.device)
    loss = F.cross_entropy(logits, labels)
    return loss


def recall_at_k(user_emb, item_emb, k=10):
    logits = torch.matmul(user_emb, item_emb.T)
    labels = torch.arange(user_emb.size(0), device=user_emb.device)
    top_k = torch.topk(logits, k=min(k, logits.size(1)), dim=1).indices
    hits = (top_k == labels.unsqueeze(1)).any(dim=1).float()
    return hits.mean().item()

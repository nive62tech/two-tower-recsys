from sentence_transformers import CrossEncoder

MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class CrossEncoderReranker:
    def __init__(self):
        self.model = CrossEncoder(MODEL_NAME)

    def rerank(self, query: str, items: list[dict], text_field: str = "description"):
        if not items:
            return []

        pairs = [[query, item[text_field]] for item in items]
        scores = self.model.predict(pairs).tolist()

        for item, score in zip(items, scores):
            item["rerank_score"] = round(float(score), 6)

        reranked = sorted(items, key=lambda x: x["rerank_score"], reverse=True)
        return reranked


cross_encoder_reranker = CrossEncoderReranker()

from typing import List, Dict, Any, Tuple
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from multimodal_rag.logger import logger
from multimodal_rag.config.config import settings
class TextEmbedder:
    """
    Token-aware sliding-window text embedder.
    Works with page-wise or character-wise chunks.
    Prevents truncation for 512-token models.
    """
    def __init__(self):
        self.model_name = settings.embedding.pretrained
        self.device = torch.device(settings.embedding.device)

        # Hard model constraint (E5 / Nomic / BERT-style encoders)
        self.max_tokens = 512
        self.token_overlap = 64
        logger.info(
            f"Loading embedding model | "
            f"model={self.model_name} | device={self.device}"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()

        logger.info("Embedding model loaded successfully")

    @staticmethod
    def _mean_pooling(
        token_embeddings: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return (token_embeddings * mask).sum(dim=1) / mask.sum(dim=1)

    def _create_token_windows(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Split tokenized input into overlapping windows.
        """
        windows = []
        stride = self.max_tokens - self.token_overlap
        total_tokens = input_ids.size(1)
        start = 0

        while start < total_tokens:
            end = min(start + self.max_tokens, total_tokens)

            windows.append(
                {
                    "input_ids": input_ids[:, start:end],
                    "attention_mask": attention_mask[:, start:end],
                    "token_start": start,
                    "token_end": end,
                }
            )

            if end == total_tokens:
                break

            start += stride

        return windows

    def embed_texts(
        self,
        text_chunks: List[Dict[str, Any]],
    ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Generate embeddings using sliding-window token strategy.

        Returns:
            embeddings: np.ndarray [N, dim]
            metadatas: aligned metadata for each embedding
        """

        all_embeddings: List[np.ndarray] = []
        all_metadatas: List[Dict[str, Any]] = []

        logger.info(f"Embedding {len(text_chunks)} text chunks")

        with torch.no_grad():
            for chunk in text_chunks:
                # Keep your ingestion metadata intact
                base_metadata = {
                    k: v for k, v in chunk.items() if k != "content"
                }

                # Prefix for E5 / Nomic (safe for others)
                text = f"passage: {chunk['content']}"

                encoded = self.tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=False,
                ).to(self.device)

                windows = self._create_token_windows(
                    encoded["input_ids"],
                    encoded["attention_mask"],
                )

                for window_id, window in enumerate(windows):
                    output = self.model(
                        input_ids=window["input_ids"],
                        attention_mask=window["attention_mask"],
                    )

                    pooled = self._mean_pooling(
                        output.last_hidden_state,
                        window["attention_mask"],
                    )

                    if settings.retrieval.metric == "cosine":
                        pooled = torch.nn.functional.normalize(
                            pooled, p=2, dim=1
                        )

                    all_embeddings.append(pooled.cpu().numpy())

                    metadata = base_metadata.copy()
                    metadata.update(
                        {
                            "window_id": window_id,
                            "token_start": window["token_start"],
                            "token_end": window["token_end"],
                        }
                    )

                    all_metadatas.append(metadata)

        embeddings_np = np.vstack(all_embeddings)

        logger.info(
            f"Generated embeddings | "
            f"vectors={embeddings_np.shape[0]} | "
            f"dim={embeddings_np.shape[1]}"
        )

        return embeddings_np, all_metadatas

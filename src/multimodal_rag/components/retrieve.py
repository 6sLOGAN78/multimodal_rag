from typing import List, Dict, Any

import torch
import open_clip

from multimodal_rag.components.embed_text import TextEmbedder
from multimodal_rag.components.store import get_chroma_collection
from multimodal_rag.config.config import settings
from multimodal_rag.logger import logger


# ==================================================
# TEXT → TEXT
# Encoder : E5
# Index   : index/text
# ==================================================
def retrieve_text_to_text(
    query: str,
    top_k: int | None = None,
) -> List[Dict[str, Any]]:
    """
    Text → Text retrieval
    Encoder: E5
    Index: index/text
    """
    top_k = top_k or settings.retrieval.top_k
    logger.info("Retrieving TEXT → TEXT (E5)")

    embedder = TextEmbedder()

    query_embedding, _ = embedder.embed_texts(
        [{
            "content": query,
            "chunk_id": -1,
            "source": "query",
        }]
    )

    collection = get_chroma_collection(
        index_type="text",
        collection_name="text_embeddings",
    )

    results = collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    return _format_results(results)


# ==================================================
# TEXT → IMAGE (PDF IMAGES ONLY)
# Encoder : CLIP (text encoder)
# Index   : index/pdf_image
# ==================================================
def retrieve_text_to_image(
    query: str,
    top_k: int | None = None,
) -> List[Dict[str, Any]]:
    """
    Text → Image retrieval (PDF images)
    Encoder: CLIP (text encoder)
    Index: index/pdf_image
    """
    top_k = top_k or settings.retrieval.top_k
    logger.info("Retrieving TEXT → PDF IMAGE (CLIP)")

    device = torch.device(settings.image_embedding.device)

    model, _, _ = open_clip.create_model_and_transforms(
        model_name=settings.image_embedding.pretrained,
        pretrained="laion2b_s32b_b79k",
    )
    tokenizer = open_clip.get_tokenizer(settings.image_embedding.pretrained)

    model = model.to(device)
    model.eval()

    with torch.no_grad():
        tokens = tokenizer([query]).to(device)
        query_embedding = model.encode_text(tokens)
        query_embedding = query_embedding / query_embedding.norm(dim=-1, keepdim=True)

    collection = get_chroma_collection(
        index_type="pdf_image",               # ✅ IMPORTANT
        collection_name="pdf_image_embeddings",
    )

    results = collection.query(
        query_embeddings=query_embedding.cpu().numpy().tolist(),
        n_results=top_k,
        include=["metadatas", "distances"],
    )

    return _format_results(results)


# ==================================================
# TEXT → (TEXT + IMAGE)
# NO score fusion, NO mixing
# ==================================================
def retrieve_text_to_text_and_image(
    query: str,
    top_k_text: int | None = None,
    top_k_image: int | None = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Runs two independent searches:

    1. text → text (E5, index/text)
    2. text → image (CLIP, index/pdf_image)

    Returns results separately.
    """
    return {
        "text_results": retrieve_text_to_text(query, top_k_text),
        "image_results": retrieve_text_to_image(query, top_k_image),
    }


# ==================================================
# SAFE RESULT FORMATTER
# ==================================================
def _format_results(results) -> List[Dict[str, Any]]:
    """
    Normalize Chroma results safely.
    Works for text and image collections.
    """
    formatted = []

    ids = results["ids"][0]
    distances = results["distances"][0]
    metadatas = results["metadatas"][0]

    documents = results.get("documents")
    docs_list = documents[0] if documents is not None else None

    for i in range(len(ids)):
        formatted.append(
            {
                "id": ids[i],
                "distance": distances[i],
                "metadata": metadatas[i],
                "document": docs_list[i] if docs_list else None,
            }
        )

    return formatted

from typing import List, Dict, Any
import chromadb
from chromadb.api.models.Collection import Collection
from multimodal_rag.config.config import settings
from multimodal_rag.logger import logger
def get_chroma_collection(
    collection_name: str = "multimodal_embeddings",
) -> Collection:
    """
    Get or create a persistent ChromaDB collection.
    Storage path is controlled by config.yaml (indexing.text).
    """

    persist_dir = settings.indexing.text

    logger.info(f"Initializing ChromaDB at: {persist_dir}")

    client = chromadb.PersistentClient(
        path=str(persist_dir)
    )

    existing = {c.name for c in client.list_collections()}

    if collection_name in existing:
        logger.info(f"Loading existing Chroma collection: {collection_name}")
        collection = client.get_collection(collection_name)
    else:
        logger.info(f"Creating new Chroma collection: {collection_name}")
        collection = client.create_collection(name=collection_name)

    return collection


# --------------------------------------------------
# Store embeddings
# --------------------------------------------------
def store_in_chroma(
    ids: List[str],
    embeddings: List[List[float]],
    metadatas: List[Dict[str, Any]],
    documents: List[str],
    collection_name: str = "multimodal_embeddings",
):
    """
    Store embeddings + metadata + documents in ChromaDB.
    """

    assert len(ids) == len(embeddings) == len(metadatas) == len(documents), (
        "ids, embeddings, metadatas, documents must be same length"
    )

    collection = get_chroma_collection(collection_name)

    collection.add(
        ids=ids,
        embeddings=embeddings,
        metadatas=metadatas,
        documents=documents,
    )

    logger.info(
        f"Stored {len(ids)} embeddings in Chroma collection '{collection_name}'"
    )

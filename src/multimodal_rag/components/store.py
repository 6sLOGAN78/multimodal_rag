from typing import List, Dict, Any, Literal

import chromadb
from chromadb.api.models.Collection import Collection

from multimodal_rag.config.config import settings
from multimodal_rag.logger import logger

IndexType = Literal[
    "text",
    "image",
    "pdf_image",
    "image_pdfimage",
    "image_text",
]



def _get_index_path(index_type: IndexType):
    try:
        return getattr(settings.indexing, index_type)
    except AttributeError:
        raise ValueError(f"Invalid index_type: {index_type}")

def get_chroma_collection(
    index_type: IndexType,
    collection_name: str,
) -> Collection:
    """
    Get or create a persistent ChromaDB collection
    for a specific index type.
    """

    persist_dir = _get_index_path(index_type)

    logger.info(
        f"Initializing ChromaDB | "
        f"index_type={index_type} | path={persist_dir}"
    )

    client = chromadb.PersistentClient(
        path=str(persist_dir)
    )

    existing = {c.name for c in client.list_collections()}

    if collection_name in existing:
        logger.info(
            f"Loading existing collection '{collection_name}' "
            f"({index_type})"
        )
        collection = client.get_collection(collection_name)
    else:
        logger.info(
            f"Creating new collection '{collection_name}' "
            f"({index_type})"
        )
        collection = client.create_collection(name=collection_name)

    return collection


# ----------------------------------------
# Store embeddings
# ----------------------------------------
def store_in_chroma(
    *,
    index_type: IndexType,
    collection_name: str,
    ids: List[str],
    embeddings: List[List[float]],
    metadatas: List[Dict[str, Any]],
    documents: List[str],
):
    """
    Store embeddings in ChromaDB under a specific index type.
    """

    n = len(ids)
    assert (
        n == len(embeddings) == len(metadatas) == len(documents)
    ), "ids, embeddings, metadatas, documents must be same length"

    collection = get_chroma_collection(
        index_type=index_type,
        collection_name=collection_name,
    )

    collection.add(
        ids=ids,
        embeddings=embeddings,
        metadatas=metadatas,
        documents=documents,
    )

    logger.info(
        f"Stored {n} embeddings | "
        f"index_type={index_type} | "
        f"collection={collection_name}"
    )

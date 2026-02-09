from collections import defaultdict
from datetime import datetime
from pathlib import Path

from multimodal_rag.components.ingest import ingest_documents
from multimodal_rag.components.embed_text import TextEmbedder
from multimodal_rag.components.store import store_in_chroma
from multimodal_rag.config.config import settings
from multimodal_rag.logger import logger


def main():
    logger.info("===== CHECK INGEST + EMBED PIPELINE =====")
    out_dir = Path("debug_outputs")
    out_dir.mkdir(exist_ok=True)
    out_file = out_dir / f"ingest_embed_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

    def write(line: str):
        with open(out_file, "a") as f:
            f.write(line + "\n")

    write("===== INGEST + EMBED FULL REPORT =====\n")
    data = ingest_documents(
        docs_dir=settings.paths.docs_dir,
        images_dir=settings.paths.images_dir,
    )

    text_chunks = data["text_chunks"]

    logger.info(f"Total logical text chunks: {len(text_chunks)}")
    write(f"Total logical text chunks: {len(text_chunks)}\n")

    write("----- INGESTED CHUNKS -----")
    for chunk in text_chunks:
        page = chunk.get("page_start", chunk.get("page", "N/A"))
        write(
            f"chunk_id={chunk.get('chunk_id')} | "
            f"page={page} | "
            f"chars={len(chunk['content'])} | "
            f"source={chunk['source']}"
        )

    embedder = TextEmbedder()
    embeddings, metadatas = embedder.embed_texts(text_chunks)

    write("\n----- EMBEDDING SUMMARY -----")
    write(f"Total embeddings generated: {embeddings.shape[0]}")
    write(f"Embedding dimension: {embeddings.shape[1]}")

    logger.info(f"Embeddings shape: {embeddings.shape}")

    page_to_windows = defaultdict(int)

    for meta in metadatas:
        if "page_start" in meta:
            key = (meta["source"], meta["page_start"])
        else:
            key = (meta["source"], meta["chunk_id"])

        page_to_windows[key] += 1

    write("\n----- WINDOW DISTRIBUTION PER PAGE / CHUNK -----")
    for (src, page), count in page_to_windows.items():
        write(f"source={src} | page/chunk={page} | windows={count}")
    write("\n----- FULL EMBEDDING METADATA -----")
    for i, meta in enumerate(metadatas):
        write(
            f"[{i}] "
            f"chunk_id={meta.get('chunk_id')} | "
            f"page={meta.get('page_start', meta.get('page', 'N/A'))} | "
            f"window_id={meta.get('window_id')} | "
            f"token_start={meta.get('token_start')} | "
            f"token_end={meta.get('token_end')} | "
            f"source={meta.get('source')}"
        )

    logger.info("===== STORING EMBEDDINGS IN CHROMADB =====")

    ids = [
        f"{meta['source']}::chunk_{meta.get('chunk_id')}::window_{meta.get('window_id')}"
        for meta in metadatas
    ]

    documents = []
    for meta in metadatas:

        chunk_id = meta.get("chunk_id")
        documents.append(text_chunks[chunk_id]["content"])

    store_in_chroma(
        ids=ids,
        embeddings=embeddings.tolist(),
        metadatas=metadatas,
        documents=documents,
    )

    write("\n===== STORED IN CHROMADB =====")

    assert embeddings.ndim == 2, "Embeddings must be 2D"
    assert embeddings.shape[1] == settings.embedding.dim, (
        f"Embedding dim mismatch: expected {settings.embedding.dim}"
    )
    assert len(embeddings) == len(metadatas), (
        "Each embedding must have metadata"
    )

    write("\n===== ALL CHECKS PASSED =====")
    logger.info("===== ALL CHECKS PASSED =====")
    logger.info(f"Detailed report saved to: {out_file}")
    logger.info("Embeddings successfully stored in ChromaDB")


if __name__ == "__main__":
    main()

from datetime import datetime
from pathlib import Path
from typing import List

from multimodal_rag.components.ingest import ingest_documents
from multimodal_rag.components.embed_text import TextEmbedder
from multimodal_rag.components.image_embed import ImageEmbedder
from multimodal_rag.components.store import store_in_chroma
from multimodal_rag.config.config import settings
from multimodal_rag.logger import logger

def make_text_ids(metadatas: List[dict]) -> List[str]:
    """
    Text embeddings are window-based → must include chunk + window
    """
    return [
        f"text::{meta['source']}::chunk_{meta['chunk_id']}::window_{meta['window_id']}"
        for meta in metadatas
    ]


def make_image_ids(prefix: str, metadatas: List[dict]) -> List[str]:
    """
    Images are atomic → image_id is already unique
    """
    return [
        f"{prefix}::{meta['image_id']}"
        for meta in metadatas
    ]



def main():
    logger.info("===== MULTIMODAL RAG FULL PIPELINE START =====")


    out_dir = Path("debug_outputs")
    out_dir.mkdir(exist_ok=True)

    report_file = out_dir / f"rg_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

    def write(line: str):
        with open(report_file, "a") as f:
            f.write(line + "\n")

    write("===== MULTIMODAL RAG PIPELINE REPORT =====\n")

    # ---------------------------------------------
    # 1. INGEST
    # ---------------------------------------------
    data = ingest_documents(
        docs_dir=settings.paths.docs_dir,
        images_dir=settings.paths.images_dir,
    )

    text_chunks = data["text_chunks"]
    pdf_images = data["pdf_images"]
    images = data["images"]

    write(f"Text chunks: {len(text_chunks)}")
    write(f"PDF images: {len(pdf_images)}")
    write(f"Normal images: {len(images)}\n")

    # ---------------------------------------------
    # 2. TEXT EMBEDDING
    # ---------------------------------------------
    logger.info("Embedding TEXT chunks")
    text_embedder = TextEmbedder()

    text_embeddings, text_metadatas = text_embedder.embed_texts(text_chunks)

    # ---- checks ----
    assert text_embeddings.ndim == 2
    assert text_embeddings.shape[1] == settings.embedding.dim
    assert len(text_embeddings) == len(text_metadatas)

    text_ids = make_text_ids(text_metadatas)
    text_docs = [
        text_chunks[m["chunk_id"]]["content"]
        for m in text_metadatas
    ]

    store_in_chroma(
        index_type="text",
        collection_name="text_embeddings",
        ids=text_ids,
        embeddings=text_embeddings.tolist(),
        metadatas=text_metadatas,
        documents=text_docs,
    )

    write(
        f"[TEXT] vectors={len(text_embeddings)} "
        f"dim={text_embeddings.shape[1]}"
    )

    # ---------------------------------------------
    # 3. IMAGE EMBEDDER (INIT ONCE)
    # ---------------------------------------------
    image_embedder = ImageEmbedder()

    # ---------------------------------------------
    # 4. NORMAL IMAGE EMBEDDING
    # ---------------------------------------------
    if images:
        logger.info("Embedding NORMAL images")

        img_embeddings, img_metadatas = image_embedder.embed_images(images)

        assert img_embeddings[0].shape[0] == settings.image_embedding.dim

        img_ids = make_image_ids("image", img_metadatas)

        store_in_chroma(
            index_type="image",
            collection_name="image_embeddings",
            ids=img_ids,
            embeddings=img_embeddings,
            metadatas=img_metadatas,
            documents=[""] * len(img_ids),
        )

        write(
            f"[IMAGE] vectors={len(img_embeddings)} "
            f"dim={img_embeddings[0].shape[0]}"
        )

    # ---------------------------------------------
    # 5. PDF IMAGE EMBEDDING
    # ---------------------------------------------
    if pdf_images:
        logger.info("Embedding PDF images")

        pdf_img_embeddings, pdf_img_metadatas = image_embedder.embed_images(pdf_images)

        assert pdf_img_embeddings[0].shape[0] == settings.image_embedding.dim

        pdf_img_ids = make_image_ids("pdf_image", pdf_img_metadatas)

        store_in_chroma(
            index_type="pdf_image",
            collection_name="pdf_image_embeddings",
            ids=pdf_img_ids,
            embeddings=pdf_img_embeddings,
            metadatas=pdf_img_metadatas,
            documents=[""] * len(pdf_img_ids),
        )

        write(
            f"[PDF_IMAGE] vectors={len(pdf_img_embeddings)} "
            f"dim={pdf_img_embeddings[0].shape[0]}"
        )

    # ---------------------------------------------
    # 6. IMAGE + PDF IMAGE COMBINED
    # ---------------------------------------------
    combined_images = images + pdf_images
    if combined_images:
        logger.info("Storing combined IMAGE + PDF_IMAGE index")

        combined_embeddings, combined_metadatas = image_embedder.embed_images(
            combined_images
        )

        combined_ids = make_image_ids(
            "image_pdfimage",
            combined_metadatas,
        )

        store_in_chroma(
            index_type="image_pdfimage",
            collection_name="image_pdfimage_embeddings",
            ids=combined_ids,
            embeddings=combined_embeddings,
            metadatas=combined_metadatas,
            documents=[""] * len(combined_ids),
        )

        write(
            f"[IMAGE_PDFIMAGE] vectors={len(combined_embeddings)} "
            f"dim={combined_embeddings[0].shape[0]}"
        )

    # ---------------------------------------------
    # FINAL
    # ---------------------------------------------
    write("\n===== ALL CHECKS PASSED =====")
    logger.info("===== PIPELINE COMPLETED SUCCESSFULLY =====")
    logger.info(f"Report saved to: {report_file}")


if __name__ == "__main__":
    main()

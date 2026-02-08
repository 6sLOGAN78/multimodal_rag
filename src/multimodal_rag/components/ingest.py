import io
from pathlib import Path
from typing import List, Dict, Any

import fitz  
import docx
from PIL import Image
from langchain.text_splitter import RecursiveCharacterTextSplitter

from multimodal_rag.logger import logger
from multimodal_rag.config.config import settings


# --------------------------------------------------
# Constants
# --------------------------------------------------
TEXT_EXTENSIONS = {".txt", ".md"}
PDF_EXTENSION = ".pdf"
DOCX_EXTENSION = ".docx"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}


# --------------------------------------------------
# Text Splitter (used everywhere except PDF page mode)
# --------------------------------------------------
def get_text_splitter() -> RecursiveCharacterTextSplitter:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunking.chunk_size,
        chunk_overlap=settings.chunking.chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],
    )

    logger.info(
        f"Text splitter initialized | "
        f"chunk_size={settings.chunking.chunk_size}, "
        f"overlap={settings.chunking.chunk_overlap}"
    )

    return splitter


def load_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def extract_text_from_docx(path: Path) -> str:
    document = docx.Document(path)
    return "\n".join(p.text for p in document.paragraphs if p.text.strip())


def extract_text_from_pdf(pdf_path: Path) -> List[str]:
    pages = []
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text = page.get_text("text")
            if text.strip():
                pages.append(text)
    return pages


def chunk_pdf_pages(
    pages: List[str],
    page_size: int,
) -> List[Dict[str, Any]]:
    chunks = []

    for i in range(0, len(pages), page_size):
        combined_text = "\n".join(pages[i:i + page_size])
        chunks.append(
            {
                "content": combined_text,
                "page_start": i,
                "page_end": min(i + page_size - 1, len(pages) - 1),
            }
        )

    return chunks

def extract_images_from_pdf(pdf_path: Path) -> List[Dict[str, Any]]:
    pdf_images = []

    with fitz.open(pdf_path) as doc:
        for page_index, page in enumerate(doc):
            for img_index, img in enumerate(page.get_images(full=True)):
                try:
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]

                    pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                    image_id = f"{pdf_path.stem}_page_{page_index}_img_{img_index}"

                    pdf_images.append(
                        {
                            "type": "pdf_image",
                            "image": pil_image,
                            "image_id": image_id,
                            "source": str(pdf_path),
                            "page": page_index,
                        }
                    )

                except Exception as e:
                    logger.warning(
                        f"Failed extracting image {img_index} "
                        f"on page {page_index} from {pdf_path}: {e}"
                    )

    return pdf_images


def ingest_normal_images(images_dir: Path) -> List[Dict[str, Any]]:
    images = []

    if not images_dir.exists():
        return images

    for path in images_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            images.append(
                {
                    "type": "image",
                    "image_path": str(path),
                    "image_id": path.stem,
                    "source": str(path),
                }
            )

    return images

def ingest_documents(
    docs_dir: Path,
    images_dir: Path,
) -> Dict[str, List[Dict[str, Any]]]:

    splitter = get_text_splitter()
    chunk_type = settings.chunking.chunk_type

    text_chunks: List[Dict[str, Any]] = []
    pdf_images: List[Dict[str, Any]] = []
    images: List[Dict[str, Any]] = []

    for path in docs_dir.rglob("*"):
        if not path.is_file():
            continue

        suffix = path.suffix.lower()

        if suffix in TEXT_EXTENSIONS:
            text = load_text_file(path)
            chunks = splitter.split_text(text)

            for i, chunk in enumerate(chunks):
                text_chunks.append(
                    {
                        "type": "text",
                        "content": chunk,
                        "source": str(path),
                        "chunk_id": i,
                    }
                )

        elif suffix == DOCX_EXTENSION:
            text = extract_text_from_docx(path)
            chunks = splitter.split_text(text)

            for i, chunk in enumerate(chunks):
                text_chunks.append(
                    {
                        "type": "text",
                        "content": chunk,
                        "source": str(path),
                        "chunk_id": i,
                    }
                )


        elif suffix == PDF_EXTENSION:
            pages = extract_text_from_pdf(path)
            if chunk_type == "page":
                page_chunks = chunk_pdf_pages(
                    pages,
                    settings.chunking.chunk_page_size,
                )

                for i, chunk in enumerate(page_chunks):
                    text_chunks.append(
                        {
                            "type": "text",
                            "content": chunk["content"],
                            "source": str(path),
                            "page_start": chunk["page_start"],
                            "page_end": chunk["page_end"],
                            "chunk_id": i,
                        }
                    )

           
            else:
                for page_id, page_text in enumerate(pages):
                    chunks = splitter.split_text(page_text)
                    for i, chunk in enumerate(chunks):
                        text_chunks.append(
                            {
                                "type": "text",
                                "content": chunk,
                                "source": str(path),
                                "page": page_id,
                                "chunk_id": i,
                            }
                        )

            pdf_images.extend(extract_images_from_pdf(path))

    images.extend(ingest_normal_images(images_dir))

    logger.info(
        f"Ingested {len(text_chunks)} text chunks | "
        f"{len(pdf_images)} PDF images | "
        f"{len(images)} normal images"
    )

    return {
        "text_chunks": text_chunks,
        "pdf_images": pdf_images,
        "images": images,
    }

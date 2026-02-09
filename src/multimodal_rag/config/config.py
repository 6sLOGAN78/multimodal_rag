from dataclasses import dataclass
from pathlib import Path
import os

from multimodal_rag.utils.common import read_yaml
from multimodal_rag.logger import logger


# -----------------------------
# Dataclasses
# -----------------------------
@dataclass(frozen=True)
class PathsConfig:
    data_dir: Path
    docs_dir: Path
    images_dir: Path
    index_dir: Path


@dataclass(frozen=True)
class IndexingConfig:
    image: Path            # only normal images
    image_pdfimage: Path   # normal images + pdf images
    pdf_image: Path        # only pdf images
    text: Path             # only text
    image_text: Path       # multimodal (text + image)


@dataclass(frozen=True)
class EmbeddingConfig:
    model: str
    pretrained: str
    dim: int
    device: str


@dataclass(frozen=True)
class ImageEmbeddingConfig:
    model: str
    pretrained: str
    dim: int
    device: str


@dataclass(frozen=True)
class ChunkingConfig:
    chunk_size: int
    chunk_overlap: int
    chunk_type: str
    chunk_page_size: int


@dataclass(frozen=True)
class RetrievalConfig:
    top_k: int
    metric: str


@dataclass(frozen=True)
class LLMConfig:
    provider: str
    model: str
    temperature: float
    max_tokens: int


@dataclass(frozen=True)
class Settings:
    paths: PathsConfig
    indexing: IndexingConfig
    embedding: EmbeddingConfig
    image_embedding: ImageEmbeddingConfig
    chunking: ChunkingConfig
    retrieval: RetrievalConfig
    llm: LLMConfig
    gemini_api_key: str


# -----------------------------
# Loader
# -----------------------------
def load_settings() -> Settings:
    base_dir = Path(__file__).resolve().parent.parent.parent.parent
    config_path = base_dir / "config" / "config.yaml"

    cfg = read_yaml(config_path)

    # ---- Paths ----
    paths = PathsConfig(
        data_dir=base_dir / cfg.paths.data_dir,
        docs_dir=base_dir / cfg.paths.docs_dir,
        images_dir=base_dir / cfg.paths.images_dir,
        index_dir=base_dir / cfg.paths.index_dir,
    )

    paths.data_dir.mkdir(exist_ok=True)
    paths.docs_dir.mkdir(parents=True, exist_ok=True)
    paths.images_dir.mkdir(parents=True, exist_ok=True)
    paths.index_dir.mkdir(exist_ok=True)

    # ---- Indexing ----
    indexing = IndexingConfig(
        image=base_dir / cfg.indexing.image,
        image_pdfimage=base_dir / cfg.indexing.image_pdfimage,
        pdf_image=base_dir / cfg.indexing.pdf_image,
        text=base_dir / cfg.indexing.text,
        image_text=base_dir / cfg.indexing.image_text,
    )

    indexing.image.mkdir(parents=True, exist_ok=True)
    indexing.image_pdfimage.mkdir(parents=True, exist_ok=True)
    indexing.pdf_image.mkdir(parents=True, exist_ok=True)
    indexing.text.mkdir(parents=True, exist_ok=True)
    indexing.image_text.mkdir(parents=True, exist_ok=True)

    embedding = EmbeddingConfig(
        model=cfg.embedding.model,
        pretrained=cfg.embedding.pretrained,
        dim=int(cfg.embedding.dim),
        device=cfg.embedding.device,
    )

    image_embedding = ImageEmbeddingConfig(
        model=cfg.image_embedding.model,
        pretrained=cfg.image_embedding.pretrained,
        dim=int(cfg.image_embedding.dim),
        device=cfg.image_embedding.device,
    )

    chunking = ChunkingConfig(
        chunk_size=int(cfg.chunking.chunk_size),
        chunk_overlap=int(cfg.chunking.chunk_overlap),
        chunk_type=cfg.chunking.chunk_type,
        chunk_page_size=int(cfg.chunking.chunk_page_size),
    )
    retrieval = RetrievalConfig(
        top_k=int(cfg.retrieval.top_k),
        metric=cfg.retrieval.metric,
    )

    llm = LLMConfig(
        provider=cfg.llm.provider,
        model=cfg.llm.model,
        temperature=float(cfg.llm.temperature),
        max_tokens=int(cfg.llm.max_tokens),
    )

    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if llm.provider == "gemini" and not gemini_api_key:
        raise EnvironmentError("GEMINI_API_KEY is not set")

    logger.info("Configuration loaded successfully")

    return Settings(
        paths=paths,
        indexing=indexing,
        embedding=embedding,
        image_embedding=image_embedding,
        chunking=chunking,
        retrieval=retrieval,
        llm=llm,
        gemini_api_key=gemini_api_key,
    )


settings = load_settings()

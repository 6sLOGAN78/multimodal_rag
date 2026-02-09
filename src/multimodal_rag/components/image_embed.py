from typing import List, Dict, Any

import torch
import open_clip
from PIL import Image

from multimodal_rag.config.config import settings
from multimodal_rag.logger import logger


class ImageEmbedder:
    """
    Image embedding using OpenCLIP ViT-H/14 (1024-dim)
    """

    def __init__(self):
        cfg = settings.image_embedding

        logger.info(
            f"Loading image embedding model | "
            f"model={cfg.pretrained} | device={cfg.device}"
        )

        self.device = torch.device(cfg.device)

        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name=cfg.pretrained,
            pretrained="laion2b_s32b_b79k",
        )

        self.model = self.model.to(self.device)
        self.model.eval()

        logger.info("Image embedding model loaded successfully")

    @torch.no_grad()
    def embed_images(
        self,
        image_items: List[Dict[str, Any]],
    ):
        """
        image_items example:
        {
            "image": PIL.Image OR
            "image_path": str,
            "image_id": str,
            "source": str,
            "page": int (optional)
        }
        """

        embeddings = []
        metadatas = []

        for item in image_items:
            if "image" in item:
                image = item["image"]
            else:
                image = Image.open(item["image_path"]).convert("RGB")

            image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)

            emb = self.model.encode_image(image_tensor)
            emb = emb / emb.norm(dim=-1, keepdim=True)

            embeddings.append(emb.cpu().numpy()[0])

            metadatas.append(
                {
                    k: v for k, v in item.items()
                    if k not in ("image",)
                }
            )

        logger.info(
            f"Generated image embeddings | "
            f"vectors={len(embeddings)} | dim={embeddings[0].shape[0]}"
        )

        return embeddings, metadatas

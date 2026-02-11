from multimodal_rag.components.retrieve import (
    retrieve_text_to_text,
    retrieve_text_to_text_and_image,
)
from multimodal_rag.logger import logger


def main():
    query = "image of attention mechanism in transformers"

    logger.info("===== TEXT → TEXT ONLY =====")

    text_results = retrieve_text_to_text(
        query=query,
        top_k=5,
    )

    for i, r in enumerate(text_results):
        print(f"\n[TEXT RESULT {i+1}]")
        print("Distance:", r["distance"])
        print("Source:", r["metadata"].get("source"))
        print("Page:", r["metadata"].get("page_start"))
        print("Text:", r["document"])

    logger.info("===== TEXT → TEXT + IMAGE =====")

    combined_results = retrieve_text_to_text_and_image(
        query=query,
        top_k_text=5,
        top_k_image=5,
    )

    print("\n====== TEXT RESULTS ======")
    for i, r in enumerate(combined_results["text_results"]):
        print(f"\n[TEXT {i+1}]")
        print("Distance:", r["distance"])
        print("Source:", r["metadata"].get("source"))
        print("Page:", r["metadata"].get("page_start"))
        print("Text:", r["document"][:300], "...")

    print("\n====== IMAGE RESULTS ======")
    for i, r in enumerate(combined_results["image_results"]):
        print(f"\n[IMAGE {i+1}]")
        print("Distance:", r["distance"])
        print("Image ID:", r["metadata"].get("image_id"))
        print("Source:", r["metadata"].get("source"))
        print("Page:", r["metadata"].get("page"))


if __name__ == "__main__":
    main()

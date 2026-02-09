from setuptools import setup, find_packages
from pathlib import Path

BASE_DIR = Path(__file__).parent

setup(
    name="multimodal-rag",
    version="0.1.0",
    author="Ayush Maurya",
    author_email="ayush_2401ct23@iitp.ac.in",
    description="Multimodal RAG pipeline",
    long_description=(BASE_DIR / "readme.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    url="https://github.com/6sLOGAN78/multimodal_rag",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "pyyaml",
        "pymupdf",
        "python-docx",
        "pillow",
        "langchain",
    ],
    python_requires=">=3.9",
)

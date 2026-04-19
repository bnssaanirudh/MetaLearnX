"""
MetaLearnX — Semantic Dataset Embedding
Uses local SentenceTransformers to generate dense text embeddings for dataset descriptions.
Ensures we capture the semantic meaning of the dataset alongside statistical meta-features.
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
from loguru import logger

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_AVAILABLE = True
except ImportError:
    SENTENCE_AVAILABLE = False


class DatasetSemanticEmbedder:
    """
    Generates text embeddings for dataset descriptions.
    Uses 'all-MiniLM-L6-v2' (384-dimensional) by default for efficiency.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self._load_model()

    def _load_model(self) -> None:
        if not SENTENCE_AVAILABLE:
            logger.warning("sentence-transformers not installed. Semantic embeddings disabled.")
            return

        try:
            # Setting environment variables to suppress tokenizer warnings and parallel issues
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
            logger.info(f"Loading SentenceTransformer model: {self.model_name}...")
            self.model = SentenceTransformer(self.model_name)
        except Exception as e:
            logger.warning(f"Failed to load Semantic Embedding model: {e}")

    def embed_text(self, description: Optional[str]) -> np.ndarray:
        """
        Embed descriptive text into a dense vector.
        Returns a 384-dimensional zero vector if description is empty or model unavailable.
        """
        # Default dimension for all-MiniLM-L6-v2
        default_dim = 384
        
        if not description or not str(description).strip() or self.model is None:
            return np.zeros(default_dim, dtype=np.float32)

        try:
            embedding = self.model.encode(str(description).strip())
            return np.array(embedding, dtype=np.float32)
        except Exception as e:
            logger.warning(f"Failed to generate semantic embedding: {e}")
            return np.zeros(default_dim, dtype=np.float32)


# Singleton
_embedder_instance: Optional[DatasetSemanticEmbedder] = None

def get_semantic_embedder() -> DatasetSemanticEmbedder:
    global _embedder_instance
    if _embedder_instance is None:
        _embedder_instance = DatasetSemanticEmbedder()
    return _embedder_instance

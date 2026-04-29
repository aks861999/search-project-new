"""
pytest configuration for wurth-search.

Stubs out heavy ML libraries (torch, sentence_transformers) before any
test module is imported.  This lets the test suite run without GPU drivers
or a multi-GB torch installation while keeping full coverage of all
business logic.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

# ── Ensure repo root is on the Python path ────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ── Stub out torch (requires libcudnn which isn't present in CI) ──────────────
def _make_torch_stub() -> types.ModuleType:
    torch = types.ModuleType("torch")
    # nn sub-module
    nn = types.ModuleType("torch.nn")
    nn.Module = object
    torch.nn = nn  # type: ignore[attr-defined]
    sys.modules["torch.nn"] = nn
    # cuda stub
    cuda = types.ModuleType("torch.cuda")
    cuda.is_available = lambda: False  # type: ignore[attr-defined]
    torch.cuda = cuda  # type: ignore[attr-defined]
    sys.modules["torch.cuda"] = cuda
    return torch


if "torch" not in sys.modules:
    sys.modules["torch"] = _make_torch_stub()
else:
    # Already imported (e.g. partially) — patch is_available to avoid GPU errors
    try:
        import torch  # noqa: F401
    except ImportError:
        sys.modules["torch"] = _make_torch_stub()


# ── Stub out sentence_transformers ────────────────────────────────────────────
class _FakeSentenceTransformer:
    """Minimal stand-in for SentenceTransformer used in unit tests."""

    def __init__(self, model_name: str = "", **kwargs):
        self.model_name = model_name

    def encode(self, texts, batch_size=32, normalize_embeddings=False,
               show_progress_bar=False, **kwargs):
        import numpy as np

        if isinstance(texts, str):
            return np.zeros(384, dtype="float32")
        return np.zeros((len(texts), 384), dtype="float32")


if "sentence_transformers" not in sys.modules:
    st_mod = types.ModuleType("sentence_transformers")
    st_mod.SentenceTransformer = _FakeSentenceTransformer  # type: ignore[attr-defined]
    sys.modules["sentence_transformers"] = st_mod


# ── Stub out datasets (HuggingFace) ───────────────────────────────────────────
if "datasets" not in sys.modules:
    ds_mod = types.ModuleType("datasets")
    ds_mod.load_dataset = MagicMock(return_value=[])  # type: ignore[attr-defined]
    sys.modules["datasets"] = ds_mod


# ── Stub out langchain_google_genai (optional dep) ───────────────────────────
if "langchain_google_genai" not in sys.modules:
    lgc = types.ModuleType("langchain_google_genai")
    lgc.ChatGoogleGenerativeAI = MagicMock  # type: ignore[attr-defined]
    sys.modules["langchain_google_genai"] = lgc


# ── Stub out langchain_huggingface (optional dep) ────────────────────────────
if "langchain_huggingface" not in sys.modules:
    lhf = types.ModuleType("langchain_huggingface")
    lhf.HuggingFaceEmbeddings = MagicMock  # type: ignore[attr-defined]
    sys.modules["langchain_huggingface"] = lhf


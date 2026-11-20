# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Text embeddings for the skill library (coarse recall, de-duplication, merge clustering).

The embedder is frozen: retrieval is not trained. Two backends:

- ``hash``: hashed word n-grams. No model, no download, deterministic; used by the unit tests and
  the CPU smoke runs. It only sees shared words.
- any other name: a Hugging Face sentence-embedding model (default ``BAAI/bge-small-en-v1.5``),
  CLS pooling, L2-normalized.

Vectors are unit length, so a dot product is the cosine similarity.
"""

from __future__ import annotations

import math
import re
import zlib
from typing import Sequence

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset(
    "a an the to of in on at for and or is are be it this that with as by from then if you your i".split()
)


def tokenize(text: str) -> list[str]:
    """Lowercased alphanumeric tokens without stopwords and bare numbers (``mug 2`` and ``mug 1`` match)."""
    return [t for t in _TOKEN_RE.findall((text or "").lower()) if t not in _STOPWORDS and not t.isdigit()]


class HashEmbedder:
    """Bag of hashed word uni- and bigrams with sub-linear term frequency."""

    name = "hash"

    def __init__(self, dim: int = 1024):
        self.dim = dim

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        out = []
        for text in texts:
            toks = tokenize(text)
            grams = toks + [f"{a} {b}" for a, b in zip(toks, toks[1:], strict=False)]
            counts: dict[int, int] = {}
            for g in grams:
                idx = zlib.crc32(g.encode()) % self.dim
                counts[idx] = counts.get(idx, 0) + 1
            vec = [0.0] * self.dim
            for idx, c in counts.items():
                vec[idx] = 1.0 + math.log(c)
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            out.append([v / norm for v in vec])
        return out


class HFEmbedder:
    """Sentence embeddings from a Hugging Face encoder (CLS pooling), on CPU."""

    def __init__(self, model_name: str, max_length: int = 256, num_threads: int = 4):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.name = model_name
        self.torch = torch
        torch.set_num_threads(num_threads)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).eval()
        self.max_length = max_length

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        torch = self.torch
        out: list[list[float]] = []
        for start in range(0, len(texts), 32):
            batch = self.tokenizer(
                list(texts[start : start + 32]),
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            with torch.no_grad():
                hidden = self.model(**batch).last_hidden_state[:, 0]
            out.extend(torch.nn.functional.normalize(hidden, dim=-1).tolist())
        return out


def make_embedder(name: str = "hash"):
    return HashEmbedder() if name in ("", "hash", None) else HFEmbedder(name)


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    return float(sum(x * y for x, y in zip(a, b, strict=True)))

"""RAG (Retrieval-Augmented Generation) and continuous learning.

Supports local document import, vectorization, retrieval-augmented generation,
style memory, and error correction learning.
All processing is local with full-precision embeddings.
"""

import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np

from engine.utils.hashing import hash_bytes
from engine.utils.storage import ensure_dir

if TYPE_CHECKING:
    from engine.core.inference import InferenceEngine

logger = logging.getLogger(__name__)


@dataclass
class DocumentChunk:
    doc_id: str
    chunk_index: int
    text: str
    embedding: Optional[np.ndarray] = None
    metadata: Dict = field(default_factory=dict)


@dataclass
class KnowledgeBase:
    name: str
    base_path: str
    documents: Dict[str, List[DocumentChunk]] = field(default_factory=dict)
    faiss_index: object = None
    chunk_map: Dict[int, DocumentChunk] = field(default_factory=dict)
    next_id: int = 0


class RAGManager:
    """Manages knowledge bases, document import, and retrieval-augmented generation."""

    def __init__(
        self,
        base_path: str = "~/.cache/ai_engine/rag",
        chunk_size: int = 512,
        top_k: int = 5,
    ):
        self._base_path = os.path.expanduser(base_path)
        self._chunk_size = chunk_size
        self._top_k = top_k
        self._kbs: Dict[str, KnowledgeBase] = {}
        self._engine: Optional["InferenceEngine"] = None
        self._embedding_dim = 4096

    def set_engine(self, engine: "InferenceEngine"):
        self._engine = engine

    def create_kb(self, name: str) -> KnowledgeBase:
        """Create a new knowledge base."""
        kb_path = os.path.join(self._base_path, name)
        ensure_dir(kb_path)

        try:
            import faiss
            index = faiss.IndexFlatIP(self._embedding_dim)
        except ImportError:
            logger.warning("faiss-cpu 未安装, 知识库搜索不可用")
            index = None

        kb = KnowledgeBase(name=name, base_path=kb_path, faiss_index=index)
        self._kbs[name] = kb
        logger.info(f"知识库已创建: {name}")
        return kb

    def get_or_create_kb(self, name: str = "default") -> KnowledgeBase:
        if name in self._kbs:
            return self._kbs[name]
        return self.create_kb(name)

    def add_document(
        self,
        filepath: str,
        kb_name: str = "default",
    ) -> int:
        """Import a document into the knowledge base. Returns number of chunks."""
        kb = self.get_or_create_kb(kb_name)
        path = Path(os.path.expanduser(filepath))

        if not path.exists():
            raise FileNotFoundError(f"文件不存在: {filepath}")

        text = self._read_document(path)
        if not text:
            return 0

        doc_id = path.name
        chunks = self._split_text(text, self._chunk_size)
        doc_chunks = []

        for i, chunk_text in enumerate(chunks):
            chunk = DocumentChunk(
                doc_id=doc_id,
                chunk_index=i,
                text=chunk_text,
                metadata={"source": str(path), "added_at": time.time()},
            )

            if self._engine is not None and self._engine.is_loaded:
                try:
                    tokens = self._engine.tokenize(chunk_text, add_bos=False)
                    embedding = self._engine.get_embedding(tokens)
                    chunk.embedding = embedding

                    if kb.faiss_index is not None:
                        vec = embedding.reshape(1, -1).astype(np.float32)
                        norm = np.linalg.norm(vec)
                        if norm > 0:
                            vec = vec / norm
                        kb.faiss_index.add(vec)
                        kb.chunk_map[kb.next_id] = chunk
                        kb.next_id += 1
                except Exception as e:
                    logger.warning(f"块向量化失败 ({i}): {e}")

            doc_chunks.append(chunk)

        kb.documents[doc_id] = doc_chunks
        logger.info(f"文档导入完成: {doc_id} ({len(doc_chunks)} 块)")
        return len(doc_chunks)

    def search(
        self,
        query: str,
        kb_name: str = "default",
        top_k: Optional[int] = None,
    ) -> List[DocumentChunk]:
        """Search knowledge base for relevant chunks."""
        kb = self._kbs.get(kb_name)
        if kb is None or kb.faiss_index is None:
            return []
        if self._engine is None or not self._engine.is_loaded:
            return []

        k = top_k or self._top_k

        try:
            tokens = self._engine.tokenize(query, add_bos=False)
            embedding = self._engine.get_embedding(tokens)
            vec = embedding.reshape(1, -1).astype(np.float32)
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm

            scores, indices = kb.faiss_index.search(vec, k)
            results = []
            for i, idx in enumerate(indices[0]):
                if idx >= 0 and idx in kb.chunk_map:
                    chunk = kb.chunk_map[idx]
                    chunk.metadata["score"] = float(scores[0][i])
                    results.append(chunk)
            return results
        except Exception as e:
            logger.warning(f"知识库搜索失败: {e}")
            return []

    def build_augmented_prompt(
        self,
        query: str,
        kb_name: str = "default",
    ) -> str:
        """Build a retrieval-augmented prompt with relevant context."""
        chunks = self.search(query, kb_name)
        if not chunks:
            return query

        context_parts = []
        for i, chunk in enumerate(chunks):
            score = chunk.metadata.get("score", 0)
            context_parts.append(
                f"[参考文档 {i+1} (相关度: {score:.2f})]:\n{chunk.text}"
            )

        context = "\n\n".join(context_parts)
        return (
            f"请基于以下参考资料回答问题。\n\n"
            f"{context}\n\n"
            f"问题: {query}"
        )

    def list_documents(self, kb_name: str = "default") -> List[Dict]:
        kb = self._kbs.get(kb_name)
        if kb is None:
            return []
        return [
            {
                "doc_id": doc_id,
                "chunks": len(chunks),
                "source": chunks[0].metadata.get("source", "") if chunks else "",
            }
            for doc_id, chunks in kb.documents.items()
        ]

    def remove_document(self, doc_id: str, kb_name: str = "default") -> bool:
        kb = self._kbs.get(kb_name)
        if kb is None or doc_id not in kb.documents:
            return False
        del kb.documents[doc_id]
        logger.info(f"文档已移除: {doc_id}")
        return True

    @staticmethod
    def _read_document(path: Path) -> str:
        suffix = path.suffix.lower()
        try:
            if suffix in (".txt", ".md", ".py", ".js", ".json", ".csv", ".xml", ".html"):
                return path.read_text(encoding="utf-8", errors="replace")
            elif suffix == ".pdf":
                try:
                    import fitz  # PyMuPDF
                    doc = fitz.open(str(path))
                    text = ""
                    for page in doc:
                        text += page.get_text()
                    return text
                except ImportError:
                    logger.warning("PyMuPDF 未安装, PDF 导入不可用")
                    return ""
            else:
                return path.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            logger.warning(f"文档读取失败: {e}")
            return ""

    @staticmethod
    def _split_text(text: str, chunk_size: int) -> List[str]:
        """Split text into chunks by approximate token count (chars / 2 ≈ tokens)."""
        char_size = chunk_size * 2
        chunks = []
        paragraphs = text.split("\n\n")
        current_chunk = []
        current_len = 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            if current_len + len(para) > char_size and current_chunk:
                chunks.append("\n\n".join(current_chunk))
                current_chunk = []
                current_len = 0
            current_chunk.append(para)
            current_len += len(para)

        if current_chunk:
            chunks.append("\n\n".join(current_chunk))

        return chunks


class StyleMemory:
    """Learns and adapts to user's preferred output style."""

    def __init__(self, config_path: str = "~/.cache/ai_engine/style_memory.json"):
        self._path = os.path.expanduser(config_path)
        self._preferences: Dict[str, Any] = {
            "avg_response_length": 0,
            "preferred_language": "auto",
            "formal_level": "neutral",
            "total_interactions": 0,
        }
        self._load()

    def _load(self):
        import json
        if os.path.exists(self._path):
            try:
                with open(self._path, "r") as f:
                    self._preferences.update(json.load(f))
            except Exception:
                pass

    def _save(self):
        import json
        ensure_dir(os.path.dirname(self._path))
        with open(self._path, "w") as f:
            json.dump(self._preferences, f, ensure_ascii=False, indent=2)

    def record_interaction(self, user_input: str, response: str):
        n = self._preferences["total_interactions"]
        avg = self._preferences["avg_response_length"]
        self._preferences["avg_response_length"] = (avg * n + len(response)) / (n + 1)
        self._preferences["total_interactions"] = n + 1
        self._save()

    @property
    def preferences(self) -> Dict:
        return self._preferences.copy()


class ErrorCorrectionMemory:
    """Stores user corrections to avoid repeating the same errors."""

    def __init__(self, config_path: str = "~/.cache/ai_engine/corrections.json"):
        self._path = os.path.expanduser(config_path)
        self._corrections: List[Dict] = []
        self._load()

    def _load(self):
        import json
        if os.path.exists(self._path):
            try:
                with open(self._path, "r") as f:
                    self._corrections = json.load(f)
            except Exception:
                pass

    def _save(self):
        import json
        ensure_dir(os.path.dirname(self._path))
        with open(self._path, "w") as f:
            json.dump(self._corrections, f, ensure_ascii=False, indent=2)

    def add_correction(self, context: str, wrong_output: str, correct_output: str):
        self._corrections.append({
            "context": context[:200],
            "wrong": wrong_output[:500],
            "correct": correct_output[:500],
            "timestamp": time.time(),
        })
        if len(self._corrections) > 1000:
            self._corrections = self._corrections[-1000:]
        self._save()

    def get_relevant_corrections(self, context: str, max_results: int = 3) -> List[Dict]:
        context_lower = context.lower()
        scored = []
        for c in self._corrections:
            overlap = len(set(context_lower.split()) & set(c["context"].lower().split()))
            if overlap > 0:
                scored.append((overlap, c))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [c for _, c in scored[:max_results]]

    @property
    def correction_count(self) -> int:
        return len(self._corrections)

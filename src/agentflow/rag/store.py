"""Vector stores: a local hybrid index and an optional Chroma adapter."""

from abc import ABC, abstractmethod
import json
import math
from pathlib import Path
import threading

from agentflow.errors import AgentFlowError
from agentflow.rag.embeddings import cosine, embed, tokenize
from agentflow.schemas import Document, ScoredDocument


class VectorStore(ABC):
    @abstractmethod
    def add(self, docs: list[Document]) -> None:
        raise NotImplementedError

    @abstractmethod
    def query(self, text: str, k: int = 4) -> list[ScoredDocument]:
        raise NotImplementedError

    @abstractmethod
    def count(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def list_documents(self) -> list[Document]:
        raise NotImplementedError


class LocalVectorStore(VectorStore):
    """BM25 blended with hashed-token cosine similarity.

    Documents persist as JSON so a restart keeps ingested notes. The blend is
    what makes short policy questions land on the right memo without a
    downloaded embedding model.
    """

    def __init__(self, path: str) -> None:
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.file = self.path / "documents.json"
        self._lock = threading.Lock()
        self.docs: list[Document] = []
        self._load()

    def add(self, docs: list[Document]) -> None:
        with self._lock:
            by_id = {doc.id: doc for doc in self.docs}
            for doc in docs:
                by_id[doc.id] = doc
            self.docs = list(by_id.values())
            self._save()

    def count(self) -> int:
        return len(self.docs)

    def list_documents(self) -> list[Document]:
        return list(self.docs)

    def query(self, text: str, k: int = 4) -> list[ScoredDocument]:
        if not self.docs:
            return []
        lexical = self._bm25(tokenize(text))
        query_vector = embed(text)
        scored: list[ScoredDocument] = []
        for index, doc in enumerate(self.docs):
            document_vector = embed(f"{doc.title}\n{doc.title}\n{doc.text}")
            blended = 0.65 * lexical[index] + 0.35 * max(0.0, cosine(query_vector, document_vector))
            if blended > 0:
                scored.append(ScoredDocument(document=doc, score=blended))
        scored.sort(key=lambda item: item.score, reverse=True)
        return scored[:k]

    def _bm25(self, query_tokens: list[str]) -> list[float]:
        documents = [tokenize(f"{doc.title} {doc.title} {doc.text}") for doc in self.docs]
        count = len(documents)
        average = sum(len(tokens) for tokens in documents) / count if count else 1
        document_frequency: dict[str, int] = {}
        for tokens in documents:
            for term in set(tokens):
                document_frequency[term] = document_frequency.get(term, 0) + 1
        scores: list[float] = []
        k1 = 1.5
        b = 0.75
        terms = list(dict.fromkeys(query_tokens))
        for tokens in documents:
            frequency: dict[str, int] = {}
            for term in tokens:
                frequency[term] = frequency.get(term, 0) + 1
            length = len(tokens) or 1
            score = 0.0
            for term in terms:
                if term not in frequency:
                    continue
                seen = document_frequency[term]
                idf = math.log(1 + (count - seen + 0.5) / (seen + 0.5))
                freq = frequency[term]
                score += idf * (freq * (k1 + 1)) / (freq + k1 * (1 - b + b * length / average))
            scores.append(score)
        peak = max(scores) if scores else 0
        if peak <= 0:
            return [0.0 for _ in scores]
        return [score / peak for score in scores]

    def _load(self) -> None:
        if not self.file.exists():
            return
        raw = json.loads(self.file.read_text(encoding="utf-8"))
        self.docs = [Document.model_validate(item) for item in raw]

    def _save(self) -> None:
        payload = [doc.model_dump() for doc in self.docs]
        self.file.write_text(json.dumps(payload, indent=2), encoding="utf-8")


class ChromaVectorStore(VectorStore):
    """Adapter over a Chroma collection.

    Tests can inject a collection. At runtime the collection is created with
    the same hashing embedder, so no model download is required.
    """

    def __init__(self, path: str, collection=None) -> None:
        if collection is None:
            try:
                import chromadb
            except ImportError as exc:
                raise AgentFlowError(
                    "Chroma is not installed. Install the chroma extra or set AGENTFLOW_VECTOR_BACKEND=local."
                ) from exc
            client = chromadb.PersistentClient(path=path)
            collection = client.get_or_create_collection(
                name="agentflow",
                embedding_function=_chroma_embedding(),
            )
        self.collection = collection

    def add(self, docs: list[Document]) -> None:
        if not docs:
            return
        kwargs = {
            "ids": [doc.id for doc in docs],
            "documents": [doc.text for doc in docs],
            "metadatas": [{"title": doc.title, "source": doc.source} for doc in docs],
        }
        upsert = getattr(self.collection, "upsert", None)
        if upsert is not None:
            upsert(**kwargs)
        else:
            self.collection.add(**kwargs)

    def query(self, text: str, k: int = 4) -> list[ScoredDocument]:
        if self.count() == 0:
            return []
        found = self.collection.query(query_texts=[text], n_results=min(k, self.count()))
        ids = (found.get("ids") or [[]])[0]
        documents = (found.get("documents") or [[]])[0]
        metadatas = (found.get("metadatas") or [[]])[0]
        distances = (found.get("distances") or [[]])[0]
        scored: list[ScoredDocument] = []
        for index, doc_id in enumerate(ids):
            meta = metadatas[index] if index < len(metadatas) and metadatas[index] else {}
            distance = distances[index] if index < len(distances) else index
            score = 1.0 / (1.0 + float(distance))
            scored.append(
                ScoredDocument(
                    document=Document(
                        id=doc_id,
                        title=str(meta.get("title") or doc_id),
                        text=documents[index] if index < len(documents) else "",
                        source=str(meta.get("source") or ""),
                    ),
                    score=score,
                )
            )
        return scored

    def count(self) -> int:
        return int(self.collection.count())

    def list_documents(self) -> list[Document]:
        found = self.collection.get()
        docs: list[Document] = []
        ids = found.get("ids") or []
        documents = found.get("documents") or []
        metadatas = found.get("metadatas") or []
        for index, doc_id in enumerate(ids):
            meta = metadatas[index] if index < len(metadatas) and metadatas[index] else {}
            docs.append(
                Document(
                    id=doc_id,
                    title=str(meta.get("title") or doc_id),
                    text=documents[index] if index < len(documents) else "",
                    source=str(meta.get("source") or ""),
                )
            )
        return docs


def _chroma_embedding():
    class HashEmbedding:
        def __call__(self, input):  # noqa: A002 - Chroma's parameter name
            return [embed(item) for item in input]

        def name(self) -> str:
            return "agentflow-hash"

        def embed_query(self, input):  # noqa: A002
            return self.__call__(input)

    try:
        from chromadb.utils.embedding_functions import EmbeddingFunction
    except Exception:
        return HashEmbedding()

    class Registered(HashEmbedding, EmbeddingFunction):
        pass

    return Registered()


def build_store(backend: str, path: str) -> VectorStore:
    if backend == "local":
        return LocalVectorStore(path)
    if backend == "chroma":
        return ChromaVectorStore(path)
    raise AgentFlowError("vector_backend must be local or chroma")

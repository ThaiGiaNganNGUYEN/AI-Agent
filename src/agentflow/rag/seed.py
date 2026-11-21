"""Load the packaged treasury notes into the vector store."""

from importlib.resources import files

from agentflow.schemas import Document


def packaged_documents() -> list[Document]:
    root = files("agentflow.resources").joinpath("knowledge")
    documents: list[Document] = []
    for item in sorted(root.iterdir(), key=lambda entry: entry.name):
        if not str(item.name).endswith(".md"):
            continue
        text = item.read_text(encoding="utf-8")
        title = next((line[2:].strip() for line in text.splitlines() if line.startswith("# ")), item.name)
        documents.append(Document(id=str(item.name), title=title, text=text, source=f"knowledge/{item.name}"))
    return documents

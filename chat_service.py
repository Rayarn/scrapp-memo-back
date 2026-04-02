import os
import chromadb
from chromadb.utils import embedding_functions
from openai import OpenAI

CHROMA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "chroma")
COLLECTION_NAME = "memoires"


def _get_collection(api_key: str):
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    ef = embedding_functions.OpenAIEmbeddingFunction(api_key=api_key, model_name="text-embedding-3-small")
    return client.get_or_create_collection(name=COLLECTION_NAME, embedding_function=ef)


def index_memoires(memoires: list[dict], api_key: str) -> dict:
    """Index selected memoirs into ChromaDB. Returns stats."""
    os.makedirs(CHROMA_DIR, exist_ok=True)
    collection = _get_collection(api_key)

    # Reset collection to re-index cleanly
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    ef = embedding_functions.OpenAIEmbeddingFunction(api_key=api_key, model_name="text-embedding-3-small")
    collection = client.get_or_create_collection(name=COLLECTION_NAME, embedding_function=ef)

    docs, ids, metas = [], [], []
    for i, m in enumerate(memoires):
        resume = str(m.get("resume_fr", "")).strip()
        if not resume or resume in ("nan", "None", ""):
            resume = f"Mémoire sans résumé : {m.get('titre', '')}"
        titre = str(m.get("titre", ""))
        auteur = str(m.get("auteur", ""))
        annee = str(m.get("annee", ""))
        doc = f"[{annee}] {titre} — {auteur}\n\n{resume}"
        docs.append(doc[:2000])
        ids.append(f"mem_{i}")
        metas.append({"titre": titre[:200], "auteur": auteur[:100], "annee": annee, "lien": str(m.get("lien", ""))})

    if docs:
        # Batch in chunks of 100
        for start in range(0, len(docs), 100):
            collection.add(documents=docs[start:start+100], ids=ids[start:start+100], metadatas=metas[start:start+100])

    return {"indexed": len(docs), "collection": COLLECTION_NAME}


def get_index_status(api_key: str) -> dict:
    """Return how many documents are indexed."""
    try:
        collection = _get_collection(api_key)
        count = collection.count()
        return {"indexed": count, "ready": count > 0}
    except Exception:
        return {"indexed": 0, "ready": False}


def delete_index(api_key: str):
    """Delete the ChromaDB collection."""
    try:
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass


def chat(message: str, history: list[dict], api_key: str, model: str) -> str:
    """RAG chat: retrieve relevant memoir chunks, then generate response."""
    collection = _get_collection(api_key)
    count = collection.count()

    context = ""
    if count > 0:
        results = collection.query(query_texts=[message], n_results=min(5, count))
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        chunks = []
        for doc, meta in zip(docs, metas):
            chunks.append(f"**[{meta.get('annee','')}] {meta.get('titre','')}** — {meta.get('auteur','')}\n{doc}")
        context = "\n\n---\n\n".join(chunks)

    system_prompt = """Tu es un assistant expert en actuariat, spécialisé dans l'analyse de mémoires d'actuariat de l'Institut des Actuaires.
Tu aides les étudiants à explorer le corpus de mémoires, identifier des tendances, comprendre les méthodes utilisées, et trouver des pistes de recherche originales.
Réponds en français, de manière claire et académique."""

    if context:
        system_prompt += f"\n\nVoici les mémoires les plus pertinents pour cette question :\n\n{context}"
    else:
        system_prompt += "\n\nAucun corpus n'est actuellement indexé. Réponds à partir de ta connaissance générale en actuariat."

    messages = [{"role": "system", "content": system_prompt}]
    for h in history[-10:]:  # Keep last 10 messages to stay within token limits
        messages.append({"role": h["role"], "content": h["content"]})
    messages.append({"role": "user", "content": message})

    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        max_tokens=1500,
        temperature=0.3,
        messages=messages,
    )
    return response.choices[0].message.content.strip()

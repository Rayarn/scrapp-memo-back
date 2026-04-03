import asyncio
import io
import json
import os
import re
from datetime import datetime
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

import analyser_service, chat_service, data_service, scraper_service

load_dotenv()

app = FastAPI(title="Mémoires Actuariat API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Le "*" autorise ton futur site Vercel à appeler l'API
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Models ────────────────────────────────────────────────────────────────────

class ScrapeParams(BaseModel):
    annee_min: int = 2010
    annee_max: int = 2024
    mots_cles: list[str] = ["Lee-Carter", "Gompertz"]
    mots_exclusion: list[str] = []
    extract_details: bool = True
    delay: int = 2
    timeout: int = 30


class SubjectParams(BaseModel):
    titre: str = ""
    description: str
    mots_cles: list[str] = []
    angle: str = ""
    api_key: str
    model: str = "gpt-4o"


class ClassifyParams(BaseModel):
    api_key: str
    model: str = "gpt-4o"
    max_tokens: int = 500
    temperature: float = 0.0
    min_resume_length: int = 100


class ChatIndexParams(BaseModel):
    memoire_ids: list[int]
    api_key: str


class ChatMessage(BaseModel):
    message: str
    history: list[dict] = []
    api_key: str
    model: str = "gpt-4o"


class KeywordGenParams(BaseModel):
    description: str
    api_key: str
    model: str = "gpt-4o"


# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/api/health")
def health():
    return {"status": "ok", "version": "2.0.0"}


# ── Keyword generation ────────────────────────────────────────────────────────

@app.post("/api/keywords/generate")
async def generate_keywords(params: KeywordGenParams):
    """Generate inclusion and exclusion keywords from a free-text subject description."""
    from openai import OpenAI
    if not params.description.strip():
        raise HTTPException(400, "Description vide")
    try:
        client = OpenAI(api_key=params.api_key)
        response = client.chat.completions.create(
            model=params.model,
            max_tokens=400,
            temperature=0.0,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": "Tu es un expert en actuariat. Tu réponds UNIQUEMENT en JSON valide.",
                },
                {
                    "role": "user",
                    "content": f"""Un étudiant actuaire veut scraper la base de mémoires de l'Institut des Actuaires.
Il décrit son sujet ainsi :

\"\"\"{params.description}\"\"\"

Génère deux listes de mots-clés pour paramétrer la recherche :
1. **mots_inclusion** : mots-clés à rechercher dans la base (termes techniques, modèles, méthodes, concepts liés au sujet). Entre 3 et 8 mots-clés pertinents.
2. **mots_exclusion** : mots présents dans les TITRES de mémoires qui signaleraient que le mémoire est hors-sujet (autres branches de l'assurance, domaines sans lien). Entre 5 et 15 mots.

Réponds avec ce JSON exact :
{{
  "mots_inclusion": ["mot1", "mot2", ...],
  "mots_exclusion": ["mot1", "mot2", ...],
  "explication": "Courte justification des choix en 1-2 phrases"
}}""",
                },
            ],
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        raise HTTPException(500, str(e))


# ── Scraping ──────────────────────────────────────────────────────────────────

@app.post("/api/scrape")
async def scrape(params: ScrapeParams):
    async def generate():
        loop = asyncio.get_event_loop()
        queue: asyncio.Queue = asyncio.Queue()

        def run_scrape():
            for event in scraper_service.scrape(params.model_dump()):
                loop.call_soon_threadsafe(queue.put_nowait, event)
            loop.call_soon_threadsafe(queue.put_nowait, None)

        future = loop.run_in_executor(None, run_scrape)

        while True:
            event = await queue.get()
            if event is None:
                break
            if event.get("type") == "done":
                data_service.save_scraped(event.get("data", []))
            yield {"data": json.dumps(event)}

        await future

    return EventSourceResponse(generate())


# ── Data ──────────────────────────────────────────────────────────────────────

@app.get("/api/data/scraped")
def get_scraped():
    return {"data": data_service.load_scraped()}


@app.get("/api/data/analysed")
def get_analysed():
    return {"data": data_service.load_analysed()}


@app.delete("/api/data/scraped")
def clear_scraped():
    data_service.save_scraped([])
    return {"ok": True}


@app.delete("/api/data/analysed")
def clear_analysed():
    data_service.save_analysed([])
    return {"ok": True}


# ── Export ────────────────────────────────────────────────────────────────────

ILLEGAL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def _clean_df(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str).apply(lambda x: ILLEGAL.sub("", x))
    return df


@app.get("/api/export/{source}/{fmt}")
def export(source: str, fmt: str):
    data = data_service.load_scraped() if source == "scraped" else data_service.load_analysed()
    if not data:
        raise HTTPException(404, "Aucune donnée disponible")
    df = pd.DataFrame(data)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    if fmt == "csv":
        buf = df.to_csv(index=False, encoding="utf-8-sig")
        return StreamingResponse(
            io.BytesIO(buf.encode("utf-8-sig")),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename=memoires_{source}_{ts}.csv"},
        )
    elif fmt == "excel":
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as writer:
            _clean_df(df.copy()).to_excel(writer, index=False, sheet_name="Données")
        buf.seek(0)
        return StreamingResponse(
            buf,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": f"attachment; filename=memoires_{source}_{ts}.xlsx"},
        )
    elif fmt == "json":
        buf = df.to_json(orient="records", force_ascii=False, indent=2)
        return StreamingResponse(
            io.BytesIO(buf.encode("utf-8")),
            media_type="application/json",
            headers={"Content-Disposition": f"attachment; filename=memoires_{source}_{ts}.json"},
        )
    else:
        raise HTTPException(400, f"Format inconnu: {fmt}")


# ── Analysis ──────────────────────────────────────────────────────────────────

@app.post("/api/analyse/subject")
async def analyse_subject(params: SubjectParams):
    corpus = data_service.load_scraped()
    if not corpus:
        raise HTTPException(400, "Aucune donnée scrapée disponible")
    try:
        result = analyser_service.analyse_subject(
            corpus,
            {"titre": params.titre, "description": params.description, "mots_cles": params.mots_cles, "angle": params.angle},
            params.api_key,
            params.model,
        )
        return result
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/api/analyse/classify")
async def classify(params: ClassifyParams):
    corpus = data_service.load_scraped()
    if not corpus:
        raise HTTPException(400, "Aucune donnée scrapée disponible")

    async def generate():
        loop = asyncio.get_event_loop()
        queue: asyncio.Queue = asyncio.Queue()

        def run_classify():
            for event in analyser_service.classify_corpus(
                corpus, params.api_key, params.model,
                params.max_tokens, params.temperature, params.min_resume_length,
            ):
                loop.call_soon_threadsafe(queue.put_nowait, event)
            loop.call_soon_threadsafe(queue.put_nowait, None)

        future = loop.run_in_executor(None, run_classify)

        while True:
            event = await queue.get()
            if event is None:
                break
            if event.get("type") == "done":
                data_service.save_analysed(event.get("data", []))
            yield {"data": json.dumps(event)}

        await future

    return EventSourceResponse(generate())


# ── Chat / RAG ────────────────────────────────────────────────────────────────

@app.post("/api/chat/index")
async def index_chat(params: ChatIndexParams):
    scraped = data_service.load_scraped()
    if not scraped:
        raise HTTPException(400, "Aucune donnée scrapée disponible")
    selected = [scraped[i] for i in params.memoire_ids if i < len(scraped)]
    if not selected:
        raise HTTPException(400, "Aucun mémoire sélectionné")
    try:
        result = await asyncio.get_event_loop().run_in_executor(
            None, chat_service.index_memoires, selected, params.api_key
        )
        return result
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/api/chat/status")
async def chat_status(api_key: str):
    try:
        status = await asyncio.get_event_loop().run_in_executor(
            None, chat_service.get_index_status, api_key
        )
        return status
    except Exception:
        return {"indexed": 0, "ready": False}


@app.delete("/api/chat/index")
async def delete_index(api_key: str):
    await asyncio.get_event_loop().run_in_executor(None, chat_service.delete_index, api_key)
    return {"ok": True}


@app.post("/api/chat/message")
async def chat_message(params: ChatMessage):
    try:
        response = await asyncio.get_event_loop().run_in_executor(
            None, chat_service.chat, params.message, params.history, params.api_key, params.model
        )
        return {"response": response}
    except Exception as e:
        raise HTTPException(500, str(e))

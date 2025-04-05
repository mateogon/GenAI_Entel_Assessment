"""
Propósito: Define y expone los endpoints de la API para búsqueda, análisis de temas y clasificación de transcripciones.
Orquesta la interacción entre FastAPI, Qdrant y OpenAI (async), y maneja la conexión a la base de datos vectorial.
"""

import os
import sys
import asyncio
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional

# Permite importar módulos desde la raíz del proyecto
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from app.search_logic import semantic_search, keyword_search
from app.analysis_logic import extract_topics, classify_transcript

from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse

# Configuración de conexión a Qdrant
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
COLLECTION_NAME = "transcripts_prod"
qdrant_client_main: Optional[QdrantClient] = None

# -------- Modelos de Entrada/Salida --------
class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    search_type: str = Field("semantic", enum=["semantic", "keyword"])
    top_n: int = Field(5, ge=1, le=20)

class SearchResponseItem(BaseModel):
    transcript_id: str
    score: Optional[float] = None

class SearchResponse(BaseModel):
    results: List[SearchResponseItem]

class AnalysisRequest(BaseModel):
    transcript_id: Optional[str] = None
    text: Optional[str] = None

class TopicResponse(BaseModel):
    topics: List[str]

class ClassificationResponse(BaseModel):
    category: Optional[str]

# -------- Inicialización de la API --------
app = FastAPI(
    title="API Análisis Transcripciones (Qdrant + Async OpenAI)",
    description="API para buscar, extraer temas y clasificar transcripciones.",
    version="0.3.0",
    debug=True
)

@app.on_event("startup")
async def startup_event():
    """Establece conexión con Qdrant al iniciar la API."""
    global qdrant_client_main
    print("Iniciando API y conexión a Qdrant...")
    try:
        qdrant_client_main = QdrantClient(
            host=QDRANT_HOST,
            port=QDRANT_PORT,
            grpc_port=QDRANT_PORT,
            timeout=10
        )
        qdrant_client_main.get_collection(collection_name=COLLECTION_NAME)
        print(f"Conexión a Qdrant ({QDRANT_HOST}:{QDRANT_PORT}) y colección '{COLLECTION_NAME}' verificadas.")
    except UnexpectedResponse as e:
        if e.status_code == 404:
            print(f"Error: La colección '{COLLECTION_NAME}' no existe en Qdrant.")
            print("Ejecuta 'scripts/generate_embeddings_openai.py' para crearla.")
        else:
            print(f"Error crítico al verificar colección Qdrant: {e}")
        qdrant_client_main = None
    except Exception as e:
        print(f"Error al conectar con Qdrant: {e}")
        qdrant_client_main = None
    print("Startup completo.")

@app.on_event("shutdown")
async def shutdown_event():
    """Limpieza en el cierre (si aplica)."""
    if qdrant_client_main:
        print("Cerrando conexión a Qdrant...")

# -------- Endpoints --------
@app.post("/search/", response_model=SearchResponse, tags=["Búsqueda"])
async def search_transcripts(request: SearchRequest):
    """Realiza búsqueda semántica o por palabra clave sobre transcripciones."""
    if qdrant_client_main is None:
        raise HTTPException(status_code=503, detail="Qdrant no disponible.")

    if request.search_type == "semantic":
        results = semantic_search(request.query, top_n=request.top_n)
        return SearchResponse(
            results=[SearchResponseItem(transcript_id=tid, score=score) for tid, score in results]
        )
    elif request.search_type == "keyword":
        results = keyword_search(request.query, top_n=request.top_n)
        return SearchResponse(
            results=[SearchResponseItem(transcript_id=tid) for tid in results]
        )
    else:
        raise HTTPException(status_code=400, detail="Tipo de búsqueda inválido.")

def get_text_for_analysis(request: AnalysisRequest) -> str:
    """Obtiene texto desde Qdrant (por ID) o directamente del request."""
    if qdrant_client_main is None:
        raise HTTPException(status_code=503, detail="Qdrant no disponible.")

    if request.transcript_id:
        transcript_id_to_find = request.transcript_id.strip()
        try:
            # Búsqueda exacta por ID en el payload
            result, _ = qdrant_client_main.scroll(
                collection_name=COLLECTION_NAME,
                scroll_filter=models.Filter(
                    must=[models.FieldCondition(
                        key="original_id",
                        match=models.MatchValue(value=transcript_id_to_find)
                    )]
                ),
                limit=1,
                with_payload=True,
                with_vectors=False
            )

            if not result:
                raise HTTPException(status_code=404, detail=f"ID '{transcript_id_to_find}' no encontrado.")

            payload = result[0].payload or {}
            text = payload.get("full_text")
            if text is None:
                raise HTTPException(status_code=404, detail=f"Texto no disponible para ID '{transcript_id_to_find}'.")

            return text

        except HTTPException:
            raise
        except Exception as e:
            print(f"Error buscando texto en Qdrant: {e}")
            raise HTTPException(status_code=500, detail=f"Error interno al buscar texto para ID '{transcript_id_to_find}'.")

    elif request.text:
        if not isinstance(request.text, str):
            raise HTTPException(status_code=400, detail="El campo 'text' debe ser una cadena.")
        return request.text

    raise HTTPException(status_code=400, detail="Proporciona 'transcript_id' o 'text'.")

@app.post("/analyze/topics/", response_model=TopicResponse, tags=["Análisis"])
async def analyze_topics(request: AnalysisRequest):
    """Extrae temas principales de una transcripción (OpenAI async)."""
    try:
        text = get_text_for_analysis(request)
        topics = await extract_topics(text)
        if topics is None and text:
            raise HTTPException(status_code=502, detail="Error del servicio IA (temas).")
        return TopicResponse(topics=topics or [])
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error en /analyze/topics/: {e}")
        raise HTTPException(status_code=500, detail="Error interno (temas).")

@app.post("/analyze/classify/", response_model=ClassificationResponse, tags=["Análisis"])
async def analyze_classification(request: AnalysisRequest):
    """Clasifica una transcripción en una categoría (OpenAI async)."""
    try:
        text = get_text_for_analysis(request)
        category = await classify_transcript(text)
        if category is None and text:
            raise HTTPException(status_code=502, detail="Error del servicio IA (clasificación).")
        return ClassificationResponse(category=category)
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error en /analyze/classify/: {e}")
        raise HTTPException(status_code=500, detail="Error interno (clasificación).")

@app.get("/status", tags=["Utilidad"])
async def get_status():
    """Devuelve el estado general del servicio y conexión a Qdrant."""
    qdrant_status = "no disponible"
    collection_points = "desconocido"
    if qdrant_client_main:
        try:
            info = qdrant_client_main.get_collection(collection_name=COLLECTION_NAME)
            qdrant_status = "conectado"
            collection_points = info.points_count
        except Exception as e:
            qdrant_status = f"error ({type(e).__name__})"
    return {
        "status": "ok",
        "qdrant_status": qdrant_status,
        "qdrant_collection": COLLECTION_NAME,
        "collection_points_count": collection_points
    }

# --- Para ejecutar manualmente con uvicorn ---
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)

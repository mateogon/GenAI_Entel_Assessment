# app/main.py
import os
import sys
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

# Añadir el directorio raíz al path para importar desde app y scripts
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from app.search_logic import semantic_search, keyword_search, load_search_data
from app.analysis_logic import extract_topics, classify_transcript
from scripts.load_data import load_processed_transcripts

# --- Modelos Pydantic para Validación ---

class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Texto o palabra clave a buscar")
    search_type: str = Field("semantic", enum=["semantic", "keyword"], description="Tipo de búsqueda: 'semantic' o 'keyword'")
    top_n: int = Field(5, ge=1, le=20, description="Número máximo de resultados a devolver")

class SearchResponseItem(BaseModel):
    transcript_id: str
    score: Optional[float] = None  # Score solo para búsqueda semántica

class SearchResponse(BaseModel):
    results: List[SearchResponseItem]

class AnalysisRequest(BaseModel):
    transcript_id: Optional[str] = None
    text: Optional[str] = None

class TopicResponse(BaseModel):
    topics: List[str]

class ClassificationResponse(BaseModel):
    category: Optional[str]

# --- Inicialización de la App y Carga de Datos ---

app = FastAPI(
    title="API Análisis Transcripciones Entel GenAI",
    description="API para buscar, extraer temas y clasificar transcripciones de llamadas.",
    version="0.1.0",
    debug=True
)

# Variable global para almacenar transcripciones procesadas cargadas
loaded_processed_transcripts: Dict[str, str] = {}

@app.on_event("startup")
async def startup_event():
    """Carga los datos necesarios al iniciar la API."""
    print("Cargando datos en startup...")
    global loaded_processed_transcripts
    load_search_data()
    processed_list = load_processed_transcripts()
    for t in processed_list:
        full_text = " ".join([utt.get("processed_text", "") for utt in t.get("processed_data", [])])
        loaded_processed_transcripts[t.get("id")] = full_text
    print(f"Startup completo. {len(loaded_processed_transcripts)} transcripciones en memoria.")

# --- Endpoints de la API ---

@app.post("/search/", response_model=SearchResponse, tags=["Búsqueda"])
async def search_transcripts(request: SearchRequest):
    """Busca transcripciones por semántica o palabra clave."""
    if request.search_type == "semantic":
        results = semantic_search(request.query, top_n=request.top_n)
        response_items = [SearchResponseItem(transcript_id=tid, score=score) for tid, score in results]
    elif request.search_type == "keyword":
        results = keyword_search(request.query, top_n=request.top_n)
        response_items = [SearchResponseItem(transcript_id=tid) for tid in results]
    else:
        raise HTTPException(status_code=400, detail="Tipo de búsqueda inválido. Usar 'semantic' o 'keyword'.")

    return SearchResponse(results=response_items)

def get_text_for_analysis(request: AnalysisRequest) -> str:
    """Obtiene el texto a analizar, ya sea por ID o directamente."""
    transcript_id_to_find = None
    text_to_return = None

    if request.transcript_id:
        raw_id = request.transcript_id.strip()
        print(f"Buscando transcript_id recibido: '{raw_id}'")

        text_to_return = loaded_processed_transcripts.get(raw_id)

        if text_to_return is None and raw_id.isdigit():
            potential_id = f"sample_{raw_id.zfill(2)}"
            print(f"ID '{raw_id}' no encontrado. Intentando con prefijo: '{potential_id}'")
            text_to_return = loaded_processed_transcripts.get(potential_id)
            if text_to_return:
                 transcript_id_to_find = potential_id

        elif text_to_return:
             transcript_id_to_find = raw_id

        if text_to_return is None:
            print(f"Error: No se encontró texto para ID '{raw_id}' ni con prefijo.")
            available_ids_preview = list(loaded_processed_transcripts.keys())[:5]
            print(f"IDs disponibles (ejemplo): {available_ids_preview}")
            raise HTTPException(status_code=404, detail=f"Transcripción con ID '{raw_id}' (o formato similar) no encontrada.")
        else:
             print(f"Texto encontrado para ID final: '{transcript_id_to_find}'")
             return text_to_return

    elif request.text:
        print("Usando texto proporcionado directamente en el request.")
        return request.text
    else:
        raise HTTPException(status_code=400, detail="Se debe proporcionar 'transcript_id' o 'text'.")

@app.post("/analyze/topics/", response_model=TopicResponse, tags=["Análisis"])
async def analyze_topics(request: AnalysisRequest):
    """Extrae temas principales de una transcripción."""
    print("Request recibido en /analyze/topics/:", request)
    try:
        text = get_text_for_analysis(request)
        topics = extract_topics(text)
        if topics is None:
            raise HTTPException(status_code=500, detail="Error al extraer temas desde la API de OpenAI.")
        return TopicResponse(topics=topics)
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        print(f"Error inesperado en /analyze/topics/: {e}")
        raise HTTPException(status_code=500, detail=f"Error interno del servidor al analizar temas: {type(e).__name__}")

@app.post("/analyze/classify/", response_model=ClassificationResponse, tags=["Análisis"])
async def analyze_classification(request: AnalysisRequest):
    """Clasifica una transcripción en una categoría predefinida."""
    print("Request recibido en /analyze/classify/:", request)
    try:
        text = get_text_for_analysis(request)
        category = classify_transcript(text)
        return ClassificationResponse(category=category)
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        print(f"Error inesperado en /analyze/classify/: {e}")
        raise HTTPException(status_code=500, detail=f"Error interno del servidor al clasificar: {type(e).__name__}")

@app.get("/status", tags=["Utilidad"])
async def get_status():
    """Verifica si la API está funcionando."""
    return {"status": "ok", "transcripts_loaded": len(loaded_processed_transcripts)}
import os
import sys
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import asyncio # Importar asyncio

# Añadir el directorio raíz
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Lógica de búsqueda y análisis
from app.search_logic import semantic_search, keyword_search
# >>> La lógica de análisis ahora contiene funciones ASYNC <<<
from app.analysis_logic import extract_topics, classify_transcript

# Qdrant Imports y configuración (sin cambios respecto a la versión anterior con Qdrant)
from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
COLLECTION_NAME = "transcripts_prod"
qdrant_client_main: Optional[QdrantClient] = None

# --- Modelos Pydantic (Sin cambios) ---
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
# --- Inicialización de la App y Conexión a Qdrant ---

app = FastAPI(
    title="API Análisis Transcripciones (Qdrant + Async OpenAI)",
    description="API para buscar, extraer temas y clasificar transcripciones.",
    version="0.3.0",
    debug=True
)

@app.on_event("startup")
async def startup_event():
    """Inicializa la conexión a Qdrant al iniciar la API."""
    global qdrant_client_main
    print("Iniciando API y conexión a Qdrant...")
    try:
        qdrant_client_main = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, grpc_port=QDRANT_PORT, timeout=10) # Añadir timeout
        # Opcional: Verificar si la colección existe
        qdrant_client_main.get_collection(collection_name=COLLECTION_NAME)
        print(f"Conexión a Qdrant ({QDRANT_HOST}:{QDRANT_PORT}) y colección '{COLLECTION_NAME}' verificadas.")
    except UnexpectedResponse as e:
         if e.status_code == 404:
             print(f"Error CRÍTICO: La colección '{COLLECTION_NAME}' no existe en Qdrant.")
             print("Asegúrate de ejecutar 'scripts/generate_embeddings_openai.py' primero.")
             qdrant_client_main = None # Marcar como no disponible
         else:
              print(f"Error CRÍTICO: No se pudo conectar o verificar la colección en Qdrant ({QDRANT_HOST}:{QDRANT_PORT}). Error: {e}")
              qdrant_client_main = None
    except Exception as e:
        print(f"Error CRÍTICO: No se pudo conectar a Qdrant ({QDRANT_HOST}:{QDRANT_PORT}). Error: {e}")
        qdrant_client_main = None

    # Ya no cargamos transcripciones en memoria
    print("Startup completo.")

@app.on_event("shutdown")
async def shutdown_event():
    """Cierra la conexión a Qdrant si es necesario."""
    global qdrant_client_main
    if qdrant_client_main:
        print("Cerrando conexión a Qdrant...")
        print("Conexión Qdrant cerrada (o gestionada automáticamente).")

# --- Endpoints de la API ---
# Función para obtener texto
@app.post("/search/", response_model=SearchResponse, tags=["Búsqueda"])
async def search_transcripts(request: SearchRequest):
    """Busca transcripciones por semántica o palabra clave usando Qdrant."""
    # Usando cliente global por simplicidad en este ejemplo
    if qdrant_client_main is None:
         raise HTTPException(status_code=503, detail="Servicio Qdrant no disponible para búsqueda.")

    if request.search_type == "semantic":
        # Pasar el cliente si search_logic lo requiere o asegurarse que search_logic usa su propia instancia bien configurada
        results = semantic_search(request.query, top_n=request.top_n)
        response_items = [SearchResponseItem(transcript_id=tid, score=score) for tid, score in results]
    elif request.search_type == "keyword":
        results = keyword_search(request.query, top_n=request.top_n)
        response_items = [SearchResponseItem(transcript_id=tid) for tid in results]
    else:
        raise HTTPException(status_code=400, detail="Tipo de búsqueda inválido. Usar 'semantic' o 'keyword'.")

    return SearchResponse(results=response_items)

def get_text_for_analysis(request: AnalysisRequest) -> str:
    """Obtiene el texto a analizar, ya sea desde Qdrant por ID o directamente del request."""
    # Usando cliente global por simplicidad
    if qdrant_client_main is None:
         raise HTTPException(status_code=503, detail="Servicio Qdrant no disponible para obtener texto.")

    if request.transcript_id:
        transcript_id_to_find = request.transcript_id.strip()
        print(f"Buscando texto en Qdrant por original_id: '{transcript_id_to_find}'")

        try:
            # Usar scroll con un filtro para buscar por el campo 'original_id' en el payload
            search_result = qdrant_client_main.scroll(
                collection_name=COLLECTION_NAME,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="original_id", # Buscar en este campo del payload
                            match=models.MatchValue(value=transcript_id_to_find) # Coincidencia exacta
                        )
                    ]
                ),
                limit=1, # Solo necesitamos el primero que coincida
                with_payload=True, # Necesitamos el payload para obtener el texto
                with_vectors=False
            )

            retrieved_points = search_result[0] # La lista de puntos encontrados

            if not retrieved_points:
                print(f"ID original '{transcript_id_to_find}' no encontrado en payload Qdrant.")
                raise HTTPException(status_code=404, detail=f"Transcripción con ID '{transcript_id_to_find}' no encontrada.")

            # Punto encontrado, extraer texto del payload
            payload = retrieved_points[0].payload
            if payload is None: # Aunque el filtro asegura que payload existe, buena práctica chequear
                print(f"Error: Punto encontrado para ID original '{transcript_id_to_find}' pero no tiene payload.")
                raise HTTPException(status_code=404, detail=f"Datos incompletos para transcripción ID '{transcript_id_to_find}'.")

            text_to_return = payload.get("full_text")
            if text_to_return is None:
                print(f"Error: Punto encontrado para ID original '{transcript_id_to_find}' pero no tiene 'full_text' en payload.")
                raise HTTPException(status_code=404, detail=f"Datos incompletos (sin texto) para transcripción ID '{transcript_id_to_find}'.")

            print(f"Texto encontrado en Qdrant para ID original: '{transcript_id_to_find}' (longitud: {len(text_to_return)})")
            return text_to_return

        except HTTPException as http_exc:
             raise http_exc # Re-lanzar excepciones HTTP
        except Exception as e:
            print(f"Error inesperado recuperando texto de Qdrant para ID '{transcript_id_to_find}': {type(e).__name__} - {e}")
            # Podría ser un error de conexión temporal o un ID inválido para Qdrant
            raise HTTPException(status_code=500, detail=f"Error interno al buscar texto para ID '{transcript_id_to_find}'.")

    elif request.text:
        print("Usando texto proporcionado directamente en el request.")
        if not isinstance(request.text, str):
             raise HTTPException(status_code=400, detail="El campo 'text' debe ser una cadena.")
        return request.text # Devolver el texto directamente
    else:
        raise HTTPException(status_code=400, detail="Se debe proporcionar 'transcript_id' o 'text'.")

@app.post("/analyze/topics/", response_model=TopicResponse, tags=["Análisis"])
async def analyze_topics(request: AnalysisRequest):
    """Extrae temas principales (llamada ASÍNCRONA a OpenAI)."""
    print("Request ASYNC recibido en /analyze/topics/")
    try:
        # get_text_for_analysis sigue siendo síncrono por ahora
        text = get_text_for_analysis(request)
        # >>> LLAMADA ASÍNCRONA a la lógica de análisis <<<
        topics = await extract_topics(text)
        if topics is None and text: # Si hubo texto pero la extracción falló
            raise HTTPException(status_code=502, detail="Error servicio IA (Temas).")
        return TopicResponse(topics=topics if topics is not None else []) # Devolver lista vacía si no hubo texto
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        print(f"Error inesperado en /analyze/topics/: {type(e).__name__} - {e}")
        raise HTTPException(status_code=500, detail="Error interno (Temas).")


@app.post("/analyze/classify/", response_model=ClassificationResponse, tags=["Análisis"])
async def analyze_classification(request: AnalysisRequest):
    """Clasifica una transcripción (llamada ASÍNCRONA a OpenAI)."""
    print("Request ASYNC recibido en /analyze/classify/")
    try:
        text = get_text_for_analysis(request)
         # >>> LLAMADA ASÍNCRONA a la lógica de análisis <<<
        category = await classify_transcript(text)
        if category is None and text: # Si hubo texto pero la clasificación falló
             raise HTTPException(status_code=502, detail="Error servicio IA (Clasificación).")
        return ClassificationResponse(category=category) # Devuelve None si no hubo texto o clasificación fallida sin texto
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        print(f"Error inesperado en /analyze/classify/: {type(e).__name__} - {e}")
        raise HTTPException(status_code=500, detail="Error interno (Clasificación).")

@app.get("/status", tags=["Utilidad"])
async def get_status():
    qdrant_status = "no disponible"
    collection_points = "desconocido"
    if qdrant_client_main:
        try:
            collection_info = qdrant_client_main.get_collection(collection_name=COLLECTION_NAME)
            qdrant_status = "conectado"
            collection_points = collection_info.points_count
        except Exception as e:
            qdrant_status = f"error ({type(e).__name__})"
    return { "status": "ok", "qdrant_status": qdrant_status, "qdrant_collection": COLLECTION_NAME, "collection_points_count": collection_points }


# --- Bloque para ejecutar con uvicorn ---
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)
"""
Script de Embeddings e Indexación en Qdrant
Este script procesa transcripciones en formato JSON, genera embeddings en lotes mediante la API de OpenAI y los indexa en la base de datos vectorial Qdrant.
"""

import os
import sys
import time
import inquirer
from typing import List, Dict, Any, Optional
from itertools import islice  # Para generar lotes (batch)
import uuid

# Permite importar módulos de 'app' y 'scripts'
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Importación de funciones críticas de OpenAI y carga de datos procesados
from app.openai_utils import get_embedding, get_embeddings_batch
from scripts.load_data import load_processed_transcripts, PROCESSED_DIR

# --- Importaciones de Qdrant ---
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import Distance, VectorParams, PointStruct

# --- Constantes y Configuración ---
MODEL_EMBEDDING = "text-embedding-3-small"
COST_PER_MILLION_TOKENS = 0.02  # Costo estimado (solo para referencia)
CHARS_PER_TOKEN_ESTIMATE = 4.5
BATCH_SIZE = 64  # Cantidad de transcripciones a procesar por lote

# Configuración de Qdrant
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
COLLECTION_NAME = "transcripts_prod"
EMBEDDING_DIM = 1536  # Debe coincidir con el modelo de embedding

def estimate_tokens(text: str) -> int:
    """Aproxima el número de tokens basado en la cantidad de caracteres."""
    if not text:
        return 0
    return int(len(text) / CHARS_PER_TOKEN_ESTIMATE)

def check_qdrant_collection_exists(client: QdrantClient, collection_name: str) -> bool:
    """
    Comprueba si la colección existe en Qdrant.
    
    Se basa en la respuesta del cliente; cualquier error con indicación de "not found" se interpreta como colección inexistente.
    """
    try:
        client.get_collection(collection_name=collection_name)
        return True
    except Exception as e:
        if "not found" in str(e).lower() or "status_code=404" in str(e):
            return False
        print(f"Advertencia: Error inesperado al verificar la colección '{collection_name}': {e}")
        return False

def prompt_recreate_collection(collection_name: str) -> bool:
    """
    Solicita al usuario si desea recrear la colección.
    
    Importante: Esto borrará todos los datos existentes.
    """
    questions = [
        inquirer.Confirm(
            'recreate',
            message=f"La colección '{collection_name}' ya existe en Qdrant ({QDRANT_HOST}:{QDRANT_PORT}). ¿Deseas recrearla? (¡ESTO BORRARÁ TODOS LOS DATOS ACTUALES!)",
            default=False
        ),
    ]
    answers = inquirer.prompt(questions)
    return answers and answers['recreate']

def create_qdrant_collection(client: QdrantClient, collection_name: str, embedding_dim: int):
    """
    Crea una nueva colección en Qdrant con la configuración de vectores.
    
    Se pueden añadir configuraciones adicionales (por ejemplo, optimizaciones de índices) según sea necesario.
    """
    try:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=embedding_dim, distance=Distance.COSINE),
            # Opcional: incluir optimizaciones o índices específicos
        )
        print(f"Colección '{collection_name}' creada exitosamente.")
    except Exception as e:
        print(f"Error CRÍTICO al crear la colección '{collection_name}': {e}")
        raise

def load_transcripts_generator(data_dir: str = PROCESSED_DIR):
    """
    Genera de manera perezosa (lazy) las transcripciones procesadas desde el directorio indicado.
    
    Permite manejar grandes volúmenes de datos sin cargar todo en memoria.
    """
    if not os.path.exists(data_dir):
        print(f"Error: El directorio de datos procesados no existe: {data_dir}")
        return
    print(f"Cargando transcripciones procesadas (JSON) desde: {data_dir} (modo generador)")

    try:
        filenames = [f for f in os.listdir(data_dir) if f.startswith("processed_") and f.endswith(".json")]
        print(f"Encontrados {len(filenames)} archivos JSON procesados.")
    except Exception as e:
        print(f"Error listando archivos en {data_dir}: {e}")
        return

    for filename in filenames:
        file_path = os.path.join(data_dir, filename)
        try:
            with open(file_path, 'rb') as f:
                try:
                    import orjson
                    data = orjson.loads(f.read())
                    yield data
                except ImportError:
                    import json
                    data = json.load(f)
                    yield data
        except Exception as e:
            print(f"Error al cargar o parsear el archivo {filename}: {e}. Saltando.")
            continue

def main():
    """Ejecuta el flujo principal: conecta a Qdrant, gestiona la colección y procesa lotes de embeddings."""
    print("Iniciando generación de embeddings e indexación en Qdrant...")

    try:
        qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        print(f"Conectado a Qdrant en {QDRANT_HOST}:{QDRANT_PORT}")
    except Exception as e:
        print(f"Error CRÍTICO: No se pudo conectar a Qdrant en {QDRANT_HOST}:{QDRANT_PORT}. Verifica la conexión. Error: {e}")
        return

    # Gestiona la colección: recrearla o usar la existente según elección del usuario
    collection_exists = check_qdrant_collection_exists(qdrant_client, COLLECTION_NAME)
    if collection_exists:
        print(f"Colección '{COLLECTION_NAME}' encontrada.")
        if prompt_recreate_collection(COLLECTION_NAME):
            print(f"Recreando la colección '{COLLECTION_NAME}'...")
            try:
                qdrant_client.delete_collection(collection_name=COLLECTION_NAME)
                print("Colección anterior eliminada.")
                create_qdrant_collection(qdrant_client, COLLECTION_NAME, EMBEDDING_DIM)
            except Exception as e:
                print(f"Error CRÍTICO al recrear la colección: {e}")
                return
        else:
            print("Usando la colección existente. Los IDs se actualizarán (upsert).")
    else:
        print(f"Colección '{COLLECTION_NAME}' no encontrada. Creando nueva...")
        create_qdrant_collection(qdrant_client, COLLECTION_NAME, EMBEDDING_DIM)

    # --- Preparación para procesamiento en lotes ---
    transcript_generator = load_transcripts_generator()
    total_processed_count = 0
    total_estimated_tokens_batch = 0
    qdrant_points_batch = []  # Lote de puntos para Qdrant
    openai_texts_batch = []   # Textos a enviar a OpenAI
    openai_ids_batch = []     # IDs correspondientes
    error_count_embeddings = 0
    error_count_qdrant = 0

    print(f"\nProcesando lotes (Colección: '{COLLECTION_NAME}', Modelo: '{MODEL_EMBEDDING}', Tamaño Lote: {BATCH_SIZE})...")
    print("Realizando llamadas batch a la API de OpenAI.")
    start_time = time.time()

    # Procesa cada transcripción generada
    for transcript_data in transcript_generator:
        transcript_id = transcript_data.get("id")
        if not transcript_id:
            print("Advertencia: Transcripción sin ID. Saltando.")
            continue

        processed_data = transcript_data.get("processed_data", [])
        full_text = " ".join([utt.get("processed_text", "") for utt in processed_data if utt.get("processed_text")])

        if not full_text.strip():
            print(f"Advertencia: Transcripción {transcript_id} sin texto procesado. Saltando.")
            continue

        # Acumula datos para el lote
        openai_texts_batch.append(full_text)
        openai_ids_batch.append(transcript_id)
        total_estimated_tokens_batch += estimate_tokens(full_text)

        # Al alcanzar el tamaño de lote, procesa la solicitud a OpenAI
        if len(openai_texts_batch) >= BATCH_SIZE:
            print(f"Procesando lote de {len(openai_texts_batch)} textos en OpenAI...")
            embeddings_list: Optional[List[Optional[List[float]]]] = get_embeddings_batch(openai_texts_batch, model=MODEL_EMBEDDING)

            if embeddings_list:
                # Construye los puntos para Qdrant a partir de los embeddings obtenidos
                for i, embedding in enumerate(embeddings_list):
                    if embedding:
                        point_qdrant_id = str(uuid.uuid4())
                        original_id = openai_ids_batch[i]
                        qdrant_points_batch.append(PointStruct(
                            id=point_qdrant_id,
                            vector=embedding,
                            payload={"original_id": original_id, "full_text": openai_texts_batch[i][:20000]}
                        ))
                        total_processed_count += 1
                    else:
                        print(f"Error: Embedding faltante para la transcripción {openai_ids_batch[i]}.")
                        error_count_embeddings += 1
            else:
                print(f"Error CRÍTICO: Falló la llamada batch a OpenAI para {len(openai_texts_batch)} textos.")
                error_count_embeddings += len(openai_texts_batch)

            print(f"  Lote procesado. {len(qdrant_points_batch)} puntos listos para enviar a Qdrant.")
            openai_texts_batch = []
            openai_ids_batch = []
            total_estimated_tokens_batch = 0

            # Envía a Qdrant si se alcanza un tamaño de lote significativo
            if len(qdrant_points_batch) >= BATCH_SIZE * 2:
                try:
                    if qdrant_points_batch:
                        print(f"  Subiendo {len(qdrant_points_batch)} puntos a Qdrant...")
                        qdrant_client.upsert(collection_name=COLLECTION_NAME, points=qdrant_points_batch, wait=False)
                        qdrant_points_batch = []
                except Exception as e:
                    print(f"Error CRÍTICO al hacer upsert en Qdrant: {e}. Se descarta el lote actual.")
                    error_count_qdrant += len(qdrant_points_batch)
                    qdrant_points_batch = []

            time.sleep(0.1)  # Pequeña pausa entre lotes

    # --- Procesamiento de datos residuales ---
    if openai_texts_batch:
        print(f"Procesando último lote de {len(openai_texts_batch)} textos en OpenAI...")
        embeddings_list = get_embeddings_batch(openai_texts_batch, model=MODEL_EMBEDDING)
        if embeddings_list:
            for i, embedding in enumerate(embeddings_list):
                if embedding:
                    point_qdrant_id = str(uuid.uuid4())
                    original_id = openai_ids_batch[i]
                    qdrant_points_batch.append(PointStruct(
                        id=point_qdrant_id,
                        vector=embedding,
                        payload={"original_id": original_id, "full_text": openai_texts_batch[i][:20000]}
                    ))
                    total_processed_count += 1
                else:
                    print(f"Error: Embedding faltante para la transcripción {openai_ids_batch[i]} en el último lote.")
                    error_count_embeddings += 1
        else:
            print("Error CRÍTICO: Falló la última llamada batch a OpenAI.")
            error_count_embeddings += len(openai_texts_batch)

    # Último upsert de los puntos acumulados en Qdrant
    if qdrant_points_batch:
        try:
            print(f"Subiendo último lote de {len(qdrant_points_batch)} puntos a Qdrant...")
            qdrant_client.upsert(collection_name=COLLECTION_NAME, points=qdrant_points_batch, wait=True)
            print("  Último lote enviado a Qdrant.")
        except Exception as e:
            print(f"Error CRÍTICO al hacer upsert del último lote: {e}")
            error_count_qdrant += len(qdrant_points_batch)

    # --- Resumen Final ---
    end_time = time.time()
    duration = end_time - start_time
    total_errors = error_count_embeddings + error_count_qdrant

    # Cálculo aproximado del costo basado en tokens (valor referencial)
    estimated_cost = (total_processed_count * (sum(len(p.payload['full_text']) for p in qdrant_points_batch) / total_processed_count if total_processed_count > 0 else 0) / CHARS_PER_TOKEN_ESTIMATE / 1_000_000) * COST_PER_MILLION_TOKENS

    print("\n--- Resumen de Indexación ---")
    print(f"Transcripciones procesadas exitosamente: {total_processed_count}")
    if total_errors > 0:
        print(f"Errores totales: {total_errors}")
        print(f"  Errores de Embedding (OpenAI): {error_count_embeddings}")
        print(f"  Errores de Upsert (Qdrant): {error_count_qdrant}")
    print(f"Tiempo total: {duration:.2f} segundos")
    print(f"Colección utilizada: '{COLLECTION_NAME}' en {QDRANT_HOST}:{QDRANT_PORT}")
    print("¡Proceso de generación e indexación completado!")

if __name__ == "__main__":
    main()

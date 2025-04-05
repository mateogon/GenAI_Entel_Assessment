import os
import sys
import time
import inquirer
from typing import List, Dict, Any, Optional
from itertools import islice # Para crear batches
import uuid

# Añadir el directorio raíz al path para importar desde 'app' y 'scripts'
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# La función get_embedding debe poder hacer llamadas reales aquí.
from app.openai_utils import get_embedding, get_embeddings_batch
from scripts.load_data import load_processed_transcripts, PROCESSED_DIR

# --- Qdrant Imports ---
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import Distance, VectorParams, PointStruct

# --- Constantes ---
MODEL_EMBEDDING = "text-embedding-3-small"
COST_PER_MILLION_TOKENS = 0.02 # Solo para estimación, las llamadas reales son el coste
CHARS_PER_TOKEN_ESTIMATE = 4.5
BATCH_SIZE = 64 # Número de transcripciones a procesar y subir a Qdrant a la vez

# --- Qdrant Config ---
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
COLLECTION_NAME = "transcripts_prod"
# Asegúrate que esta dimensión coincide con el modelo de embedding usado
EMBEDDING_DIM = 1536 # Para text-embedding-3-small

def estimate_tokens(text: str) -> int:
    """Estima el número de tokens basado en caracteres."""
    if not text: return 0
    # Es una aproximación muy burda, OpenAI usa tiktoken para el cálculo exacto
    return int(len(text) / CHARS_PER_TOKEN_ESTIMATE)

def check_qdrant_collection_exists(client: QdrantClient, collection_name: str) -> bool:
    """Verifica si la colección ya existe en Qdrant."""
    try:
        client.get_collection(collection_name=collection_name)
        return True
    except Exception as e:
        # Asume que la excepción significa que no existe (podría ser otro error)
        # Un manejo más robusto verificaría el tipo de error
        if "not found" in str(e).lower() or "status_code=404" in str(e):
             return False
        print(f"Advertencia: Error inesperado al verificar colección '{collection_name}': {e}")
        return False # Asumir que no existe en caso de error desconocido

def prompt_recreate_collection(collection_name: str) -> bool:
    """Pregunta al usuario si desea recrear la colección Qdrant (borrar y empezar de nuevo)."""
    questions = [
        inquirer.Confirm('recreate',
                          message=f"La colección '{collection_name}' ya existe en Qdrant ({QDRANT_HOST}:{QDRANT_PORT}). ¿Deseas recrearla? (¡ESTO BORRARÁ TODOS LOS DATOS ACTUALES!)",
                          default=False),
    ]
    answers = inquirer.prompt(questions)
    return answers and answers['recreate']

def create_qdrant_collection(client: QdrantClient, collection_name: str, embedding_dim: int):
    """Crea la colección en Qdrant."""
    try:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=embedding_dim, distance=Distance.COSINE),
            # Opcional: añadir optimizaciones o índices aquí
            # hnsw_config=models.HnswConfig(...)
            # optimizers_config=models.OptimizersConfigDiff(...)
        )
        print(f"Colección '{collection_name}' creada exitosamente.")
    except Exception as e:
        print(f"Error CRÍTICO al crear la colección '{collection_name}': {e}")
        raise # Relanzar el error, no podemos continuar sin colección

def load_transcripts_generator(data_dir: str = PROCESSED_DIR):
    """Generador para cargar transcripciones procesadas una por una."""
    if not os.path.exists(data_dir):
        print(f"Error: El directorio de datos procesados no existe: {data_dir}")
        return
    print(f"Cargando transcripciones procesadas (JSON) desde: {data_dir} (modo generador)")

    try:
        filenames = [f for f in os.listdir(data_dir) if f.startswith("processed_") and f.endswith(".json")]
        print(f"Encontrados {len(filenames)} archivos JSON procesados.")
    except Exception as e:
        print(f"Error listando archivos procesados en {data_dir}: {e}")
        return

    for filename in filenames:
        file_path = os.path.join(data_dir, filename)
        try:
            with open(file_path, 'rb') as f:
                # Usar orjson si está instalado y es más rápido
                try:
                    import orjson
                    data = orjson.loads(f.read())
                    yield data
                except ImportError:
                    import json
                    data = json.load(f) # Fallback a json estándar
                    yield data
        except Exception as e:
            print(f"Error cargando o parseando archivo {filename}: {e}. Saltando.")
            continue # Saltar al siguiente archivo

def main():
    """Función principal para generar embeddings e insertarlos en Qdrant."""
    print("Iniciando script de generación de embeddings e indexación en Qdrant...")

    # Inicializar cliente Qdrant
    try:
        qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
       
        print(f"Conectado a Qdrant en {QDRANT_HOST}:{QDRANT_PORT}")
    except Exception as e:
        print(f"Error CRÍTICO: No se pudo conectar a Qdrant en {QDRANT_HOST}:{QDRANT_PORT}. Verifica que esté corriendo. Error: {e}")
        return

    # Verificar si la colección existe y preguntar si se recrea
    collection_exists = check_qdrant_collection_exists(qdrant_client, COLLECTION_NAME)
    if collection_exists:
        print(f"Colección '{COLLECTION_NAME}' encontrada.")
        if prompt_recreate_collection(COLLECTION_NAME):
            print(f"Recreando colección '{COLLECTION_NAME}'...")
            try:
                qdrant_client.delete_collection(collection_name=COLLECTION_NAME)
                print("Colección anterior eliminada.")
                create_qdrant_collection(qdrant_client, COLLECTION_NAME, EMBEDDING_DIM)
            except Exception as e:
                print(f"Error CRÍTICO al recrear la colección: {e}")
                return
        else:
            print("Usando la colección existente. Los IDs existentes serán actualizados (upsert).")
            # Aquí podríamos añadir lógica para solo procesar archivos nuevos/modificados
    else:
        print(f"Colección '{COLLECTION_NAME}' no encontrada. Creando...")
        create_qdrant_collection(qdrant_client, COLLECTION_NAME, EMBEDDING_DIM)

    # --- Lógica de Batching ---
    transcript_generator = load_transcripts_generator()
    total_processed_count = 0
    total_estimated_tokens_batch = 0
    qdrant_points_batch = [] # Lote para Qdrant
    openai_texts_batch = [] # Lote de textos para OpenAI
    openai_ids_batch = []   # Lote de IDs correspondientes para OpenAI
    error_count_embeddings = 0
    error_count_qdrant = 0

    print(f"\nGenerando embeddings BATCH e indexando (Colección: '{COLLECTION_NAME}', Modelo: '{MODEL_EMBEDDING}', Tamaño Lote: {BATCH_SIZE})...")
    print("Esto realizará llamadas BATCH a la API de OpenAI.")
    start_time = time.time()

    # Iterar sobre el generador
    for transcript_data in transcript_generator:
        transcript_id = transcript_data.get("id")
        if not transcript_id: print("Advertencia: Transcripción sin ID. Saltando."); continue

        processed_data = transcript_data.get("processed_data", [])
        full_text = " ".join([utt.get("processed_text", "") for utt in processed_data if utt.get("processed_text")])

        if not full_text.strip(): print(f"Advertencia: ID {transcript_id} sin texto procesado. Saltando."); continue

        # Acumular para el lote de OpenAI
        openai_texts_batch.append(full_text)
        openai_ids_batch.append(transcript_id)
        total_estimated_tokens_batch += estimate_tokens(full_text)

        # Si el lote de OpenAI está lleno, procesarlo
        if len(openai_texts_batch) >= BATCH_SIZE:
            print(f"Procesando lote OpenAI de {len(openai_texts_batch)} textos...")
            # >>> LLAMAR A LA FUNCIÓN BATCH <<<
            embeddings_list: Optional[List[Optional[List[float]]]] = get_embeddings_batch(openai_texts_batch, model=MODEL_EMBEDDING)

            if embeddings_list:
                # Crear puntos Qdrant para los embeddings obtenidos
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
                        # Si un embedding específico falló (poco común con batch actual)
                        print(f"Error: Embedding faltante para ID {openai_ids_batch[i]} dentro del lote.")
                        error_count_embeddings += 1
            else:
                # Si toda la llamada batch falló
                print(f"Error CRÍTICO: Falló la llamada batch a OpenAI para {len(openai_texts_batch)} textos.")
                error_count_embeddings += len(openai_texts_batch) # Contar todos como error

            # Resetear lotes de OpenAI
            print(f"  Lote OpenAI procesado. {len(qdrant_points_batch)} puntos listos para Qdrant.")
            openai_texts_batch = []
            openai_ids_batch = []
            total_estimated_tokens_batch = 0 # Resetear contador de tokens del lote

            # Opcional: Hacer upsert a Qdrant aquí si qdrant_points_batch es grande
            if len(qdrant_points_batch) >= BATCH_SIZE * 2: # Ejemplo: subir cada 2 lotes de OpenAI
                 try:
                    if qdrant_points_batch:
                         print(f"  Enviando {len(qdrant_points_batch)} puntos a Qdrant...")
                         qdrant_client.upsert(collection_name=COLLECTION_NAME, points=qdrant_points_batch, wait=False)
                         qdrant_points_batch = [] # Resetear lote Qdrant
                 except Exception as e:
                     print(f"Error CRÍTICO haciendo upsert a Qdrant: {e}. Descartando lote Qdrant.")
                     error_count_qdrant += len(qdrant_points_batch)
                     qdrant_points_batch = []

            time.sleep(0.1) # Pausa entre lotes OpenAI

    # --- Procesar Lotes Restantes ---
    # Procesar cualquier texto restante en el lote de OpenAI
    if openai_texts_batch:
        print(f"Procesando último lote OpenAI de {len(openai_texts_batch)} textos...")
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
                    print(f"Error: Embedding faltante para ID {openai_ids_batch[i]} en último lote.")
                    error_count_embeddings += 1
        else:
            print(f"Error CRÍTICO: Falló la última llamada batch a OpenAI.")
            error_count_embeddings += len(openai_texts_batch)

    # Hacer upsert final a Qdrant con los puntos restantes
    if qdrant_points_batch:
        try:
            print(f"Enviando último lote de {len(qdrant_points_batch)} puntos a Qdrant...")
            qdrant_client.upsert(collection_name=COLLECTION_NAME, points=qdrant_points_batch, wait=True) # Esperar al final
            print("  Último lote Qdrant enviado.")
        except Exception as e:
            print(f"Error CRÍTICO haciendo upsert del último lote a Qdrant: {e}")
            error_count_qdrant += len(qdrant_points_batch)

    # --- Resumen Final ---
    end_time = time.time()
    duration = end_time - start_time
    total_errors = error_count_embeddings + error_count_qdrant
    # Nota: El cálculo de costo/tokens es más complejo ahora, necesita sumar los tokens de cada batch exitoso.
    # Este cálculo es una sobreestimación si hubo fallos.
    # total_estimated_tokens_final = ... (requiere sumar tokens de batches OK)
    estimated_cost = (total_processed_count * (sum(len(p.payload['full_text']) for p in qdrant_points_batch) / total_processed_count if total_processed_count > 0 else 0) / CHARS_PER_TOKEN_ESTIMATE / 1_000_000) * COST_PER_MILLION_TOKENS


    print("\n--- Resumen de Indexación BATCH en Qdrant ---")
    print(f"Transcripciones procesadas exitosamente (embeddings generados y puntos creados): {total_processed_count}")
    if total_errors > 0:
        print(f"Errores totales (fallos de embedding + fallos de upsert Qdrant): {total_errors}")
        print(f"  Errores de Embedding (OpenAI): {error_count_embeddings}")
        print(f"  Errores de Upsert (Qdrant): {error_count_qdrant}")
    print(f"Tiempo total: {duration:.2f} segundos")
    # print(f"Tokens estimados enviados a OpenAI: {total_estimated_tokens_final}") # Cálculo más preciso necesario
    # print(f"Costo estimado (solo OpenAI embeddings): ${estimated_cost:.6f}") # Cálculo más preciso necesario
    print(f"Colección Qdrant utilizada: '{COLLECTION_NAME}' en {QDRANT_HOST}:{QDRANT_PORT}")
    print("¡Generación e indexación BATCH completada!")

if __name__ == "__main__":
    main()
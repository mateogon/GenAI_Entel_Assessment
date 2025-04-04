import os
import sys
import numpy as np
import pickle
import time
import inquirer
from typing import List, Dict, Any

# Añadir el directorio raíz al path para importar desde 'app'
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# La función get_embedding debe poder hacer llamadas reales aquí.
from app.openai_utils import get_embedding
from scripts.load_data import load_processed_transcripts, EMBEDDINGS_FILE, ID_MAP_FILE

# --- Constantes ---
MODEL_EMBEDDING = "text-embedding-3-small"
COST_PER_MILLION_TOKENS = 0.02
CHARS_PER_TOKEN_ESTIMATE = 4.5

def estimate_tokens(text: str) -> int:
    if not text: return 0
    return int(len(text) / CHARS_PER_TOKEN_ESTIMATE)

def check_existing_files() -> bool:
    """Verifica si los archivos de embeddings ya existen."""
    return os.path.exists(EMBEDDINGS_FILE) and os.path.exists(ID_MAP_FILE)

def prompt_overwrite() -> bool:
    """Pregunta al usuario si desea sobrescribir los archivos existentes."""
    questions = [
        inquirer.Confirm('overwrite',
                          message=f"Los archivos '{os.path.basename(EMBEDDINGS_FILE)}' y '{os.path.basename(ID_MAP_FILE)}' ya existen. ¿Deseas sobrescribirlos? (Esto realizará llamadas a la API de OpenAI)",
                          default=False),
    ]
    answers = inquirer.prompt(questions)
    return answers and answers['overwrite']

def main():
    print("Iniciando script de generación de embeddings con OpenAI...")

    # Verificar si los archivos ya existen
    if check_existing_files():
        print("Archivos de embeddings encontrados.")
        if not prompt_overwrite():
            print("Operación cancelada por el usuario. No se sobrescribirán los archivos.")
            return # Salir del script si el usuario no quiere sobrescribir
        else:
            print("Procediendo a sobrescribir los archivos existentes...")
    else:
        print("No se encontraron archivos de embeddings existentes. Se generarán nuevos.")

    processed_transcripts = load_processed_transcripts()

    if not processed_transcripts:
        print("No se encontraron transcripciones procesadas. Ejecuta preprocess_data.py primero. Abortando.")
        return

    embeddings = []
    id_map = {} # Mapea índice numérico (0, 1, 2...) a transcript_id original
    total_estimated_tokens = 0

    print(f"\nGenerando embeddings para {len(processed_transcripts)} transcripciones usando '{MODEL_EMBEDDING}'...")
    print("Esto realizará llamadas a la API de OpenAI.")

    start_time = time.time()

    for index, transcript in enumerate(processed_transcripts):
        transcript_id = transcript.get("id")
        processed_data = transcript.get("processed_data", [])
        full_text = " ".join([utterance.get("processed_text", "") for utterance in processed_data if utterance.get("processed_text")])

        if not full_text.strip():
            print(f"Advertencia: Transcripción ID {transcript_id} no tiene texto procesado. Saltando.")
            continue

        estimated_tokens = estimate_tokens(full_text)
        total_estimated_tokens += estimated_tokens

        # Llamada real a la API (asegúrate que openai_utils lo permite para get_embedding)
        embedding = get_embedding(full_text, model=MODEL_EMBEDDING)

        if embedding:
            embeddings.append(embedding)
            id_map[index] = transcript_id
        else:
            print(f"Error: No se pudo generar embedding para ID {transcript_id}. Saltando.")
            # Considerar una estrategia de reintento más robusta o parar el script si fallan muchos

        time.sleep(0.05) # Pausa

        if (index + 1) % 10 == 0 or index + 1 == len(processed_transcripts):
            print(f"  Procesadas {index + 1}/{len(processed_transcripts)} transcripciones...")

    end_time = time.time()
    duration = end_time - start_time

    if not embeddings:
        print("Error: No se generó ningún embedding. Abortando.")
        return

    embeddings_np = np.array(embeddings, dtype=np.float32)

    try:
        print(f"\nGuardando {embeddings_np.shape[0]} embeddings (dimension: {embeddings_np.shape[1]}) en {EMBEDDINGS_FILE}...")
        np.save(EMBEDDINGS_FILE, embeddings_np)

        print(f"Guardando mapa de IDs ({len(id_map)} entradas) en {ID_MAP_FILE}...")
        with open(ID_MAP_FILE, 'wb') as f:
            pickle.dump(id_map, f)

        estimated_cost = (total_estimated_tokens / 1_000_000) * COST_PER_MILLION_TOKENS

        print("\n--- Resumen ---")
        print(f"Embeddings generados/sobrescritos: {embeddings_np.shape[0]}")
        print(f"Tiempo total: {duration:.2f} segundos")
        print(f"Tokens estimados procesados: {total_estimated_tokens}")
        print(f"Costo estimado (solo embeddings): ${estimated_cost:.6f}")
        print("¡Generación de embeddings completada!")

    except Exception as e:
        print(f"\nError guardando archivos de embeddings/mapa: {e}")

if __name__ == "__main__":
    # Antes de ejecutar, instala inquirer: pip install inquirer
    main()
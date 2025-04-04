import os
import sys
import numpy as np
import pickle
import re
from typing import List, Dict, Tuple, Optional
from sklearn.metrics.pairwise import cosine_similarity

# Añadir el directorio raíz al path para importar desde 'app' y 'scripts'
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from app.openai_utils import get_embedding
from scripts.load_data import load_processed_transcripts, EMBEDDINGS_FILE, ID_MAP_FILE

# --- Carga de datos ---
embeddings_data: Optional[np.ndarray] = None
id_map: Optional[Dict[int, str]] = None
processed_transcripts_map: Optional[Dict[str, str]] = None # Mapa ID -> texto completo

def load_search_data():
    """Carga los embeddings, mapa de IDs y texto procesado."""
    global embeddings_data, id_map, processed_transcripts_map
    try:
        if os.path.exists(EMBEDDINGS_FILE) and embeddings_data is None:
            print("Cargando embeddings...")
            embeddings_data = np.load(EMBEDDINGS_FILE)
            print(f"Embeddings cargados: {embeddings_data.shape}")
        if os.path.exists(ID_MAP_FILE) and id_map is None:
            print("Cargando mapa de IDs...")
            with open(ID_MAP_FILE, 'rb') as f:
                id_map = pickle.load(f)
            print(f"Mapa de IDs cargado: {len(id_map)} entradas")
        if processed_transcripts_map is None:
            print("Cargando texto procesado...")
            processed_transcripts = load_processed_transcripts()
            processed_transcripts_map = {}
            for t in processed_transcripts:
                 full_text = " ".join([utt.get("processed_text", "") for utt in t.get("processed_data", [])])
                 processed_transcripts_map[t.get("id")] = full_text
            print(f"Texto procesado cargado para {len(processed_transcripts_map)} transcripciones.")

    except FileNotFoundError as e:
        print(f"Error: Archivo no encontrado al cargar datos de búsqueda: {e}. Asegúrate de generar los embeddings primero.")
    except Exception as e:
        print(f"Error inesperado cargando datos de búsqueda: {e}")

# Cargar datos al iniciar el módulo
load_search_data()

def semantic_search(query: str, top_n: int = 5) -> List[Tuple[str, float]]:
    """Realiza búsqueda semántica usando embeddings de OpenAI."""
    if embeddings_data is None or id_map is None:
        print("Error: Datos de embeddings no cargados.")
        return []
    if not query:
        return []

    print(f"Realizando búsqueda semántica para: '{query}'")
    query_embedding = get_embedding(query)

    if query_embedding is None:
        print("Error: No se pudo obtener embedding para la consulta.")
        return []

    # Calcular similitud coseno
    # query_embedding es (1, dim), embeddings_data es (N, dim)
    # El resultado de cosine_similarity será (1, N)
    similarities = cosine_similarity([query_embedding], embeddings_data)[0] # Obtener el array 1D

    # Obtener los índices de los top_n más similares (orden descendente)
    # argsort devuelve índices de menor a mayor, por eso usamos slicing [::-1]
    top_indices = np.argsort(similarities)[-top_n:][::-1]

    results = []
    for i in top_indices:
        transcript_id = id_map.get(i)
        score = similarities[i]
        if transcript_id:
            results.append((transcript_id, float(score))) # Convertir score a float nativo

    print(f"Resultados encontrados: {len(results)}")
    return results

def keyword_search(query: str, top_n: int = 10) -> List[str]:
    """Realiza búsqueda por palabra clave usando regex."""
    if processed_transcripts_map is None:
        print("Error: Texto procesado no cargado.")
        return []
    if not query:
        return []

    print(f"Realizando búsqueda por palabra clave para: '{query}'")
    results = []
    try:
        # Usar word boundaries (\b) para buscar palabras completas, ignorar mayúsculas/minúsculas
        regex = re.compile(r'\b' + re.escape(query) + r'\b', re.IGNORECASE)
    except re.error as e:
        print(f"Error en la expresión regular: {e}. Usando búsqueda simple.")
        # Fallback a búsqueda simple si la regex es inválida
        query_lower = query.lower()
        for transcript_id, text in processed_transcripts_map.items():
            if query_lower in text.lower():
                results.append(transcript_id)
                if len(results) == top_n: break # Limitar resultados
        return results

    for transcript_id, text in processed_transcripts_map.items():
        if regex.search(text):
            results.append(transcript_id)
            if len(results) == top_n: # Limitar resultados
                break

    print(f"Resultados encontrados: {len(results)}")
    return results


# Ejemplo de uso (descomentar para probar)
# if __name__ == "__main__":
#     if embeddings_data is not None:
#         print("\n--- Prueba Búsqueda Semántica ---")
#         sem_results = semantic_search("problema con mi factura")
#         print(sem_results)
#         sem_results_2 = semantic_search("internet lento")
#         print(sem_results_2)
#     if processed_transcripts_map is not None:
#         print("\n--- Prueba Búsqueda Keyword ---")
#         key_results = keyword_search("activar") # Buscar la palabra exacta "activar"
#         print(key_results)
#         key_results_2 = keyword_search("plan")
#         print(key_results_2)
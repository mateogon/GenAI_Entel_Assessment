import os
import sys
import re
from typing import List, Dict, Tuple, Optional

# Añadir el directorio raíz al path para importar desde 'app' y 'scripts'
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from app.openai_utils import get_embedding # Necesario para obtener embedding de la query

# --- Qdrant Imports ---
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import Filter, FieldCondition, MatchText, PointStruct

# --- Qdrant Config ---
# Es mejor pasar el cliente Qdrant a través de dependencias FastAPI en main.py
# pero por simplicidad en este módulo, lo inicializamos aquí.
# Asegúrate que estas variables de entorno o valores coinciden con tu config.
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
COLLECTION_NAME = "transcripts_prod"

try:
    qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    print(f"(search_logic) Cliente Qdrant conectado a {QDRANT_HOST}:{QDRANT_PORT}")
except Exception as e:
    print(f"Error CRÍTICO (search_logic): No se pudo conectar a Qdrant en {QDRANT_HOST}:{QDRANT_PORT}. Las búsquedas fallarán. Error: {e}")
    # Podríamos asignar None al cliente y chequearlo en las funciones,
    # o dejar que las funciones fallen si el cliente no está disponible.
    qdrant_client = None # Indicar que la conexión falló

# --- Funciones de Búsqueda (Usando Qdrant) ---

def semantic_search(query: str, top_n: int = 5) -> List[Tuple[str, float]]:
    """Realiza búsqueda semántica usando Qdrant y embeddings de OpenAI."""
    if qdrant_client is None:
        print("Error: Cliente Qdrant no disponible.")
        return []
    if not query:
        return []

    print(f"Realizando búsqueda semántica (Qdrant) para: '{query}'")
    query_embedding = get_embedding(query) # Obtener embedding para la consulta

    if query_embedding is None:
        print("Error: No se pudo obtener embedding para la consulta.")
        return []

    try:
        search_result = qdrant_client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_embedding,
            limit=top_n,
            with_payload=True # <--- NECESITAMOS EL PAYLOAD
        )
        results = []
        for hit in search_result:
            # Extraer el ID original del payload
            original_id = hit.payload.get("original_id", str(hit.id)) # Fallback al UUID si no está
            score = float(hit.score)
            results.append((original_id, score)) # <-- Devolver el ID original

        print(f"Resultados encontrados (Qdrant Semántica): {len(results)}")
        return results

    except Exception as e:
        print(f"Error durante la búsqueda semántica en Qdrant: {type(e).__name__} - {e}")
        return []

def keyword_search(query: str, top_n: int = 10) -> List[str]:
    """Realiza búsqueda por palabra clave usando filtros de Qdrant (búsqueda de texto)."""
    if qdrant_client is None:
        print("Error: Cliente Qdrant no disponible.")
        return []
    if not query:
        return []

    print(f"Realizando búsqueda por palabra clave (Qdrant Filter) para: '{query}'")

    try:
        scroll_result = qdrant_client.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=Filter(
                must=[FieldCondition(key="full_text", match=MatchText(text=query))]
            ),
            limit=top_n,
            with_payload=True, # <--- ¡NECESITAMOS EL PAYLOAD!
            with_vectors=False
        )
        # Extraer el ID original del payload de cada hit
        results = [hit.payload.get("original_id", str(hit.id)) for hit in scroll_result[0] if hit.payload] # <-- Extraer de payload

        print(f"Resultados encontrados (Qdrant Keyword): {len(results)}")
        return results

    except Exception as e:
        print(f"Error durante la búsqueda por palabra clave en Qdrant: {type(e).__name__} - {e}")
        return []

# Mantener el bloque de ejemplo comentado o eliminarlo si no se usa
# if __name__ == "__main__":
#     if qdrant_client:
#         print("\n--- Prueba Búsqueda Semántica (Qdrant) ---")
#         sem_results = semantic_search("problema con mi factura")
#         print(sem_results)
#         sem_results_2 = semantic_search("internet lento")
#         print(sem_results_2)
#
#         print("\n--- Prueba Búsqueda Keyword (Qdrant) ---")
#         # Nota: Keyword search con MatchText es sensible a mayúsculas/minúsculas por defecto
#         key_results = keyword_search("activar")
#         print(key_results)
#         key_results_2 = keyword_search("plan")
#         print(key_results_2)
#     else:
#          print("Cliente Qdrant no disponible, no se pueden ejecutar pruebas.")
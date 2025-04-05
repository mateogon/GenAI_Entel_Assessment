"""
Propósito: Funciones para realizar búsquedas en Qdrant.
Incluye búsqueda semántica mediante embeddings y búsqueda por palabra clave utilizando filtros.
"""

import os
import sys
import re
from typing import List, Dict, Tuple, Optional

# Permite acceder a módulos en el directorio raíz
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from app.openai_utils import get_embedding

# Importa clases y métodos de Qdrant para consultas y filtrado
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import Filter, FieldCondition, MatchText, PointStruct

# Configuración de Qdrant; se conecta utilizando variables de entorno o valores por defecto
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
COLLECTION_NAME = "transcripts_prod"

try:
    qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    print(f"(search_logic) Cliente Qdrant conectado a {QDRANT_HOST}:{QDRANT_PORT}")
except Exception as e:
    print(f"Error CRÍTICO (search_logic): No se pudo conectar a Qdrant en {QDRANT_HOST}:{QDRANT_PORT}. Error: {e}")
    qdrant_client = None  # Indica fallo en la conexión; las búsquedas no estarán disponibles

def semantic_search(query: str, top_n: int = 5) -> List[Tuple[str, float]]:
    """Realiza búsqueda semántica devolviendo transcripciones y sus puntuaciones."""
    if qdrant_client is None:
        print("Error: Cliente Qdrant no disponible.")
        return []
    if not query:
        return []

    print(f"Realizando búsqueda semántica para: '{query}'")
    query_embedding = get_embedding(query)
    if query_embedding is None:
        print("Error: No se pudo obtener embedding para la consulta.")
        return []

    try:
        search_result = qdrant_client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_embedding,
            limit=top_n,
            with_payload=True  # Se requiere el payload para extraer el ID original
        )
        results = []
        for hit in search_result:
            # Se prefiere el ID original del payload; si no existe, se usa el UUID del hit
            original_id = hit.payload.get("original_id", str(hit.id))
            results.append((original_id, float(hit.score)))
        print(f"Resultados encontrados (Semántica): {len(results)}")
        return results

    except Exception as e:
        print(f"Error durante la búsqueda semántica: {type(e).__name__} - {e}")
        return []

def keyword_search(query: str, top_n: int = 10) -> List[str]:
    """Realiza búsqueda por palabra clave utilizando filtros de Qdrant."""
    if qdrant_client is None:
        print("Error: Cliente Qdrant no disponible.")
        return []
    if not query:
        return []

    print(f"Realizando búsqueda por palabra clave para: '{query}'")
    try:
        scroll_result = qdrant_client.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=Filter(
                must=[FieldCondition(key="full_text", match=MatchText(text=query))]
            ),
            limit=top_n,
            with_payload=True,  # Se requiere el payload para extraer el ID original
            with_vectors=False
        )
        # Extrae el ID original de cada resultado filtrado
        results = [hit.payload.get("original_id", str(hit.id)) for hit in scroll_result[0] if hit.payload]
        print(f"Resultados encontrados (Keyword): {len(results)}")
        return results

    except Exception as e:
        print(f"Error durante la búsqueda por palabra clave: {type(e).__name__} - {e}")
        return []

# Bloque de ejemplo para pruebas locales (comentado para evitar ejecución en producción)
# if __name__ == "__main__":
#     if qdrant_client:
#         print("\n--- Prueba Búsqueda Semántica ---")
#         print(semantic_search("problema con mi factura"))
#         print(semantic_search("internet lento"))
#
#         print("\n--- Prueba Búsqueda Keyword ---")
#         print(keyword_search("activar"))
#         print(keyword_search("plan"))
#     else:
#         print("Cliente Qdrant no disponible, no se pueden ejecutar pruebas.")

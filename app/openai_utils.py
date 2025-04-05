"""
Módulo de integración con la API de OpenAI.
Proporciona funciones para obtener embeddings y respuestas de chat de forma
síncrona y asíncrona, incluyendo lógica de reintentos y manejo de errores.
"""

import os
import time
import openai
import asyncio
from openai import AsyncOpenAI
from dotenv import load_dotenv
from typing import List, Optional, Dict
import random

# Cargar configuración del entorno
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
ENABLE_OPENAI_CALLS = os.getenv("ENABLE_OPENAI_CALLS", "false").lower() == "true"

if ENABLE_OPENAI_CALLS and not api_key:
    raise ValueError("ENABLE_OPENAI_CALLS=true pero OPENAI_API_KEY no encontrada.")

# Inicializar clientes para llamadas síncronas y asíncronas
sync_client: Optional[openai.OpenAI] = None
async_client: Optional[AsyncOpenAI] = None

if ENABLE_OPENAI_CALLS:
    try:
        sync_client = openai.OpenAI(api_key=api_key)
        print("Cliente OpenAI SÍNCRONO inicializado (LLAMADAS REALES HABILITADAS).")
    except Exception as e:
        print(f"Error inicializando cliente OpenAI SÍNCRONO: {e}")
        sync_client = None

    try:
        async_client = AsyncOpenAI(api_key=api_key)
        print("Cliente OpenAI ASÍNCRONO inicializado (LLAMADAS REALES HABILITADAS).")
    except Exception as e:
        print(f"Error inicializando cliente OpenAI ASÍNCRONO: {e}")
        async_client = None
else:
    print("Clientes OpenAI DESHABILITADOS (modo simulación).")

# Constantes para el manejo de reintentos
MAX_RETRIES = 3
RETRY_DELAY = 2  # segundos

# -----------------------------------------------------------------------------
# Función para obtener embedding de un texto (síncrono)
# -----------------------------------------------------------------------------
def get_embedding(text: str, model="text-embedding-3-small") -> Optional[List[float]]:
    """Retorna el embedding de un texto usando el modelo especificado."""
    if not api_key:
        print("Error: No hay API key para obtener embeddings.")
        return None

    current_client = sync_client
    if not current_client:
        try:
            current_client = openai.OpenAI(api_key=api_key)
        except Exception as e:
            print(f"Error creando cliente local para embeddings: {e}")
            return None

    if not text or not isinstance(text, str):
        return None
    text = text.replace("\n", " ")
    attempts = 0
    while attempts < MAX_RETRIES:
        try:
            response = current_client.embeddings.create(input=[text], model=model)
            return response.data[0].embedding
        except openai.RateLimitError as e:
            attempts += 1
            print(f"Rate limit en embeddings síncronos. Reintentando en {RETRY_DELAY}s... ({attempts}/{MAX_RETRIES})")
            if attempts >= MAX_RETRIES:
                return None
            time.sleep(RETRY_DELAY + random.uniform(0, 0.5))
        except openai.APIConnectionError as e:
            attempts += 1
            print(f"Error de conexión en embeddings síncronos: {e}. Reintentando... ({attempts}/{MAX_RETRIES})")
            if attempts >= MAX_RETRIES:
                return None
            time.sleep(RETRY_DELAY + random.uniform(0, 0.5))
        except openai.APIStatusError as e:
            print(f"Error de API (Código {e.status_code}) en embeddings síncronos: {e.message}")
            if 400 <= e.status_code < 500 and e.status_code != 429:
                return None
            attempts += 1
            if attempts >= MAX_RETRIES:
                return None
            time.sleep(RETRY_DELAY + random.uniform(0, 0.5))
        except openai.APIError as e:
            print(f"Error general de API en embeddings síncronos: {e}")
            return None
        except Exception as e:
            print(f"Error inesperado en embeddings síncronos para '{text[:50]}...': {type(e).__name__} - {e}")
            return None
    print(f"No se pudo obtener embedding (síncrono) para '{text[:50]}...' tras {MAX_RETRIES} intentos.")
    return None

# -----------------------------------------------------------------------------
# Función para obtener embeddings en lote (síncrono)
# -----------------------------------------------------------------------------
def get_embeddings_batch(texts: List[str], model="text-embedding-3-small") -> Optional[List[Optional[List[float]]]]:
    """
    Obtiene embeddings para una lista de textos en una única llamada.
    Devuelve una lista con los embeddings correspondientes a cada texto válido.
    """
    if not api_key:
        print("Error: No hay API key para obtener embeddings batch.")
        return None

    if not texts:
        return []  # Lista vacía si no hay textos.
    valid_texts = [str(t).replace("\n", " ") for t in texts if t and isinstance(t, str)]
    if not valid_texts:
        print("Advertencia: No hay textos válidos en el lote.")
        return [None] * len(texts)

    current_client = sync_client
    if not current_client:
        try:
            current_client = openai.OpenAI(api_key=api_key)
            print("(Batch) Cliente local creado para embeddings.")
        except Exception as e:
            print(f"Error creando cliente para embeddings batch: {e}")
            return None

    attempts = 0
    while attempts < MAX_RETRIES:
        try:
            print(f"Intentando obtener embeddings para {len(valid_texts)} textos (Intento {attempts + 1})...")
            response = current_client.embeddings.create(input=valid_texts, model=model)
            # Mapear cada índice del input a su embedding
            embeddings_map: Dict[int, List[float]] = {}
            for item in response.data:
                embeddings_map[item.index] = item.embedding

            result_list = [embeddings_map.get(i) for i in range(len(valid_texts))]
            print(f"Embeddings obtenidos para {len(result_list)} textos válidos.")
            return result_list

        except openai.RateLimitError as e:
            attempts += 1
            print(f"Rate limit en embeddings batch síncronos. Reintentando en {RETRY_DELAY}s... ({attempts}/{MAX_RETRIES})")
            if attempts >= MAX_RETRIES:
                return None
            time.sleep(RETRY_DELAY + random.uniform(0, 0.5))
        except openai.APIConnectionError as e:
            attempts += 1
            print(f"Error de conexión en embeddings batch síncronos: {e}. Reintentando... ({attempts}/{MAX_RETRIES})")
            if attempts >= MAX_RETRIES:
                return None
            time.sleep(RETRY_DELAY + random.uniform(0, 0.5))
        except openai.APIStatusError as e:
            print(f"Error de API (Código {e.status_code}) en embeddings batch síncronos: {e.message}")
            if 400 <= e.status_code < 500 and e.status_code != 429:
                return None
            attempts += 1
            if attempts >= MAX_RETRIES:
                return None
            time.sleep(RETRY_DELAY + random.uniform(0, 0.5))
        except openai.APIError as e:
            print(f"Error general de API en embeddings batch síncronos: {e}")
            return None
        except Exception as e:
            print(f"Error inesperado en embeddings batch síncronos: {type(e).__name__} - {e}")
            return None

    print(f"No se pudieron obtener embeddings batch (síncrono) tras {MAX_RETRIES} intentos.")
    return None

# -----------------------------------------------------------------------------
# Función asíncrona para obtener completions de chat
# -----------------------------------------------------------------------------
async def get_chat_completion_async(prompt: str, model="gpt-4o-mini", max_tokens=60, temperature=0.2) -> Optional[str]:
    """
    Retorna una respuesta de chat de forma asíncrona.
    Si las llamadas reales están deshabilitadas, se simula una respuesta.
    """
    if not ENABLE_OPENAI_CALLS:
        print(f"Simulación ASYNC para modelo {model}.")
        print(f"Prompt (primeros 100 caracteres): {prompt[:100]}...")
        await asyncio.sleep(random.uniform(0.1, 0.3))
        if "Temas principales:" in prompt:
            simulated_topics = ["Conectividad Internet", "Facturación", "Cambio de Plan", "Soporte Técnico", "Consulta General"]
            random.shuffle(simulated_topics)
            response = ", ".join(simulated_topics[:random.randint(2, 3)])
            print(f"Simulación ASYNC de temas: {response}")
            return response
        elif "Categoría:" in prompt:
            try:
                from app.analysis_logic import CLASSIFICATION_CATEGORIES
                categories_to_use = CLASSIFICATION_CATEGORIES
            except ImportError:
                print("No se pudieron importar categorías; usando lista interna.")
                categories_to_use = ["Problemas Técnicos", "Soporte Comercial", "Solicitudes Administrativas", "Consultas Generales", "Reclamos"]
            response = random.choice(categories_to_use)
            print(f"Simulación ASYNC de clasificación: {response}")
            return response
        else:
            response = f"Respuesta simulada ASYNC para prompt que inicia con: {prompt[:30]}..."
            print(f"Simulación ASYNC genérica: {response}")
            return response

    if not async_client:
        print("Error: Cliente asíncrono no inicializado.")
        return None

    attempts = 0
    messages = [{"role": "user", "content": prompt}]

    while attempts < MAX_RETRIES:
        try:
            print(f"Llamada ASYNC a OpenAI (modelo {model}) - Intento {attempts + 1}")
            start_call_time = time.time()
            response = await async_client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                n=1,
                stop=None,
            )
            duration = time.time() - start_call_time
            content = response.choices[0].message.content.strip()
            print(f"Respuesta ASYNC recibida en {duration:.2f}s: {content[:100]}...")
            if response.usage:
                usage = response.usage
                print(f"Uso de tokens: Prompt={usage.prompt_tokens}, Completación={usage.completion_tokens}, Total={usage.total_tokens}")
            return content
        except openai.RateLimitError as e:
            attempts += 1
            print(f"Rate limit en chat ASYNC. Reintentando en {RETRY_DELAY}s... ({attempts}/{MAX_RETRIES})")
            if attempts >= MAX_RETRIES:
                return None
            await asyncio.sleep(RETRY_DELAY + random.uniform(0, 0.5))
        except openai.APIConnectionError as e:
            attempts += 1
            print(f"Error de conexión en chat ASYNC: {e}. Reintentando... ({attempts}/{MAX_RETRIES})")
            if attempts >= MAX_RETRIES:
                return None
            await asyncio.sleep(RETRY_DELAY + random.uniform(0, 0.5))
        except openai.APIStatusError as e:
            print(f"Error de API (Código {e.status_code}) en chat ASYNC: {e.message}")
            if 400 <= e.status_code < 500 and e.status_code != 429:
                print("Error no recuperable en chat ASYNC.")
                return None
            attempts += 1
            print(f"Error 5xx en chat ASYNC. Reintentando... ({attempts}/{MAX_RETRIES})")
            if attempts >= MAX_RETRIES:
                return None
            await asyncio.sleep(RETRY_DELAY + random.uniform(0, 0.5))
        except openai.APIError as e:
            print(f"Error general en chat ASYNC: {e}")
            return None
        except Exception as e:
            print(f"Error inesperado en chat ASYNC: {type(e).__name__} - {e}")
            return None

    print(f"No se pudo obtener chat completion ASYNC tras {MAX_RETRIES} intentos.")
    return None

# -----------------------------------------------------------------------------
# Función de chat síncrona (para compatibilidad, no se usa en análisis)
# -----------------------------------------------------------------------------
def get_chat_completion(prompt: str, model="gpt-4o-mini", max_tokens=60, temperature=0.2) -> Optional[str]:
    """Retorna una respuesta de chat de forma síncrona."""
    if not ENABLE_OPENAI_CALLS:
        print(f"Simulación SYNC para modelo {model}.")
        return "Respuesta simulada SYNC"

    if not sync_client:
        print("Error: Cliente síncrono no inicializado.")
        return None

    attempts = 0
    messages = [{"role": "user", "content": prompt}]
    while attempts < MAX_RETRIES:
        try:
            print(f"Llamada SYNC a OpenAI (modelo {model}) - Intento {attempts + 1}")
            response = sync_client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                n=1,
                stop=None
            )
            content = response.choices[0].message.content.strip()
            return content
        except openai.RateLimitError as e:
            attempts += 1
            print(f"Rate limit en chat SYNC. Reintentando... ({attempts}/{MAX_RETRIES})")
            if attempts >= MAX_RETRIES:
                return None
            time.sleep(RETRY_DELAY + random.uniform(0, 0.5))
        except Exception as e:
            print(f"Error inesperado en chat SYNC: {type(e).__name__} - {e}")
            return None
    print(f"No se pudo obtener chat completion SYNC tras {MAX_RETRIES} intentos.")
    return None

# -----------------------------------------------------------------------------
# Bloque de pruebas asíncronas (para validar funciones)
# -----------------------------------------------------------------------------
async def run_async_tests():
    print("\nPrueba ASYNC: Extracción de temas")
    try:
        from app.analysis_logic import TOPIC_EXTRACTION_PROMPT_TEMPLATE
        topic_prompt = TOPIC_EXTRACTION_PROMPT_TEMPLATE.format(transcript_text="...")
        topic_sim = await get_chat_completion_async(topic_prompt)
        print(f"Resultado ASYNC Temas: {topic_sim}")
    except ImportError:
        print("No se pudo importar TOPIC_EXTRACTION_PROMPT_TEMPLATE.")

    print("\nPrueba ASYNC: Clasificación")
    try:
        from app.analysis_logic import CLASSIFICATION_PROMPT_TEMPLATE, CLASSIFICATION_CATEGORIES
        class_prompt = CLASSIFICATION_PROMPT_TEMPLATE.format(
            categories=", ".join(CLASSIFICATION_CATEGORIES), transcript_text="..."
        )
        class_sim = await get_chat_completion_async(class_prompt)
        print(f"Resultado ASYNC Clasificación: {class_sim}")
    except ImportError:
        print("No se pudo importar CLASSIFICATION_PROMPT_TEMPLATE.")

if __name__ == "__main__":
    print("\nPrueba SYNC: Embedding individual")
    embedding = get_embedding("Texto de prueba individual.")
    if embedding:
        print(f"Embedding individual OK (Dim: {len(embedding)})")
    else:
        print("Fallo en embedding individual.")

    print("\nPrueba SYNC: Embeddings batch")
    test_batch_texts = [
        "Este es el primer texto del lote.",
        "Segundo texto, un poco diferente.",
        "Y el tercero para completar.",
        "",  # Texto vacío
        None,  # Texto inválido
        "Último texto válido."
    ]
    embeddings_batch = get_embeddings_batch(test_batch_texts)
    if embeddings_batch:
        print(f"Embeddings batch OK. Recibidos {len(embeddings_batch)} resultados.")
        for i, emb in enumerate(embeddings_batch):
            if emb:
                print(f"Texto {i}: OK (Dim: {len(emb)})")
            else:
                print(f"Texto {i}: Faltante o inválido")
    else:
        print("Fallo en embeddings batch.")

    # Descomentar para ejecutar pruebas ASYNC de chat completion
    # async def run_async_tests_wrapper():
    #     print("\nEjecutando pruebas ASYNC de chat completion")
    #     await get_chat_completion_async("Test async prompt")
    # asyncio.run(run_async_tests_wrapper())

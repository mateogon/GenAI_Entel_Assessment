# app/openai_utils.py
import os
import time
import openai
import asyncio # Importar asyncio
from openai import AsyncOpenAI # Importar el cliente asíncrono
from dotenv import load_dotenv
from typing import List, Optional,Dict
import random

# --- Configuración Inicial ---
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
ENABLE_OPENAI_CALLS = os.getenv("ENABLE_OPENAI_CALLS", "false").lower() == "true"

if ENABLE_OPENAI_CALLS and not api_key:
    raise ValueError("ENABLE_OPENAI_CALLS=true pero OPENAI_API_KEY no encontrada.")

# --- Clientes OpenAI ---
sync_client: Optional[openai.OpenAI] = None
async_client: Optional[AsyncOpenAI] = None # Cliente asíncrono

if ENABLE_OPENAI_CALLS:
    try:
        # Inicializar cliente síncrono (puede ser usado por get_embedding o scripts)
        sync_client = openai.OpenAI(api_key=api_key)
        print("Cliente OpenAI SÍNCRONO inicializado (LLAMADAS REALES HABILITADAS).")
    except Exception as e:
        print(f"Error CRÍTICO inicializando cliente OpenAI SÍNCRONO: {e}")
        sync_client = None

    try:
        # Inicializar cliente ASÍNCRONO (para la API FastAPI)
        async_client = AsyncOpenAI(api_key=api_key)
        print("Cliente OpenAI ASÍNCRONO inicializado (LLAMADAS REALES HABILITADAS).")
    except Exception as e:
        print(f"Error CRÍTICO inicializando cliente OpenAI ASÍNCRONO: {e}")
        async_client = None
else:
    print("Clientes OpenAI DESHABILITADOS (modo simulación).")

# --- Constantes para Reintentos ---
MAX_RETRIES = 3
RETRY_DELAY = 2 # segundos

# --- Funciones ---

# get_embedding se mantiene síncrono por ahora (usado principalmente en script de generación)
def get_embedding(text: str, model="text-embedding-3-small") -> Optional[List[float]]:
    """Obtiene el embedding para un texto dado usando la API de OpenAI (Síncrono)."""
    if not api_key:
        print("Error: No hay API key disponible para obtener embeddings.")
        return None
    # Usar cliente síncrono o crear uno local si es necesario
    current_client = sync_client
    if not current_client:
         try:
             current_client = openai.OpenAI(api_key=api_key)
         except Exception as e:
             print(f"Error creando cliente local síncrono de OpenAI para embeddings: {e}")
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
            print(f"Rate limit alcanzado (Embeddings Sync). Reintentando en {RETRY_DELAY}s... ({attempts}/{MAX_RETRIES})")
            if attempts >= MAX_RETRIES: return None
            time.sleep(RETRY_DELAY + random.uniform(0, 0.5))
        except openai.APIConnectionError as e:
            attempts += 1
            print(f"Error de conexión OpenAI (Embeddings Sync): {e}. Reintentando... ({attempts}/{MAX_RETRIES})")
            if attempts >= MAX_RETRIES: return None
            time.sleep(RETRY_DELAY + random.uniform(0, 0.5))
        except openai.APIStatusError as e:
             print(f"Error de API OpenAI (Embeddings Sync - Código {e.status_code}): {e.message}")
             if 400 <= e.status_code < 500 and e.status_code != 429: return None
             attempts += 1
             if attempts >= MAX_RETRIES: return None
             time.sleep(RETRY_DELAY + random.uniform(0, 0.5))
        except openai.APIError as e:
            print(f"Error general de API OpenAI (Embeddings Sync): {e}")
            return None
        except Exception as e:
            print(f"Error inesperado (Embeddings Sync) para '{text[:50]}...': {type(e).__name__} - {e}")
            return None
    print(f"Fallo al obtener embedding (Sync) para '{text[:50]}...' después de {MAX_RETRIES} intentos.")
    return None

def get_embeddings_batch(texts: List[str], model="text-embedding-3-small") -> Optional[List[Optional[List[float]]]]:
    """
    Obtiene embeddings para una lista de textos en una sola llamada API (Síncrono).
    Devuelve una lista de embeddings en el mismo orden que los textos de entrada.
    Si un embedding falla específicamente (raro), puede contener None en esa posición.
    Si toda la llamada falla, devuelve None.
    """
    if not api_key:
        print("Error: No hay API key disponible para obtener embeddings batch.")
        return None

    # Validar entrada
    if not texts:
        return [] # Devolver lista vacía si la entrada está vacía
    valid_texts = [str(t).replace("\n", " ") for t in texts if t and isinstance(t, str)]
    if not valid_texts:
        print("Advertencia: No hay textos válidos en el lote para generar embeddings.")
        # Devolver una lista de Nones de la longitud original si es necesario mantener índices
        return [None] * len(texts)

    # Usar cliente síncrono o crear uno local si es necesario
    current_client = sync_client
    if not current_client:
         try:
             current_client = openai.OpenAI(api_key=api_key)
             print("(Batch) Cliente local síncrono creado.")
         except Exception as e:
             print(f"Error creando cliente local síncrono para embeddings batch: {e}")
             return None

    attempts = 0
    while attempts < MAX_RETRIES:
        try:
            # >>> LLAMADA BATCH A LA API <<<
            print(f"Intentando obtener embeddings para lote de {len(valid_texts)} textos (Intento {attempts + 1})...")
            response = current_client.embeddings.create(input=valid_texts, model=model)

            # Procesar respuesta: OpenAI garantiza que response.data tiene el mismo orden que la entrada
            embeddings_map: Dict[int, List[float]] = {}
            for item in response.data:
                embeddings_map[item.index] = item.embedding

            # Construir lista de resultados manteniendo el orden original (incluyendo Nones para entradas inválidas)
            # Asumimos que los índices de valid_texts corresponden a los índices en embeddings_map
            result_list = [embeddings_map.get(i) for i in range(len(valid_texts))]

            # Si necesitamos mapear de vuelta a la longitud original incluyendo inválidos:
            # final_results = []
            # valid_idx = 0
            # for original_text in texts:
            #     if original_text and isinstance(original_text, str):
            #         final_results.append(result_list[valid_idx])
            #         valid_idx += 1
            #     else:
            #         final_results.append(None)
            # return final_results

            print(f"  Embeddings obtenidos para lote de {len(result_list)} textos.")
            # Devolvemos solo los embeddings para los textos válidos procesados
            return result_list

        except openai.RateLimitError as e:
            attempts += 1
            print(f"Rate limit alcanzado (Embeddings Batch Sync). Reintentando en {RETRY_DELAY}s... ({attempts}/{MAX_RETRIES})")
            if attempts >= MAX_RETRIES: return None
            time.sleep(RETRY_DELAY + random.uniform(0, 0.5))
        except openai.APIConnectionError as e:
            attempts += 1
            print(f"Error de conexión OpenAI (Embeddings Batch Sync): {e}. Reintentando... ({attempts}/{MAX_RETRIES})")
            if attempts >= MAX_RETRIES: return None
            time.sleep(RETRY_DELAY + random.uniform(0, 0.5))
        except openai.APIStatusError as e:
             print(f"Error de API OpenAI (Embeddings Batch Sync - Código {e.status_code}): {e.message}")
             if 400 <= e.status_code < 500 and e.status_code != 429: return None # Error no recuperable
             attempts += 1
             if attempts >= MAX_RETRIES: return None
             time.sleep(RETRY_DELAY + random.uniform(0, 0.5))
        except openai.APIError as e: # Otros errores generales de la API
            print(f"Error general de API OpenAI (Embeddings Batch Sync): {e}")
            return None
        except Exception as e:
            print(f"Error inesperado al obtener embeddings batch (Sync): {type(e).__name__} - {e}")
            return None

    print(f"Fallo al obtener embeddings batch (Sync) después de {MAX_RETRIES} intentos.")
    return None

# --- get_chat_completion asincrona ---
async def get_chat_completion_async(prompt: str, model="gpt-4o-mini", max_tokens=60, temperature=0.2) -> Optional[str]:
    """
    Obtiene una completación de chat usando la API de OpenAI DE FORMA ASÍNCRONA.
    Respeta el flag ENABLE_OPENAI_CALLS para simular la respuesta si es 'false'.
    """
    if not ENABLE_OPENAI_CALLS:
        # --- Bloque de Simulación ASÍNCRONO ---
        print(f"--- SIMULANDO LLAMADA ASYNC A OpenAI ({model}) ---")
        print(f"--- Prompt recibido (primeros 100 chars): {prompt[:100]}...")
        await asyncio.sleep(random.uniform(0.1, 0.3)) # Simular espera ASÍNCRONA

        # Simular respuesta basada en palabras clave en el prompt (igual que antes)
        if "Temas principales:" in prompt:
            simulated_topics = ["Conectividad Internet", "Facturación", "Cambio de Plan", "Soporte Técnico", "Consulta General"]
            random.shuffle(simulated_topics)
            response = ", ".join(simulated_topics[:random.randint(2,3)])
            print(f"--- Respuesta Simulada ASYNC (Temas): {response} ---")
            return response
        elif "Categoría:" in prompt:
            try:
                 from app.analysis_logic import CLASSIFICATION_CATEGORIES
                 categories_to_use = CLASSIFICATION_CATEGORIES
            except ImportError:
                 print("Advertencia: No se pudieron importar categorías de analysis_logic. Usando lista interna.")
                 categories_to_use = ["Problemas Técnicos", "Soporte Comercial", "Solicitudes Administrativas", "Consultas Generales", "Reclamos"]
            response = random.choice(categories_to_use)
            print(f"--- Respuesta Simulada ASYNC (Clasificación): {response} ---")
            return response
        else:
            response = f"Respuesta simulada ASYNC para prompt que empieza con: {prompt[:30]}..."
            print(f"--- Respuesta Simulada ASYNC (Genérica): {response} ---")
            return response
        # --- Fin Bloque de Simulación ASÍNCRONO ---

    # --- Bloque de Llamada Real ASÍNCRONO ---
    if not async_client:
        print("Error CRÍTICO: Cliente ASÍNCRONO de OpenAI no inicializado (se esperaban llamadas reales).")
        return None

    attempts = 0
    messages = [{"role": "user", "content": prompt}]

    while attempts < MAX_RETRIES:
        try:
            print(f"--- LLAMADA ASÍNCRONA A OpenAI ({model}) - Intento {attempts + 1} ---")
            start_call_time = time.time()
            # >>> USA EL CLIENTE ASÍNCRONO Y AWAIT <<<
            response = await async_client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                n=1,
                stop=None,
            )
            end_call_time = time.time()
            call_duration = end_call_time - start_call_time
            content = response.choices[0].message.content.strip()
            usage = response.usage
            print(f"--- Respuesta Real ASYNC Recibida ({call_duration:.2f}s): {content[:100]}... ---")
            if usage:
                 print(f"--- Uso de Tokens ASYNC: Prompt={usage.prompt_tokens}, Completación={usage.completion_tokens}, Total={usage.total_tokens} ---")
            return content
        except openai.RateLimitError as e:
            attempts += 1
            print(f"Rate limit alcanzado (Chat Async). Reintentando en {RETRY_DELAY}s... ({attempts}/{MAX_RETRIES})")
            if attempts >= MAX_RETRIES: return None
            await asyncio.sleep(RETRY_DELAY + random.uniform(0, 0.5)) # Espera ASÍNCRONA
        except openai.APIConnectionError as e:
            attempts += 1
            print(f"Error de conexión OpenAI (Chat Async): {e}. Reintentando... ({attempts}/{MAX_RETRIES})")
            if attempts >= MAX_RETRIES: return None
            await asyncio.sleep(RETRY_DELAY + random.uniform(0, 0.5)) # Espera ASÍNCRONA
        except openai.APIStatusError as e:
            print(f"Error de API OpenAI (Chat Async - Código {e.status_code}): {e.message}")
            if 400 <= e.status_code < 500 and e.status_code != 429:
                 print("Error 4xx no recuperable (Async).")
                 return None
            attempts += 1
            print(f"Error 5xx del servidor OpenAI (Async). Reintentando... ({attempts}/{MAX_RETRIES})")
            if attempts >= MAX_RETRIES: return None
            await asyncio.sleep(RETRY_DELAY + random.uniform(0, 0.5)) # Espera ASÍNCRONA
        except openai.APIError as e:
            print(f"Error general de API OpenAI (Chat Async): {e}")
            return None
        except Exception as e:
            print(f"Error inesperado al obtener chat completion ASÍNCRONO: {type(e).__name__} - {e}")
            return None

    print(f"Fallo al obtener chat completion ASÍNCRONO después de {MAX_RETRIES} intentos.")
    return None


# --- VERSIÓN SÍNCRONA original (se mantiene por si acaso, pero no se usará en análisis) ---
def get_chat_completion(prompt: str, model="gpt-4o-mini", max_tokens=60, temperature=0.2) -> Optional[str]:
    """Obtiene una completación de chat usando la API de OpenAI (Síncrono)."""
    if not ENABLE_OPENAI_CALLS:
        # Simulación síncrona (igual que antes)
        print(f"--- SIMULANDO LLAMADA SYNC A OpenAI ({model}) ---")
        # ... (código de simulación síncrona omitido por brevedad, es el original) ...
        return "Respuesta simulada SYNC"

    if not sync_client:
        print("Error CRÍTICO: Cliente SÍNCRONO OpenAI no inicializado.")
        return None

    attempts = 0
    messages = [{"role": "user", "content": prompt}]
    while attempts < MAX_RETRIES:
        try:
            print(f"--- LLAMADA SÍNCRONA A OpenAI ({model}) - Intento {attempts + 1} ---")
            response = sync_client.chat.completions.create( model=model, messages=messages, max_tokens=max_tokens, temperature=temperature, n=1, stop=None )
            content = response.choices[0].message.content.strip()
            # ... (logs síncronos) ...
            return content
        except openai.RateLimitError as e:
            attempts += 1
            print(f"Rate limit (Sync). Reintentando... ({attempts}/{MAX_RETRIES})")
            if attempts >= MAX_RETRIES: return None
            time.sleep(RETRY_DELAY + random.uniform(0, 0.5)) # Sleep SÍNCRONO
        # ... (resto de manejo de errores síncronos omitido por brevedad) ...
        except Exception as e:
            print(f"Error inesperado (Sync): {type(e).__name__} - {e}")
            return None
    print(f"Fallo chat completion (Sync) después de {MAX_RETRIES} intentos.")
    return None


# --- Bloque de Ejemplo (modificado para probar async) ---
async def run_async_tests():
    # Prueba específica de simulación de temas (ASYNC)
    print("\n--- Probando simulación ASYNC de temas ---")
    # Necesita importar el template de análisis_logic si está en el mismo nivel
    try:
        from app.analysis_logic import TOPIC_EXTRACTION_PROMPT_TEMPLATE
        topic_prompt = TOPIC_EXTRACTION_PROMPT_TEMPLATE.format(transcript_text="...")
        topic_sim = await get_chat_completion_async(topic_prompt)
        print(f"  Resultado Simulación ASYNC Temas: {topic_sim}")
    except ImportError:
        print("  No se pudo importar TOPIC_EXTRACTION_PROMPT_TEMPLATE para prueba.")

    # Prueba específica de simulación de clasificación (ASYNC)
    print("\n--- Probando simulación ASYNC de clasificación ---")
    try:
        from app.analysis_logic import CLASSIFICATION_PROMPT_TEMPLATE, CLASSIFICATION_CATEGORIES
        class_prompt = CLASSIFICATION_PROMPT_TEMPLATE.format(categories=", ".join(CLASSIFICATION_CATEGORIES), transcript_text="...")
        class_sim = await get_chat_completion_async(class_prompt)
        print(f"  Resultado Simulación ASYNC Clasificación: {class_sim}")
    except ImportError:
         print("  No se pudo importar CLASSIFICATION_PROMPT_TEMPLATE para prueba.")

if __name__ == "__main__":
    # Probar embedding individual (Sync)
    print("\n--- Probando get_embedding (Sync Individual) ---")
    embedding = get_embedding("Texto de prueba individual.")
    if embedding: print(f"  Embedding individual OK (Dim: {len(embedding)})")
    else: print("  Fallo embedding individual.")

    # >>> Probar embedding BATCH (Sync) <<<
    print("\n--- Probando get_embeddings_batch (Sync Batch) ---")
    test_batch_texts = [
        "Este es el primer texto del lote.",
        "Segundo texto, un poco diferente.",
        "Y el tercero para completar.",
        "", # Texto vacío
        None, # Texto inválido
        "Último texto válido."
    ]
    embeddings_batch = get_embeddings_batch(test_batch_texts)
    if embeddings_batch:
        print(f"  Embeddings batch OK. Recibidos {len(embeddings_batch)} resultados.")
        for i, emb in enumerate(embeddings_batch):
            if emb:
                print(f"    Texto {i}: OK (Dim: {len(emb)})")
            else:
                # Esto reflejaría los textos válidos para los que se obtuvo embedding
                print(f"    Texto {i}: Faltante o Inválido en origen")
    else:
        print("  Fallo al obtener embeddings batch.")

    # Ejecutar pruebas asíncronas de chat completion
    async def run_async_tests():
        print("\n--- Ejecutando pruebas ASÍNCRONAS de chat completion ---")
        # ... (pruebas async sin cambios) ...
        await get_chat_completion_async("Test async prompt")
    # asyncio.run(run_async_tests()) # Comentado para enfocarnos en batch sync
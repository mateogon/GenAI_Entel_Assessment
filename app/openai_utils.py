import os
import time
import openai
from dotenv import load_dotenv
from typing import List, Optional
import random # Para simular respuestas

# --- Configuración Inicial ---

# Cargar variables de entorno (API Key y flag de simulación)
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
# Lee la variable de entorno. Si no existe, por defecto es "false". Convierte a minúsculas y compara.
ENABLE_OPENAI_CALLS = os.getenv("ENABLE_OPENAI_CALLS", "false").lower() == "true"

# Validar API Key solo si las llamadas están habilitadas
if ENABLE_OPENAI_CALLS and not api_key:
    raise ValueError("ENABLE_OPENAI_CALLS está configurado como 'true' en el archivo .env, pero no se encontró la variable de entorno OPENAI_API_KEY.")

# Inicializar cliente OpenAI SOLO si las llamadas están habilitadas
client = None
if ENABLE_OPENAI_CALLS:
    try:
        client = openai.OpenAI(api_key=api_key)
        print("Cliente OpenAI inicializado (LLAMADAS REALES HABILITADAS).")
    except Exception as e:
        # Si falla la inicialización con llamadas habilitadas, es un problema.
        print(f"Error CRÍTICO inicializando cliente OpenAI con llamadas habilitadas: {e}")
        # Podrías querer detener la aplicación aquí o manejarlo de otra forma
        client = None
        # raise e # Opcional: relanzar la excepción para detener la app
else:
    print("Cliente OpenAI DESHABILITADO (modo simulación). No se realizarán llamadas reales a Chat Completions.")


# --- Constantes para Reintentos ---
MAX_RETRIES = 3
RETRY_DELAY = 2 # segundos

# --- Funciones ---

def get_embedding(text: str, model="text-embedding-3-small") -> Optional[List[float]]:
    """
    Obtiene el embedding para un texto dado usando la API de OpenAI.
    """
    # Usamos una instancia temporal del cliente aquí para asegurar que se intente
    # incluso si el cliente global no se inicializó (porque ENABLE_OPENAI_CALLS era false).
    if not api_key:
        print("Error: No hay API key disponible para obtener embeddings.")
        return None
    try:
        local_client = openai.OpenAI(api_key=api_key)
    except Exception as e:
        print(f"Error creando cliente local de OpenAI para embeddings: {e}")
        return None

    if not text or not isinstance(text, str):
        # print("Advertencia: Texto inválido o vacío para embedding.")
        return None

    # OpenAI requiere que no haya saltos de línea
    text = text.replace("\n", " ")
    attempts = 0

    while attempts < MAX_RETRIES:
        try:
            # print(f"Intentando obtener embedding para: '{text[:50]}...' (Intento {attempts + 1})")
            response = local_client.embeddings.create(input=[text], model=model)
            embedding_data = response.data[0].embedding
            # print(f"  Embedding obtenido (Dim: {len(embedding_data)})")
            return embedding_data
        except openai.RateLimitError as e:
            attempts += 1
            print(f"Rate limit alcanzado (Embeddings). Reintentando en {RETRY_DELAY}s... (Intento {attempts}/{MAX_RETRIES})")
            if attempts >= MAX_RETRIES:
                print("Máximo de reintentos alcanzado por RateLimit.")
                return None
            time.sleep(RETRY_DELAY + random.uniform(0, 0.5)) # Añadir jitter
        except openai.APIConnectionError as e:
            attempts += 1
            print(f"Error de conexión OpenAI (Embeddings): {e}. Reintentando... (Intento {attempts}/{MAX_RETRIES})")
            if attempts >= MAX_RETRIES:
                print("Máximo de reintentos alcanzado por ConnectionError.")
                return None
            time.sleep(RETRY_DELAY + random.uniform(0, 0.5))
        except openai.APIStatusError as e: # Captura errores HTTP específicos
             print(f"Error de API OpenAI (Embeddings - Código {e.status_code}): {e.message}")
             # No reintentar en errores 4xx (excepto 429 que es RateLimit)
             if 400 <= e.status_code < 500 and e.status_code != 429:
                 return None
             # Reintentar errores 5xx (problemas del servidor de OpenAI)
             attempts += 1
             if attempts >= MAX_RETRIES:
                 print("Máximo de reintentos alcanzado por APIStatusError.")
                 return None
             time.sleep(RETRY_DELAY + random.uniform(0, 0.5))
        except openai.APIError as e: # Otros errores generales de la API
            print(f"Error general de API OpenAI (Embeddings): {e}")
            return None # No reintentar estos por defecto
        except Exception as e:
            print(f"Error inesperado al obtener embedding para '{text[:50]}...': {type(e).__name__} - {e}")
            return None # No reintentar errores inesperados

    # Si sale del bucle sin éxito
    print(f"Fallo al obtener embedding para '{text[:50]}...' después de {MAX_RETRIES} intentos.")
    return None


def get_chat_completion(prompt: str, model="gpt-4o-mini", max_tokens=60, temperature=0.2) -> Optional[str]:
    """
    Obtiene una completación de chat usando la API de OpenAI.
    Respeta el flag ENABLE_OPENAI_CALLS para simular la respuesta si es 'false'.
    """
    if not ENABLE_OPENAI_CALLS:
        # --- Bloque de Simulación ---
        print(f"--- SIMULANDO LLAMADA A OpenAI ({model}) ---")
        print(f"--- Prompt recibido (primeros 100 chars): {prompt[:100]}...")
        time.sleep(random.uniform(0.1, 0.3)) # Simular pequeña latencia

        # Simular respuesta basada en palabras clave en el prompt
        if "Temas principales:" in prompt:
            simulated_topics = ["Conectividad Internet", "Facturación", "Cambio de Plan", "Soporte Técnico", "Consulta General"]
            random.shuffle(simulated_topics)
            response = ", ".join(simulated_topics[:random.randint(2,3)])
            print(f"--- Respuesta Simulada (Temas): {response} ---")
            return response
        elif "Categoría:" in prompt:
            # Lazy import para evitar dependencia circular si se llama desde otro módulo
            try:
                 from app.analysis_logic import CLASSIFICATION_CATEGORIES
                 categories_to_use = CLASSIFICATION_CATEGORIES
            except ImportError:
                 # Fallback si hay importación circular
                 print("Advertencia: No se pudieron importar categorías de analysis_logic. Usando lista interna.")
                 categories_to_use = [
                     "Problemas Técnicos", "Soporte Comercial", "Solicitudes Administrativas",
                     "Consultas Generales", "Reclamos"
                 ]
            response = random.choice(categories_to_use)
            print(f"--- Respuesta Simulada (Clasificación): {response} ---")
            return response
        else:
            # Respuesta genérica si no coincide con patrones conocidos
            response = f"Respuesta simulada para prompt que empieza con: {prompt[:30]}..."
            print(f"--- Respuesta Simulada (Genérica): {response} ---")
            return response
        # --- Fin Bloque de Simulación ---

    # --- Bloque de Llamada Real ---
    if not client:
        # Esto solo debería ocurrir si ENABLE_OPENAI_CALLS es true pero la inicialización falló.
        print("Error CRÍTICO: Cliente OpenAI no inicializado (se esperaban llamadas reales).")
        return None

    attempts = 0
    # Asumimos un prompt simple de usuario para el rol
    messages = [{"role": "user", "content": prompt}]

    while attempts < MAX_RETRIES:
        try:
            print(f"--- LLAMADA REAL A OpenAI ({model}) - Intento {attempts + 1} ---")
            start_call_time = time.time()
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature, # Usar valor pasado como argumento
                n=1, # Pedir solo una completación
                stop=None, # No definir secuencia de parada por defecto
            )
            end_call_time = time.time()
            call_duration = end_call_time - start_call_time
            # Extraer contenido de la respuesta
            content = response.choices[0].message.content.strip()
            # Log de información útil (opcional)
            usage = response.usage
            print(f"--- Respuesta Real Recibida ({call_duration:.2f}s): {content[:100]}... ---")
            if usage:
                 print(f"--- Uso de Tokens: Prompt={usage.prompt_tokens}, Completación={usage.completion_tokens}, Total={usage.total_tokens} ---")
            return content
        except openai.RateLimitError as e:
            attempts += 1
            print(f"Rate limit alcanzado (Chat). Reintentando en {RETRY_DELAY}s... (Intento {attempts}/{MAX_RETRIES})")
            if attempts >= MAX_RETRIES:
                print("Máximo de reintentos alcanzado por RateLimit.")
                return None
            time.sleep(RETRY_DELAY + random.uniform(0, 0.5)) # Añadir jitter
        except openai.APIConnectionError as e:
            attempts += 1
            print(f"Error de conexión OpenAI (Chat): {e}. Reintentando... (Intento {attempts}/{MAX_RETRIES})")
            if attempts >= MAX_RETRIES:
                print("Máximo de reintentos alcanzado por ConnectionError.")
                return None
            time.sleep(RETRY_DELAY + random.uniform(0, 0.5))
        except openai.APIStatusError as e: # Captura errores HTTP específicos
            print(f"Error de API OpenAI (Chat - Código {e.status_code}): {e.message}")
            # No reintentar en errores 4xx (excepto 429 que es RateLimit)
            if 400 <= e.status_code < 500 and e.status_code != 429:
                 # Errores comunes: 400 (bad request - prompt inválido?), 401 (auth), 404 (modelo no encontrado?)
                 print("Error 4xx no recuperable. Revisar prompt o configuración.")
                 return None
            # Reintentar errores 5xx (problemas del servidor de OpenAI)
            attempts += 1
            print(f"Error 5xx del servidor OpenAI. Reintentando... (Intento {attempts}/{MAX_RETRIES})")
            if attempts >= MAX_RETRIES:
                 print("Máximo de reintentos alcanzado por APIStatusError (5xx).")
                 return None
            time.sleep(RETRY_DELAY + random.uniform(0, 0.5))
        except openai.APIError as e: # Otros errores generales de la API
            print(f"Error general de API OpenAI (Chat): {e}")
            return None # No reintentar estos por defecto
        except Exception as e:
            print(f"Error inesperado al obtener chat completion: {type(e).__name__} - {e}")
            return None # No reintentar errores inesperados

    # Si sale del bucle sin éxito
    print(f"Fallo al obtener chat completion después de {MAX_RETRIES} intentos.")
    return None

# --- Bloque de Ejemplo (para probar el módulo directamente) ---
if __name__ == "__main__":
    # Probar embeddings (siempre intentará llamada real si hay API Key)
    print("\n--- Probando get_embedding ---")
    test_text_emb = "Este texto es para probar embeddings."
    embedding = get_embedding(test_text_emb)
    if embedding:
        print(f"  Embedding OK (primeros 5): {embedding[:5]} (Dim: {len(embedding)})")
    else:
        print("  Fallo al obtener embedding.")

    # Prueba específica de simulación de temas
    print("\n--- Probando simulación de temas (si está habilitada) ---")
    from app.analysis_logic import TOPIC_EXTRACTION_PROMPT_TEMPLATE # Asume que está definido
    topic_prompt = TOPIC_EXTRACTION_PROMPT_TEMPLATE.format(transcript_text="...")
    topic_sim = get_chat_completion(topic_prompt)
    print(f"  Resultado Simulación Temas: {topic_sim}")

    # Prueba específica de simulación de clasificación
    print("\n--- Probando simulación de clasificación (si está habilitada) ---")
    from app.analysis_logic import CLASSIFICATION_PROMPT_TEMPLATE, CLASSIFICATION_CATEGORIES # Asume definidos
    class_prompt = CLASSIFICATION_PROMPT_TEMPLATE.format(categories=", ".join(CLASSIFICATION_CATEGORIES), transcript_text="...")
    class_sim = get_chat_completion(class_prompt)
    print(f"  Resultado Simulación Clasificación: {class_sim}")
import streamlit as st
import requests
import os
from dotenv import load_dotenv
from PIL import Image  # Para cargar la imagen del logo

# --- Configuración Inicial ---
# Carga las variables de entorno desde el archivo .env
load_dotenv()
# URL base de la API.
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
# Ruta al archivo de imagen del logo
LOGO_PATH = "assets/entel.webp"

# --- Configuración de la Página ---
try:
    page_icon = Image.open(LOGO_PATH)
except FileNotFoundError:
    page_icon = "🤖"

st.set_page_config(
    layout="wide",
    page_title="Análisis Transcripciones Entel",
    page_icon=page_icon
)

# --- Funciones para Llamar a la API ---
# Estas funciones encapsulan las llamadas a la API y manejan errores.

def search_api(query: str, search_type: str = "semantic", top_n: int = 5):
    """
    Llama al endpoint de búsqueda de la API.

    Args:
        query: La consulta de búsqueda.
        search_type: Tipo de búsqueda ("semantic" o "keyword").
        top_n: Número máximo de resultados a devolver.

    Returns:
        Una lista de resultados de búsqueda (diccionarios), o None si hay un error.
    """
    url = f"{API_BASE_URL}/search/"
    payload = {"query": query, "search_type": search_type, "top_n": top_n}
    try:
        response = requests.post(url, json=payload, timeout=25)  # Timeout más largo
        response.raise_for_status()  # Lanza una excepción para códigos de error HTTP
        return response.json().get("results", [])
    except requests.exceptions.Timeout:
        st.error(f"Error: Tiempo de espera agotado al contactar {url}. La API podría estar ocupada.")
        return None
    except requests.exceptions.ConnectionError:
        st.error(f"Error: No se pudo conectar a la API en {API_BASE_URL}. Verifica que el backend esté ejecutándose.")
        return None
    except requests.exceptions.RequestException as e:
        st.error(f"Error en la llamada a la API de búsqueda ({url}): {e}")
        return None
    except Exception as e:
        st.error(f"Error inesperado durante la búsqueda: {e}")
        return None

def analyze_api(endpoint: str, transcript_id: str = None, text: str = None):
    """
    Llama a los endpoints de análisis de la API (temas, clasificación).

    Args:
        endpoint: El endpoint a llamar ("topics" o "classify").
        transcript_id: ID de la transcripción a analizar (opcional).
        text: Texto de la transcripción a analizar (opcional).

    Returns:
        Un diccionario con los resultados del análisis, o None si hay un error.
    """
    payload = {}
    if transcript_id:
        payload["transcript_id"] = transcript_id
    elif text:
        payload["text"] = text
    else:
        st.error("Error interno: Se llamó a la función de análisis sin ID ni texto.")
        return None

    url = f"{API_BASE_URL}/analyze/{endpoint}/"
    try:
        response = requests.post(url, json=payload, timeout=45)  # Timeout más largo
        response.raise_for_status()
        return response.json()
    except requests.exceptions.Timeout:
        st.error(f"Error: Tiempo de espera agotado al contactar {url}. El análisis podría tardar.")
        return None
    except requests.exceptions.ConnectionError:
        st.error(f"Error: No se pudo conectar a la API en {API_BASE_URL}. Verifica que el backend esté ejecutándose.")
        return None
    except requests.exceptions.RequestException as e:
        st.error(f"Error en la llamada a la API de análisis ({url}): {e}")
        return None
    except Exception as e:
        st.error(f"Error inesperado durante el análisis ({endpoint}): {e}")
        return None

# --- Barra Lateral ---
# Contiene el logo, información del estado de la API y modo de operación.

# Muestra el logo en la barra lateral
st.sidebar.image(LOGO_PATH)
st.sidebar.title("Opciones y Estado")

st.sidebar.divider()  # Separador visual

# Estado de la API
st.sidebar.subheader("Estado de la API Backend")
status_placeholder = st.sidebar.empty()  # Placeholder para mensajes de estado

try:
    status_res = requests.get(f"{API_BASE_URL}/status", timeout=5)
    if status_res.status_code == 200:
        status_data = status_res.json()
        status_placeholder.success(f"API Conectada ({API_BASE_URL})")
        st.sidebar.metric("Transcripciones Cargadas", status_data.get("transcripts_loaded", "N/A"))
    else:
        status_placeholder.error(f"API No Responde ({status_res.status_code})")
except requests.exceptions.ConnectionError:
    status_placeholder.error(f"No se pudo conectar a {API_BASE_URL}")
except Exception as e:
    status_placeholder.error(f"Error al verificar API: {e}")

st.sidebar.divider()

# Modo de Operación (Simulación/Real)
st.sidebar.subheader("Modo de Operación (Análisis)")
sim_mode = not (os.getenv("ENABLE_OPENAI_CALLS", "false").lower() == "true")
if sim_mode:
    st.sidebar.warning("🟡 **Modo Simulación:**\nAnálisis (temas, clasificación) con respuestas simuladas (OpenAI deshabilitado).")
else:
    st.sidebar.info("🟢 **Modo Real:**\nAnálisis realizado via OpenAI.")

st.sidebar.divider()
st.sidebar.caption("© Entel Corp - Herramienta Interna")  # Pie de página

# --- Interfaz Principal ---
# Título principal y descripción general

st.title("🤖 Analizador de Transcripciones de Llamadas Entel")
st.markdown("Interfaz para buscar y analizar transcripciones usando la API de procesamiento semántico.")

# --- Sección 1: Búsqueda de Transcripciones ---
# Permite buscar transcripciones por palabra clave o semántica.

with st.container():  # Agrupa los elementos
    st.header("🔍 Búsqueda de Transcripciones")
    st.markdown("Encuentra transcripciones relevantes por palabra clave o significado semántico.")

    # Campo de entrada para la consulta de búsqueda
    search_query = st.text_input("Introduce tu consulta:", placeholder="Ej: problema con la factura, baja del servicio...", key="search_query")

    # Columnas para alinear los selectores de tipo de búsqueda y número de resultados
    search_col1, search_col2 = st.columns([1, 1])
    with search_col1:
        search_type = st.selectbox("Tipo de Búsqueda:", ["semantic", "keyword"], index=0, key="search_type", help="`semantic`: Busca por significado similar. `keyword`: Busca coincidencias exactas de palabras.")
    with search_col2:
        top_n_search = st.number_input("Máx. Resultados:", min_value=1, max_value=30, value=5, key="top_n_search", help="Número de transcripciones más relevantes a mostrar.")

    # Botón de búsqueda
    if st.button("Buscar Transcripciones", key="search_button", type="primary"):
        if search_query:
            search_message = st.empty()  # Placeholder para mostrar el mensaje de "Buscando..."
            search_message.info(f"Buscando '{search_query}' (Tipo: {search_type})... ⏳")
            with st.spinner("Realizando búsqueda..."):  # Muestra un spinner mientras la API está buscando
                results = search_api(search_query, search_type, top_n_search)

            if results is not None:
                search_message.empty()  # Limpia el mensaje "Buscando..."
                if results:
                    st.success(f"Se encontraron {len(results)} resultados:")
                    # Muestra los resultados en columnas
                    results_cols = st.columns(len(results) if len(results) <= 5 else 5)  # Máximo 5 columnas
                    for i, res in enumerate(results):
                        col = results_cols[i % 5]  # Distribuye los resultados entre las columnas
                        with col:
                            st.markdown(f"**ID: {res['transcript_id']}**")
                            if 'score' in res:
                                st.markdown(f"Score: `{res['score']:.4f}`")  # Formatea el score
                            st.markdown("---")
                else:
                    st.warning("No se encontraron resultados para tu consulta.")
            else:
                search_message.empty()  # Limpia el mensaje de buscando si hay un error (ya mostrado por search_api)

        else:
            st.warning("Por favor, introduce una consulta para buscar.")

st.divider()  # Separador visual

# --- Sección 2: Análisis Individual de Transcripciones ---
# Permite analizar una transcripción específica para extraer temas o clasificarla.

with st.container():  # Agrupa los elementos
    st.header("📄 Análisis Individual de Transcripción")
    st.markdown("Obtén temas principales o clasifica una transcripción específica.")

    # Selector para elegir si se analiza por ID o por texto
    analysis_option = st.radio(
        "Seleccionar Transcripción por:",
        ["ID de Transcripción", "Texto Directo"],
        key="analysis_option",
        horizontal=True,
        help="Puedes analizar una transcripción existente por su ID o pegar un texto nuevo."
    )

    transcript_id_input = None
    text_input = None

    # Campo de entrada para el ID de la transcripción
    if analysis_option == "ID de Transcripción":
        transcript_id_input = st.text_input(
            "ID de la Transcripción:",
            placeholder="Ej: 01, 31, sample_45, ... (Obtenido de la búsqueda)",
            key="transcript_id_input"
        )
    # Área de texto para pegar la transcripción
    else:
        text_input = st.text_area(
            "Pega el texto de la transcripción aquí:",
            height=150,
            key="text_input",
            placeholder="Ej: 'Cliente: Hola, tengo un problema con mi última factura...'"
        )

    # --- Botones de Análisis ---
    # Permiten extraer temas o clasificar la transcripción.
    analyze_col1, analyze_col2 = st.columns(2)

    with analyze_col1:
        # Botón para extraer temas
        if st.button("📊 Extraer Temas", key="analyze_topics_button"):
            if transcript_id_input or text_input:
                analysis_msg_topics = st.empty()  # Placeholder para mensajes
                analysis_msg_topics.info("Extrayendo temas... ⏳")
                with st.spinner("Analizando temas..."):  # Spinner durante el análisis
                    analysis_result = analyze_api("topics", transcript_id=transcript_id_input, text=text_input)

                analysis_msg_topics.empty()  # Limpia el mensaje
                if analysis_result:
                    topics = analysis_result.get("topics", [])
                    if topics:
                        st.success("Temas principales extraídos:")
                        # Muestra los temas como una lista con markdown
                        md_topics = "\n".join([f"- {topic}" for topic in topics])
                        st.markdown(md_topics)
                    elif "error" in analysis_result:
                        st.error(f"Error de la API al extraer temas: {analysis_result['error']}")
                    else:
                        st.warning("No se pudieron extraer temas (respuesta vacía o inesperada).")
                # Error handling in analyze_api
            else:
                st.warning("Por favor, introduce un ID o texto para analizar.")

    with analyze_col2:
        # Botón para clasificar la transcripción
        if st.button("🏷️ Clasificar Transcripción", key="analyze_classify_button"):
            if transcript_id_input or text_input:
                analysis_msg_classify = st.empty()  # Placeholder
                analysis_msg_classify.info("Clasificando transcripción... ⏳")
                with st.spinner("Clasificando..."):  # Spinner
                    analysis_result = analyze_api("classify", transcript_id=transcript_id_input, text=text_input)

                analysis_msg_classify.empty()  # Limpia el mensaje
                if analysis_result:
                    category = analysis_result.get("category")
                    if category:
                        st.success(f"Categoría Predicha: **{category}**")
                    elif "error" in analysis_result:
                        st.error(f"Error de la API al clasificar: {analysis_result['error']}")
                    else:
                        st.warning("No se pudo clasificar (respuesta vacía o inesperada).")
                 # Error handling in analyze_api
            else:
                st.warning("Por favor, introduce un ID o texto para analizar.")
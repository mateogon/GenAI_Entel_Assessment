# 🤖 Assessment AI Engineer - Análisis Semántico de Transcripciones (Entel GenAI)

Este repositorio contiene la solución para el assessment de AI Engineer del equipo GenAI de Entel. El proyecto implementa un sistema backend para analizar semánticamente transcripciones de llamadas de atención al cliente, utilizando la API de OpenAI de forma optimizada.

## 📝 Descripción General

El sistema permite:

*   🔍 **Búsqueda:** Encontrar transcripciones relevantes mediante palabras clave o significado semántico.
*   📊 **Extracción de Temas:** Identificar los temas principales discutidos en una conversación.
*   🏷️ **Clasificación Automática:** Asignar categorías predefinidas (ej: "Problemas Técnicos", "Soporte Comercial") a las llamadas.

Se ha desarrollado priorizando la funcionalidad y una gestión eficiente del presupuesto de OpenAI ($5 USD).

## ✨ Características Principales

*   **Backend API:** Construido con FastAPI, proporcionando endpoints RESTful para búsqueda y análisis.
*   **Procesamiento NLP:**
    *   Utiliza `text-embedding-3-small` de OpenAI para búsqueda semántica (generación inicial offline, queries on-demand).
    *   Utiliza `gpt-4o-mini` de OpenAI para extracción de temas y clasificación.
*   **Optimización de Costos:**
    *   Generación única de embeddings para el dataset inicial.
    *   Modo de simulación controlable (`ENABLE_OPENAI_CALLS` en `.env`) para desarrollo y pruebas sin consumir créditos de OpenAI para análisis.
*   **Preprocesamiento:** Limpieza de texto, manejo de formato `.txt` de transcripciones y anonimización de PII (incluyendo RUT chileno) con Presidio.
*   **Interfaz Opcional:** Frontend básico desarrollado con Streamlit para interactuar con la API.
*   **Documentación API:** Auto-generada vía Swagger UI (`/docs`) y ReDoc (`/redoc`).

## 🛠️ Configuración del Entorno

Sigue estos pasos para configurar el proyecto localmente en Windows (usando PowerShell):

1.  **Clonar el Repositorio:**
    ```powershell
    git clone <url-del-repositorio>
    cd <nombre-del-repositorio> # Ej: GenAI_Entel_Assessment
    ```

2.  **Crear y Activar Entorno Virtual:**
    ```powershell
    python -m venv .venv
    # Podrías necesitar ajustar la política de ejecución:
    # Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
    .venv\Scripts\Activate.ps1
    # Deberías ver (.venv) al inicio de tu prompt
    ```

3.  **Instalar Dependencias:**
    ```powershell
    pip install -r requirements.txt
    ```

4.  **Descargar Modelo de Lenguaje spaCy:** (Necesario para Presidio)
    ```powershell
    python -m spacy download es_core_news_md
    # Si usaste 'es_core_news_sm' en el código, descarga ese en su lugar
    ```

5.  **Configurar API Key de OpenAI:**
    *   Crea un archivo llamado `.env` en la raíz del proyecto.
    *   Abre `.env` con un editor y añade tu clave de API:
        ```dotenv
        OPENAI_API_KEY=sk-TuClaveRealDeOpenAIaqui
        ```
    *   **(Importante)** Control del Modo de Simulación:
        *   Para **deshabilitar** las llamadas reales a GPT (modo simulación, **recomendado para probar**):
            ```dotenv
            # Añade o asegura que esta línea esté presente y en false
            ENABLE_OPENAI_CALLS=false
            ```
        *   Para **habilitar** las llamadas reales a GPT (consumirá créditos):
            ```dotenv
            # Cambia o añade esta línea
            ENABLE_OPENAI_CALLS=true
            ```
        *   Si la línea `ENABLE_OPENAI_CALLS` no existe, por defecto operará en **modo simulación** (`false`).

6.  **Colocar Datos Crudos:**
    *   Asegúrate de que las 100 transcripciones en formato `.txt` (ej: `sample_01.txt`) estén dentro de la carpeta `data/raw/`.

## 🚀 Ejecución

1.  **Preparar Datos (Ejecutar solo una vez inicialmente):**
    *   Abre PowerShell en la raíz del proyecto (con el venv activado).
    *   **Paso 1: Preprocesamiento (Limpieza y Anonimización):**
        ```powershell
        python scripts/preprocess_data.py
        ```
        Esto leerá los `.txt` de `data/raw/`, los limpiará/anonimizará y guardará archivos `.json` en `data/processed/`.
    *   **Paso 2: Generación de Embeddings (Usa API OpenAI - bajo costo):**
        ```powershell
        python scripts/generate_embeddings_openai.py
        ```
        Esto generará los archivos `data/embeddings.npy` y `data/id_map.pkl`. Si ya existen, te preguntará si deseas sobrescribirlos (lo cual volvería a usar la API de OpenAI). Responde 'y' solo si es necesario.

2.  **Iniciar el Backend (API FastAPI):**
    *   En la misma terminal (con venv activado):
        ```powershell
        uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
        ```
    *   La API estará disponible en `http://localhost:8000`.
    *   Puedes acceder a la documentación interactiva en `http://localhost:8000/docs`.

3.  **Iniciar el Frontend (Streamlit - Opcional):**
    *   Abre OTRA terminal PowerShell en la raíz del proyecto.
    *   Activa el entorno virtual: `.venv\Scripts\Activate.ps1`
    *   Ejecuta:
        ```powershell
        streamlit run frontend.py
        ```
    *   Se abrirá automáticamente una pestaña en tu navegador (usualmente `http://localhost:8501`).

## ⚙️ Arquitectura y Decisiones Técnicas

*   **Backend:** FastAPI por su rendimiento, facilidad de uso y auto-documentación OpenAPI.
*   **Procesamiento de Texto Crudo:** Scripts Python con `regex` para parsear el formato `.txt` y `Presidio` para una robusta anonimización de PII (incluyendo RUT chileno).
*   **Embeddings:** `text-embedding-3-small` de OpenAI por su balance costo/rendimiento. Se generan offline y se almacenan localmente (`.npy`) para minimizar costos de API en búsquedas posteriores. La similitud se calcula localmente (NumPy/Scikit-learn).
*   **Análisis (Temas/Clasificación):** `gpt-4o-mini` de OpenAI por su capacidad de seguir instrucciones (prompting) para estas tareas con una configuración rápida, usando prompts específicos para cada tarea.
*   **Gestión de Presupuesto:** La estrategia clave es la generación única de embeddings y el modo de simulación para análisis. Esto permite desarrollar y probar extensivamente sin apenas consumir el presupuesto. El gasto principal (y mínimo) ocurre solo al generar los embeddings iniciales o al activar explícitamente las llamadas reales para análisis.
*   **Escalabilidad:** La solución actual (carga en memoria) es adecuada para las 100 transcripciones del assessment. Para escalar a miles/millones, se recomendaría:
    *   Usar una base de datos vectorial (ej: ChromaDB, Weaviate, Pinecone) para los embeddings.
    *   Almacenar los textos procesados en una base de datos NoSQL o un Data Lake.
    *   Implementar procesamiento batch para embeddings y análisis si se usan APIs externas.
    *   Considerar alternativas open-source (ej: Sentence-Transformers, BERTopic, modelos de clasificación de Hugging Face) para eliminar la dependencia y costo de APIs externas a gran escala.
*   **Frontend:** Streamlit por su rapidez para crear interfaces de datos directamente desde Python.

## 📖 Documentación de la API

La documentación interactiva de la API está disponible automáticamente cuando el backend está corriendo:

*   **Swagger UI:** [http://localhost:8000/docs](http://localhost:8000/docs)
*   **ReDoc:** [http://localhost:8000/redoc](http://localhost:8000/redoc)

**Endpoints Principales:**

*   `POST /search/`: Realiza búsquedas.
    *   **Request Body:**
        ```json
        {
          "query": "texto a buscar",
          "search_type": "semantic" or "keyword",
          "top_n": 5 (opcional, default 5)
        }
        ```
    *   **Response Body (Éxito):**
        ```json
        {
          "results": [
            {"transcript_id": "sample_042", "score": 0.85}, // Score solo en semántica
            {"transcript_id": "sample_015", "score": 0.79},
            ...
          ]
        }
        ```
*   `POST /analyze/topics/`: Extrae temas.
    *   **Request Body:**
        ```json
        // Opción 1: Por ID
        {"transcript_id": "sample_007"}
        // Opción 2: Por Texto Directo
        {"text": "El cliente llamó por problemas con su factura..."}
        ```
    *   **Response Body (Éxito):**
        ```json
        {"topics": ["Problema Facturación", "Consulta Saldo"]}
        ```
*   `POST /analyze/classify/`: Clasifica la transcripción.
    *   **Request Body:** (Igual que para `/topics/`)
    *   **Response Body (Éxito):**
        ```json
        {"category": "Soporte Comercial"}
        ```
*   `GET /status`: Verifica el estado de la API.
    *   **Response Body (Éxito):**
        ```json
        {"status": "ok", "transcripts_loaded": 100}
        ```

## 💡 Posibles Mejoras Futuras

*   Implementar caché para respuestas de análisis de OpenAI.
*   Usar una base de datos vectorial para búsqueda semántica más eficiente a gran escala.
*   Integrar modelos open-source para reducir dependencia y costos.
*   Añadir más métricas y logging detallado.
*   Implementar pruebas automatizadas (unit, integration).
*   Mejorar la interfaz de usuario del frontend.


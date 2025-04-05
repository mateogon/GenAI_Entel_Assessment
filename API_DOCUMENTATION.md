# Documentación Detallada de la API - Análisis de Transcripciones

Esta documentación describe los endpoints disponibles en la API backend, incluyendo ejemplos de uso y respuestas esperadas. Para una referencia interactiva y completa de los esquemas, utiliza los enlaces auto-generados por FastAPI cuando el backend esté en ejecución:

*   **Swagger UI:** `http://localhost:8000/docs`
*   **ReDoc:** `http://localhost:8000/redoc`

---

## Endpoints Principales

### 1. Búsqueda de Transcripciones

*   **Endpoint:** `POST /search/`
*   **Descripción:** Permite buscar transcripciones indexadas en Qdrant utilizando búsqueda semántica (por significado) o por palabra clave (coincidencia exacta).
*   **Request Body (JSON):**
    ```json
    {
      "query": "string", // Texto o palabra clave a buscar
      "search_type": "semantic" | "keyword", // Tipo de búsqueda
      "top_n": int // (Opcional, default: 5) Máx. resultados a devolver (1-20)
    }
    ```
*   **Ejemplo de Llamada (`curl` - Búsqueda Semántica):**
    ```bash
    curl -X POST "http://localhost:8000/search/" \
         -H "Content-Type: application/json" \
         -d '{
           "query": "problemas con mi factura de este mes",
           "search_type": "semantic",
           "top_n": 3
         }'
    ```
*   **Respuesta Exitosa (200 OK - Semántica):**
    ```json
    {
      "results": [
        {"transcript_id": "sample_42", "score": 0.8134},
        {"transcript_id": "sample_88", "score": 0.7950},
        {"transcript_id": "sample_05", "score": 0.7712}
      ]
    }
    ```
    *(Nota: El `score` representa la similitud coseno, solo presente en búsquedas semánticas).*
*   **Respuesta Exitosa (200 OK - Keyword):**
    ```json
    {
      "results": [
        {"transcript_id": "sample_15", "score": null},
        {"transcript_id": "sample_63", "score": null}
      ]
    }
    ```

---

### 2. Extracción de Temas

*   **Endpoint:** `POST /analyze/topics/`
*   **Descripción:** Analiza el texto de una transcripción (identificada por su ID original o proporcionada directamente) para extraer los 2-3 temas principales discutidos. Utiliza `gpt-4o-mini` de OpenAI de forma asíncrona.
*   **Request Body (JSON):** Proporcionar *uno* de los siguientes:
    ```json
    // Opción 1: Por ID
    {"transcript_id": "sample_07"}

    // Opción 2: Por Texto Directo
    {"text": "El cliente llamó indicando que su servicio de internet no funciona desde ayer, probó reiniciando el router sin éxito."}
    ```
*   **Respuesta Exitosa (200 OK):**
    ```json
    {
      "topics": ["Falla Servicio Internet", "Reinicio Router", "Soporte Técnico"]
    }
    ```

---

### 3. Clasificación de Transcripción

*   **Endpoint:** `POST /analyze/classify/`
*   **Descripción:** Clasifica una transcripción (identificada por su ID original o proporcionada directamente) en una de las categorías predefinidas (e.g., `Problemas Técnicos`, `Soporte Comercial`). Utiliza `gpt-4o-mini` de OpenAI de forma asíncrona.
*   **Request Body (JSON):** Proporcionar *uno* de los siguientes:
    ```json
    // Opción 1: Por ID
    {"transcript_id": "sample_21"}

    // Opción 2: Por Texto Directo
    {"text": "Buenas tardes, necesito hacer un reclamo por un cobro que no reconozco en mi boleta."}
    ```
*   **Respuesta Exitosa (200 OK):**
    ```json
    {
      "category": "Reclamos" // O la categoría predicha
    }
    ```

---

### 4. Estado del Servicio

*   **Endpoint:** `GET /status`
*   **Descripción:** Devuelve información básica sobre el estado de la API y su conexión con la base de datos Qdrant, incluyendo el número de puntos (transcripciones) actualmente indexados.
*   **Respuesta Exitosa (200 OK):**
    ```json
    {
      "status": "ok",
      "qdrant_status": "conectado", // o "error", "no disponible"
      "qdrant_collection": "transcripts_prod",
      "collection_points_count": 100 // Número de puntos en Qdrant
    }
    ```

---

**Códigos de Estado Comunes:**

*   `200 OK`: La solicitud fue exitosa.
*   `404 Not Found`: Recurso no encontrado (e.g., `transcript_id` inválido al buscar texto).
*   `422 Unprocessable Entity`: Error de validación en los datos de entrada (e.g., falta `query`, `search_type` inválido). Detalles en el cuerpo de la respuesta.
*   `500 Internal Server Error`: Error inesperado en el servidor backend.
*   `502 Bad Gateway`: Error al comunicarse con un servicio externo (e.g., OpenAI API).
*   `503 Service Unavailable`: El servicio depende de una conexión (e.g., Qdrant) que no está disponible actualmente.
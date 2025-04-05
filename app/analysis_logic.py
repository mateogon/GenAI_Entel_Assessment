# app/analysis_logic.py
import os
import sys
from typing import List, Optional
import asyncio # Necesario si usamos await

# Añadir el directorio raíz al path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# >>> Importar la versión ASÍNCRONA <<<
from app.openai_utils import get_chat_completion_async

# --- Constantes (sin cambios) ---
CLASSIFICATION_CATEGORIES = [
    "Problemas Técnicos", "Soporte Comercial", "Solicitudes Administrativas",
    "Consultas Generales", "Reclamos"
]

# --- Prompts (sin cambios) ---
TOPIC_EXTRACTION_PROMPT_TEMPLATE = """
Analiza la siguiente transcripción de una llamada de atención al cliente.
Extrae los 2 o 3 temas o problemas principales discutidos.
Responde únicamente con una lista de temas breves separados por comas. No incluyas introducciones ni explicaciones.

Transcripción:
---
{transcript_text}
---
Temas principales:
"""

CLASSIFICATION_PROMPT_TEMPLATE = """
Clasifica la siguiente transcripción de atención al cliente en UNA de las siguientes categorías:
{categories}

Responde únicamente con el nombre exacto de la categoría elegida. No añadas ninguna otra palabra o puntuación.

Transcripción:
---
{transcript_text}
---
Categoría:
"""

# >>> Convertir a ASYNC DEF <<<
async def extract_topics(text: str) -> List[str]:
    """Extrae temas principales usando un LLM de OpenAI (ASÍNCRONO)."""
    if not text:
        return []
    print("(Async) Extrayendo temas...")
    prompt = TOPIC_EXTRACTION_PROMPT_TEMPLATE.format(transcript_text=text[:4000])
    # >>> USAR AWAIT y la función ASYNC <<<
    response = await get_chat_completion_async(prompt, model="gpt-4o-mini", max_tokens=50, temperature=0.1)

    if response:
        topics = [topic.strip() for topic in response.split(',') if topic.strip()]
        print(f"(Async) Temas extraídos: {topics}")
        return topics
    else:
        print("(Async) Error al extraer temas.")
        return []

# >>> Convertir a ASYNC DEF <<<
async def classify_transcript(text: str) -> Optional[str]:
    """Clasifica la transcripción usando un LLM de OpenAI (ASÍNCRONO)."""
    if not text:
        return None
    print("(Async) Clasificando transcripción...")
    categories_str = ", ".join(CLASSIFICATION_CATEGORIES)
    prompt = CLASSIFICATION_PROMPT_TEMPLATE.format(
        categories=categories_str,
        transcript_text=text[:4000]
    )
    # >>> USAR AWAIT y la función ASYNC <<<
    response = await get_chat_completion_async(prompt, model="gpt-4o-mini", max_tokens=20, temperature=0.0)

    if response:
        # La validación de la respuesta sigue igual
        if response in CLASSIFICATION_CATEGORIES:
            print(f"(Async) Categoría clasificada: {response}")
            return response
        else:
            print(f"(Async) Advertencia: Respuesta de clasificación no coincide ('{response}').")
            # Decidir si devolver la respuesta cruda o None en caso de no coincidencia
            return response # Devolver respuesta cruda por ahora
    else:
        print("(Async) Error al clasificar.")
        return None
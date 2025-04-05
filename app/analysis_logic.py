"""
Propósito: Funciones asíncronas para analizar transcripciones.
Se encarga de extraer temas y clasificar transcripciones usando LLMs de OpenAI.
"""

import os
import sys
from typing import List, Optional
import asyncio  # Necesario para el uso de await en operaciones asíncronas

# Permite acceder a módulos en el directorio raíz
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Importa la función asíncrona para llamar a OpenAI
from app.openai_utils import get_chat_completion_async

# Categorías predefinidas para la clasificación
CLASSIFICATION_CATEGORIES = [
    "Problemas Técnicos", "Soporte Comercial", "Solicitudes Administrativas",
    "Consultas Generales", "Reclamos"
]

# Template para la extracción de temas; se formatea con la transcripción
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

# Template para la clasificación; incluye las categorías y la transcripción
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

async def extract_topics(text: str) -> List[str]:
    """Extrae temas principales de una transcripción utilizando un LLM de OpenAI."""
    if not text:
        return []
    print("(Async) Extrayendo temas...")
    prompt = TOPIC_EXTRACTION_PROMPT_TEMPLATE.format(transcript_text=text[:4000])
    # Llamada asíncrona a la API de OpenAI para obtener la respuesta
    response = await get_chat_completion_async(prompt, model="gpt-4o-mini", max_tokens=50, temperature=0.1)

    if response:
        topics = [topic.strip() for topic in response.split(',') if topic.strip()]
        print(f"(Async) Temas extraídos: {topics}")
        return topics
    else:
        print("(Async) Error al extraer temas.")
        return []

async def classify_transcript(text: str) -> Optional[str]:
    """Clasifica una transcripción en una categoría predefinida usando un LLM de OpenAI."""
    if not text:
        return None
    print("(Async) Clasificando transcripción...")
    categories_str = ", ".join(CLASSIFICATION_CATEGORIES)
    prompt = CLASSIFICATION_PROMPT_TEMPLATE.format(
        categories=categories_str,
        transcript_text=text[:4000]
    )
    # Llamada asíncrona a la API para clasificación
    response = await get_chat_completion_async(prompt, model="gpt-4o-mini", max_tokens=20, temperature=0.0)

    if response:
        # Verifica si la respuesta coincide con alguna categoría predefinida
        if response in CLASSIFICATION_CATEGORIES:
            print(f"(Async) Categoría clasificada: {response}")
            return response
        else:
            print(f"(Async) Advertencia: Respuesta de clasificación no coincide ('{response}').")
            # Devuelve la respuesta cruda para revisión
            return response
    else:
        print("(Async) Error al clasificar.")
        return None

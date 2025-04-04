# app/analysis_logic.py
import os
import sys
from typing import List, Optional

# Añadir el directorio raíz al path para importar desde 'app'
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from app.openai_utils import get_chat_completion

# --- Constantes ---
CLASSIFICATION_CATEGORIES = [
    "Problemas Técnicos",
    "Soporte Comercial",  # Incluye facturación, planes
    "Solicitudes Administrativas",  # Cambios de datos, etc.
    "Consultas Generales",
    "Reclamos"
]

# --- Prompts ---
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

def extract_topics(text: str) -> List[str]:
    """Extrae temas principales usando un LLM de OpenAI."""
    if not text:
        return []
    print("Extrayendo temas...")
    prompt = TOPIC_EXTRACTION_PROMPT_TEMPLATE.format(transcript_text=text[:4000])  # Limitar longitud para evitar exceder tokens
    response = get_chat_completion(prompt, model="gpt-4o-mini", max_tokens=50, temperature=0.1)

    if response:
        topics = [topic.strip() for topic in response.split(',') if topic.strip()]
        print(f"Temas extraídos: {topics}")
        return topics
    else:
        print("Error al extraer temas.")
        return []

def classify_transcript(text: str) -> Optional[str]:
    """Clasifica la transcripción usando un LLM de OpenAI."""
    if not text:
        return None
    print("Clasificando transcripción...")
    categories_str = ", ".join(CLASSIFICATION_CATEGORIES)
    prompt = CLASSIFICATION_PROMPT_TEMPLATE.format(
        categories=categories_str,
        transcript_text=text[:4000]  # Limitar longitud
    )
    response = get_chat_completion(prompt, model="gpt-4o-mini", max_tokens=20, temperature=0.0)  # Temperatura 0 para máxima consistencia

    if response:
        if response in CLASSIFICATION_CATEGORIES:
            print(f"Categoría clasificada: {response}")
            return response
        else:
            print(f"Advertencia: Respuesta de clasificación no coincide con categorías esperadas ('{response}'). Devolviendo respuesta cruda.")
            return response
    else:
        print("Error al clasificar.")
        return None
"""
Propósito: Preprocesar transcripciones .txt aplicando limpieza y anonimización de información sensible.
Utiliza Presidio para detectar y reemplazar PII, y ejecuta el procesamiento de forma concurrente.
"""

import re
import os
import sys
import concurrent.futures
from typing import List, Dict, Any, Optional

# Se ajusta el path para acceder a los módulos del proyecto
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

# Intentar importar Presidio; si falla, se definen clases dummy para evitar errores
try:
    from presidio_analyzer import AnalyzerEngine, PatternRecognizer, RecognizerRegistry, Pattern
    from presidio_anonymizer import AnonymizerEngine
    from presidio_anonymizer.entities import OperatorConfig
    from presidio_analyzer.nlp_engine import NlpEngineProvider
    PRESIDIO_INSTALLED = True
except ImportError:
    print("ADVERTENCIA: Librerías Presidio no encontradas. La anonimización será omitida.")
    print("Instala: pip install presidio-analyzer presidio-anonymizer")
    PRESIDIO_INSTALLED = False
    class AnalyzerEngine: pass
    class AnonymizerEngine: pass
    class OperatorConfig: pass
    class PatternRecognizer: pass
    class RecognizerRegistry: pass
    class Pattern: pass

from scripts.load_data import load_raw_transcripts, save_processed_transcript, PROCESSED_DIR

# --- Configuración global para Presidio ---
analyzer = None
anonymizer = None
PRESIDIO_AVAILABLE = False

if PRESIDIO_INSTALLED:
    try:
        print("Inicializando Presidio Analyzer y Anonymizer...")
        # Inicializa el motor NLP para español con spaCy
        nlp_provider = NlpEngineProvider(nlp_configuration={
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": "es", "model_name": "es_core_news_md"}]
        })
        nlp_engine = nlp_provider.create_engine()

        # Configura AnalyzerEngine para el idioma español
        analyzer = AnalyzerEngine(nlp_engine=nlp_engine, supported_languages=["es"])
        analyzer.registry.load_predefined_recognizers(languages=["es"], nlp_engine=nlp_engine)
        print("Recognizers predeterminados para 'es' cargados.")

        # Definición y registro de patrones personalizados para RUT y números telefónicos
        rut_pattern_standard = Pattern(
            name="rut_pattern",
            regex=r"\b(\d{1,2}\.\d{3}\.\d{3}-[\dkK])\b",
            score=0.8
        )
        rut_pattern_dashes = Pattern(
            name="rut_pattern_dashes",
            regex=r"\b(\d{1,2}-\d{3}-\d{3}-[\dkK]?)\b",
            score=0.75
        )
        rut_pattern_spaces = Pattern(
            name="rut_pattern_spaces",
            regex=r"\b(\d{1,2}\s\d{3}\s\d{3}-[\dkK])\b",
            score=0.75
        )
        phone_pattern = Pattern(
            name="chile_phone_pattern",
            regex=r"\b(\+?56\s?[2-9][\s\d]{8,11})\b",
            score=0.7
        )
        phone_recognizer = PatternRecognizer(
            supported_entity="PHONE_NUMBER",
            name="ChileanPhoneRecognizer",
            patterns=[phone_pattern],
            context=["teléfono", "fono", "celular", "móvil", "llamar", "contacto"],
            supported_language="es"
        )
        email_pattern = Pattern(
            name="email_pattern",
            regex=r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
            score=0.9
        )
        email_recognizer = PatternRecognizer(
            supported_entity="EMAIL_ADDRESS",
            name="EmailRecognizer",
            patterns=[email_pattern],
            context=["email", "correo", "e-mail", "@"],
            supported_language="es"
        )
        rut_recognizer = PatternRecognizer(
            supported_entity="CL_RUT",
            name="ChileanRUTRecognizer",
            patterns=[rut_pattern_standard, rut_pattern_dashes, rut_pattern_spaces],
            context=["RUT", "rut", "Rol Único Tributario"],
            supported_language="es"
        )
        analyzer.registry.add_recognizer(rut_recognizer)
        analyzer.registry.add_recognizer(phone_recognizer)
        analyzer.registry.add_recognizer(email_recognizer)

        # Inicializa el motor de anonimización
        anonymizer = AnonymizerEngine()

        print("Presidio inicializado correctamente con reconocedor de RUT.")
        PRESIDIO_AVAILABLE = True

    except Exception as e:
        print(f"ADVERTENCIA: Falló la inicialización de Presidio. La anonimización será omitida. Error: {type(e).__name__} - {e}")
        analyzer = None
        anonymizer = None
        PRESIDIO_AVAILABLE = False
else:
    print("Presidio no está instalado. La anonimización será omitida.")

# --- Expresiones regulares para limpieza de texto ---
FILLER_WORDS_ES = re.compile(
    r'\b(eh|este|pues|bueno|o sea|¿sabes\?|¿no\?|¿vale\?|¿entiendes\?|um|uh|hmm|como)\b',
    re.IGNORECASE
)
MULTIPLE_SPACES = re.compile(r'\s{2,}')
LEADING_TRAILING_SPACES = re.compile(r'^\s+|\s+$')
MARKDOWN_BOLD = re.compile(r'\*\*(.*?)\*\*')  # Elimina formato bold de Markdown

def clean_text(text: str) -> str:
    """
    Limpia el texto eliminando markdown, espacios redundantes y palabras de relleno.
    """
    if not isinstance(text, str):
        return ""
    cleaned = MARKDOWN_BOLD.sub(r'\1', text)
    cleaned = LEADING_TRAILING_SPACES.sub('', cleaned)
    cleaned = FILLER_WORDS_ES.sub('', cleaned)
    cleaned = MULTIPLE_SPACES.sub(' ', cleaned)
    return cleaned.strip()

def anonymize_text(text: str) -> str:
    """
    Si Presidio está disponible, detecta y reemplaza información sensible (PII) en el texto.
    En caso contrario, devuelve el texto limpio original.
    """
    if not PRESIDIO_AVAILABLE or not analyzer or not anonymizer or not isinstance(text, str) or not text.strip():
        return text

    try:
        analyzer_results = analyzer.analyze(
            text=text,
            language="es",
            entities=[
                "PERSON", "LOCATION", "PHONE_NUMBER", "EMAIL_ADDRESS",
                "URL", "DATE_TIME", "CREDIT_CARD", "IBAN_CODE", "NRP", "CL_RUT"
            ],
            allow_list=None,
        )
        if not analyzer_results:
            return text

        anonymized_results = anonymizer.anonymize(
            text=text,
            analyzer_results=analyzer_results,
            operators={
                "PERSON": OperatorConfig("replace", {"new_value": "<PERSONA>"}),
                "PHONE_NUMBER": OperatorConfig("replace", {"new_value": "<TELEFONO>"}),
                "EMAIL_ADDRESS": OperatorConfig("replace", {"new_value": "<EMAIL>"}),
                "LOCATION": OperatorConfig("replace", {"new_value": "<LUGAR>"}),
                "URL": OperatorConfig("replace", {"new_value": "<URL>"}),
                "DATE_TIME": OperatorConfig("replace", {"new_value": "<FECHA_HORA>"}),
                "CREDIT_CARD": OperatorConfig("mask", {"type": "mask", "masking_char": "*", "chars_to_mask": 16, "from_end": False}),
                "IBAN_CODE": OperatorConfig("mask", {"type": "mask", "masking_char": "*", "chars_to_mask": 16, "from_end": False}),
                "NRP": OperatorConfig("replace", {"new_value": "<ID_NUMERICO>"}),
                "CL_RUT": OperatorConfig("replace", {"new_value": "<RUT>"}),
                "DEFAULT": OperatorConfig("replace", {"new_value": "<PII>"})
            }
        )
        return anonymized_results.text

    except ValueError as ve:
        if "No matching recognizers" in str(ve):
            return text
        else:
            print(f"Error ValueError en Presidio procesando texto: '{text[:50]}...'. Error: {ve}")
            return text
    except Exception as e:
        print(f"Error Inesperado en Presidio procesando texto: '{text[:50]}...'. Error: {type(e).__name__} - {e}")
        return text

def preprocess_single_transcript(transcript_item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Procesa una transcripción individual:
      - Limpia y anonimiza cada turno de diálogo (si aplica).
      - Omite turnos del sistema o sin contenido relevante.
    """
    processed_utterances = []
    transcript_id = transcript_item.get("id", "ID_DESCONOCIDO")
    original_utterances = transcript_item.get('data', [])

    if not isinstance(original_utterances, list):
         print(f"Advertencia: Datos en formato inesperado para ID {transcript_id}. Se esperaba una lista.")
         return None

    for utterance in original_utterances:
        speaker = utterance.get('speaker', 'DESCONOCIDO')
        timestamp = utterance.get('timestamp', 'N/A')
        original_text = utterance.get('text', '')

        original_preview = original_text[:70] + ("..." if len(original_text) > 70 else "")

        # Se omiten textos de turnos del sistema, notas o sin contenido
        if speaker in ["SISTEMA", "NOTA", "DESCONOCIDO"] or not original_text.strip():
             processed_text = ""
        else:
            cleaned_text = clean_text(original_text)
            processed_text = anonymize_text(cleaned_text)

        processed_utterance = {
            "speaker": speaker,
            "timestamp": timestamp,
            "original_text_preview": original_preview,
            "processed_text": processed_text
        }
        processed_utterances.append(processed_utterance)

    return {"id": transcript_id, "processed_data": processed_utterances}

def process_and_save_transcript(transcript_item: Dict[str, Any], index: int) -> bool:
    """
    Procesa y guarda una transcripción. Se utiliza en procesamiento concurrente.
    
    Retorna True si el procesamiento y guardado fueron exitosos, de lo contrario False.
    """
    transcript_id = transcript_item.get("id", f"desconocido_{index}")
    print(f"Procesando transcripción {index+1} (ID: {transcript_id})...")
    processed_transcript_data = preprocess_single_transcript(transcript_item)
    if processed_transcript_data:
        save_processed_transcript(processed_transcript_data)
        return True
    else:
        print(f"Error procesando transcripción ID: {transcript_id}. Se omite.")
        return False

def main():
    """
    Función principal para:
      1. Cargar las transcripciones crudas.
      2. Procesarlas (limpieza y anonimización) de forma concurrente.
      3. Guardar cada transcripción procesada en formato JSON.
    """
    print("Iniciando preprocesamiento desde archivos .txt...")
    raw_transcripts = load_raw_transcripts()

    if not raw_transcripts:
        print("No se encontraron transcripciones crudas para procesar. Abortando.")
        return

    if not os.path.exists(PROCESSED_DIR):
        try:
            os.makedirs(PROCESSED_DIR)
            print(f"Directorio de salida creado: {PROCESSED_DIR}")
        except OSError as e:
            print(f"Error CRÍTICO creando directorio de salida {PROCESSED_DIR}: {e}")
            return

    processed_count = 0
    error_count = 0

    # Se utiliza ThreadPoolExecutor para procesar transcripciones en paralelo
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_and_save_transcript, transcript, i)
                   for i, transcript in enumerate(raw_transcripts)]
        for future in concurrent.futures.as_completed(futures):
            try:
                if future.result():
                    processed_count += 1
                else:
                    error_count += 1
            except Exception as exc:
                error_count += 1
                print(f"Error inesperado: {exc}")

    print(f"\n--- Preprocesamiento Completo ---")
    print(f"Transcripciones procesadas exitosamente: {processed_count}")
    if error_count > 0:
        print(f"Transcripciones omitidas o con errores: {error_count}")
    print(f"Archivos procesados guardados como JSON en: {PROCESSED_DIR}")

def test_presidio():
    """
    Ejecuta casos de prueba para verificar la funcionalidad de Presidio.
    Se evalúa la anonimización de diferentes escenarios.
    """
    if not PRESIDIO_AVAILABLE:
        print("\nPresidio no está disponible. Se omite la prueba.")
        return

    print("\n--- Probando Presidio ---")
    test_cases = {
        "Sin PII": "Este es un texto normal sin información sensible.",
        "Con PII": "Hola, mi nombre es Juan Pérez y vivo en Santiago. Mi fono es +56912345678 y mi email es juan.perez@email.com. Mi RUT es 12.345.678-9",
        "Con Markdown": "El cliente **Juan Pérez** llamó con RUT 11.111.111-1.",
        "Solo RUT": "El RUT del contribuyente es 9.876.543-K.",
        "Texto Vacío": ""
    }

    for name, text in test_cases.items():
        print(f"\nCaso: {name}")
        print(f"  Texto Original   : '{text}'")
        cleaned = clean_text(text)
        print(f"  Texto Limpio     : '{cleaned}'")
        anonymized = anonymize_text(cleaned)
        print(f"  Texto Anonimizado: '{anonymized}'")

        if name == "Sin PII" and anonymized != cleaned:
             print("  [!] ADVERTENCIA: ¡El texto sin PII fue modificado!")
        if name == "Con PII":
            if anonymized == cleaned:
                print("  [!] ADVERTENCIA: ¡El texto con PII no fue anonimizado!")
            if "<PERSONA>" not in anonymized:
                print("  [!] ADVERTENCIA: ¡Falta anonimizar 'PERSONA'!")
            if "<RUT>" not in anonymized:
                print("  [!] ADVERTENCIA: ¡Falta anonimizar 'RUT'!")
            if "<TELEFONO>" not in anonymized:
                print("  [!] ADVERTENCIA: ¡Falta anonimizar 'TELEFONO'!")
            if "<EMAIL>" not in anonymized:
                print("  [!] ADVERTENCIA: ¡Falta anonimizar 'EMAIL'!")
        if name == "Con Markdown":
             if anonymized == cleaned:
                 print("  [!] ADVERTENCIA: ¡El texto con Markdown no fue anonimizado!")
             if "<PERSONA>" not in anonymized:
                 print("  [!] ADVERTENCIA: ¡Falta anonimizar 'PERSONA' en Markdown!")
             if "<RUT>" not in anonymized:
                 print("  [!] ADVERTENCIA: ¡Falta anonimizar 'RUT' en Markdown!")

    print("\n--- Fin Prueba Presidio ---")

if __name__ == "__main__":
    main()
    # Descomenta la siguiente línea para ejecutar la prueba de Presidio
    # test_presidio()

import re
import os
import sys
from typing import List, Dict, Any, Optional

# --- Añadir Directorio Raíz al Path ---
# Determinar el directorio base del proyecto (un nivel arriba de 'scripts')
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

# --- Importaciones ---
# Importar Presidio y nuestras funciones DESPUÉS de ajustar el path
try:
    from presidio_analyzer import AnalyzerEngine, PatternRecognizer, RecognizerRegistry, Pattern
    from presidio_anonymizer import AnonymizerEngine
    from presidio_anonymizer.entities import OperatorConfig
    from presidio_analyzer.nlp_engine import NlpEngineProvider
    PRESIDIO_INSTALLED = True
except ImportError:
    print("ADVERTENCIA: Librerías Presidio no encontradas. La anonimización será omitida.")
    print("Por favor, instala: pip install presidio-analyzer presidio-anonymizer")
    PRESIDIO_INSTALLED = False
    # Definir clases dummy para que el resto del código no falle si no está instalado
    class AnalyzerEngine: pass
    class AnonymizerEngine: pass
    class OperatorConfig: pass
    class PatternRecognizer: pass
    class RecognizerRegistry: pass
    class Pattern: pass

from scripts.load_data import load_raw_transcripts, save_processed_transcript, PROCESSED_DIR

# --- Configuración Presidio (Global para eficiencia) ---
analyzer = None
anonymizer = None
PRESIDIO_AVAILABLE = False

if PRESIDIO_INSTALLED:
    try:
        print("Inicializando Presidio Analyzer y Anonymizer...")

        # 1. Inicializar el NLP engine para español usando spaCy
        nlp_provider = NlpEngineProvider(nlp_configuration={
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": "es", "model_name": "es_core_news_md"}]
        })
        nlp_engine = nlp_provider.create_engine()

        # 2. Inicializar AnalyzerEngine con el NLP engine y soporte para "es"
        analyzer = AnalyzerEngine(nlp_engine=nlp_engine, supported_languages=["es"])

        # 3. Cargar los reconocedores predefinidos para español (incluye PERSON, LOCATION, etc.)
        analyzer.registry.load_predefined_recognizers(languages=["es"], nlp_engine=nlp_engine)
        print("Recognizers predeterminados para 'es' cargados.")

        # --- Reconocedor de RUT ---
        # Definir el patrón para RUT Chileno
        rut_pattern_standard = Pattern(
            name="rut_pattern",
            regex=r"\b(\d{1,2}\.\d{3}\.\d{3}-[\dkK])\b",
            score=0.8  # Puntuación alta por ser un patrón específico
        )
        # Nuevo patrón para formato con guiones
        rut_pattern_dashes = Pattern(
            name="rut_pattern_dashes",
            regex=r"\b(\d{1,2}-\d{3}-\d{3}-[\dkK]?)\b",
            score=0.75  # Puntuación ligeramente menor para diferenciar
        )
        rut_pattern_spaces = Pattern(
            name="rut_pattern_spaces",
            regex=r"\b(\d{1,2}\s\d{3}\s\d{3}-[\dkK])\b",  # Ejemplo: 12 345 678-9
            score=0.75  # Puntuación ligeramente menor para diferenciar
        )   
        # --- Reconocedor de Números Telefónicos (formato chileno) ---
        # Agregar un reconocedor personalizado para formatos telefónicos chilenos
        phone_pattern = Pattern(
            name="chile_phone_pattern",
            # Coincide con +56XXXXXXXXX, +56 9 XXXX XXXX y otros formatos comunes chilenos
            regex=r"\b(\+?56\s?[2-9][\s\d]{8,11})\b",
            score=0.7  # Puntuación adecuada para asegurar detección
        )

        # Crear el reconocedor de teléfono
        phone_recognizer = PatternRecognizer(
            supported_entity="PHONE_NUMBER",  # Tipo de entidad estándar para compatibilidad
            name="ChileanPhoneRecognizer",
            patterns=[phone_pattern],
            context=["teléfono", "fono", "celular", "móvil", "llamar", "contacto"],
            supported_language="es"
        )

        # --- Reconocedor de Email (mejorado) ---
        # Agregar un reconocedor de email personalizado con mayor puntuación para asegurar la detección
        email_pattern = Pattern(
            name="email_pattern",
            regex=r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
            score=0.9  # Puntuación más alta para priorizar la detección
        )

        # Crear el reconocedor de email
        email_recognizer = PatternRecognizer(
            supported_entity="EMAIL_ADDRESS",  # Tipo de entidad estándar para compatibilidad
            name="EmailRecognizer",
            patterns=[email_pattern],
            context=["email", "correo", "e-mail", "@"],
            supported_language="es"
        )


        # Crear el reconocedor usando los patrones, asignándole explícitamente el idioma "es"
        rut_recognizer = PatternRecognizer(
        supported_entity="CL_RUT",
        name="ChileanRUTRecognizer",
        patterns=[rut_pattern_standard, rut_pattern_dashes, rut_pattern_spaces],  # Lista de patrones
        context=["RUT", "rut", "Rol Único Tributario"],
        supported_language="es"
    )
        # Añadir todos los reconocedores personalizados al registro
        analyzer.registry.add_recognizer(rut_recognizer)
        analyzer.registry.add_recognizer(phone_recognizer)
        analyzer.registry.add_recognizer(email_recognizer)

        # 4. Inicializar AnonymizerEngine
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


# --- Regex para Limpieza ---
FILLER_WORDS_ES = re.compile(
    r'\b(eh|este|pues|bueno|o sea|¿sabes\?|¿no\?|¿vale\?|¿entiendes\?|um|uh|hmm|como)\b',
    re.IGNORECASE
)
MULTIPLE_SPACES = re.compile(r'\s{2,}')
LEADING_TRAILING_SPACES = re.compile(r'^\s+|\s+$')
MARKDOWN_BOLD = re.compile(r'\*\*(.*?)\*\*')  # Regex para quitar ** de markdown

# --- Funciones de Procesamiento ---

def clean_text(text: str) -> str:
    """Realiza limpieza básica del texto, incluyendo markdown simple."""
    if not isinstance(text, str):
        return ""
    # Quitar markdown bold (**text**) -> text
    cleaned = MARKDOWN_BOLD.sub(r'\1', text)
    # Limpieza general
    cleaned = LEADING_TRAILING_SPACES.sub('', cleaned)
    cleaned = FILLER_WORDS_ES.sub('', cleaned)
    cleaned = MULTIPLE_SPACES.sub(' ', cleaned)
    return cleaned.strip()

def anonymize_text(text: str) -> str:
    """Detecta y anonimiza PII usando Presidio, si está disponible."""
    if not PRESIDIO_AVAILABLE or not analyzer or not anonymizer or not isinstance(text, str) or not text.strip():
        # Si Presidio no está disponible o texto inválido, devolver texto original (ya limpio)
        return text

    try:
        # Analizar buscando entidades comunes + CL_RUT
        analyzer_results = analyzer.analyze(
            text=text,
            language="es",
            # Lista de entidades a buscar (incluyendo la personalizada)
            entities=[
                "PERSON", "LOCATION", "PHONE_NUMBER", "EMAIL_ADDRESS",
                "URL", "DATE_TIME", "CREDIT_CARD", "IBAN_CODE", "NRP", "CL_RUT"
            ],
            allow_list=None,
        )

        # Si no se encontraron PII, devolver el texto tal cual
        if not analyzer_results:
            return text

        # Configurar cómo reemplazar cada tipo de PII encontrado
        anonymized_results = anonymizer.anonymize(
        text=text,
        analyzer_results=analyzer_results,
        operators={
            "PERSON": OperatorConfig("replace", {"new_value": "<PERSONA>"}),
            "PHONE_NUMBER": OperatorConfig("replace", {"new_value": "<TELEFONO>"}),  # Cambiado a reemplazo en lugar de enmascaramiento
            "EMAIL_ADDRESS": OperatorConfig("replace", {"new_value": "<EMAIL>"}),    # Cambiado a reemplazo en lugar de enmascaramiento
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
        # Manejar específicamente el error "No matching recognizers" como advertencia silenciosa
        if "No matching recognizers" in str(ve):
            # Devolver texto original si no encontró nada
            return text
        else:
            print(f"Error ValueError en Presidio procesando texto: '{text[:50]}...'. Error: {ve}")
            return text
    except Exception as e:
        print(f"Error Inesperado en Presidio procesando texto: '{text[:50]}...'. Error: {type(e).__name__} - {e}")
        return text

def preprocess_single_transcript(transcript_item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Limpia y anonimiza el texto dentro de cada turno de la transcripción parseada."""
    processed_utterances = []
    transcript_id = transcript_item.get("id", "ID_DESCONOCIDO")
    original_utterances = transcript_item.get('data', [])

    # Validar que los datos parseados sean una lista
    if not isinstance(original_utterances, list):
         print(f"Advertencia: Formato de datos parseados inválido para ID {transcript_id}. Se esperaba lista. Saltando transcripción.")
         return None  # Indica que esta transcripción no se pudo procesar

    # Iterar sobre cada línea/turno parseado
    for utterance in original_utterances:
        speaker = utterance.get('speaker', 'DESCONOCIDO')
        timestamp = utterance.get('timestamp', 'N/A')
        original_text = utterance.get('text', '')

        # Preparar el preview del texto original (independiente del procesamiento)
        original_preview = original_text[:70] + ("..." if len(original_text) > 70 else "")

        # Decidir si el texto de este turno necesita limpieza y anonimización
        # Ignorar hablantes de sistema/nota o texto vacío
        if speaker in ["SISTEMA", "NOTA", "DESCONOCIDO"] or not original_text.strip():
             processed_text = ""  # Dejar vacío el texto procesado para estos casos
        else:
            # Aplicar limpieza primero
            cleaned_text = clean_text(original_text)
            # Luego anonimizar el texto ya limpio
            processed_text = anonymize_text(cleaned_text)

        # Construir el diccionario para este turno procesado
        processed_utterance = {
            "speaker": speaker,
            "timestamp": timestamp,
            "original_text_preview": original_preview,  # Guardar preview del *original*
            "processed_text": processed_text            # Texto final limpio y anonimizado
        }
        processed_utterances.append(processed_utterance)

    # Devolver la estructura completa de la transcripción procesada
    return {"id": transcript_id, "processed_data": processed_utterances}

# --- Función Principal ---
def main():
    """Función principal para ejecutar el preprocesamiento completo."""
    print("Iniciando preprocesamiento desde archivos .txt...")
    # Cargar y parsear los archivos .txt crudos
    raw_transcripts = load_raw_transcripts()

    if not raw_transcripts:
        print("No se encontraron transcripciones crudas para procesar. Abortando.")
        return

    # Asegurar que el directorio de salida exista
    if not os.path.exists(PROCESSED_DIR):
        try:
            os.makedirs(PROCESSED_DIR)
            print(f"Directorio de salida creado: {PROCESSED_DIR}")
        except OSError as e:
             print(f"Error CRÍTICO creando directorio de salida {PROCESSED_DIR}: {e}")
             return  # No podemos continuar si no podemos guardar

    # Contadores para el resumen final
    processed_count = 0
    error_count = 0

    # Procesar cada transcripción cargada
    for i, transcript_item in enumerate(raw_transcripts):
        transcript_id = transcript_item.get('id', f'desconocido_{i}')
        print(f"Procesando transcripción {i+1}/{len(raw_transcripts)} (ID: {transcript_id})...")

        # Llamar a la función de procesamiento individual
        processed_transcript_data = preprocess_single_transcript(transcript_item)

        # Guardar el resultado si el procesamiento fue exitoso
        if processed_transcript_data:
            save_processed_transcript(processed_transcript_data)  # Guarda como JSON
            processed_count += 1
        else:
            # Si preprocess_single_transcript devuelve None, contar como error
            error_count += 1
            print(f"  Error procesando transcripción ID: {transcript_id}. Omitida.")

    # Imprimir resumen del proceso
    print(f"\n--- Preprocesamiento Completo ---")
    print(f"Transcripciones procesadas exitosamente: {processed_count}")
    if error_count > 0:
        print(f"Transcripciones con errores o formato inválido (omitidas): {error_count}")
    print(f"Archivos procesados guardados como JSON en: {PROCESSED_DIR}")

# --- Bloque de Prueba Opcional para Presidio ---
def test_presidio():
    """Realiza una prueba básica de Presidio para verificar la anonimización."""
    if not PRESIDIO_AVAILABLE:
        print("\nPresidio no está disponible. Omitiendo prueba.")
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
        print(f"  Texto Original : '{text}'")
        cleaned = clean_text(text)
        print(f"  Texto Limpio   : '{cleaned}'")
        anonymized = anonymize_text(cleaned)
        print(f"  Texto Anonimizado: '{anonymized}'")

        # Verificaciones básicas
        if name == "Sin PII" and anonymized != cleaned:
             print("  [!] ADVERTENCIA: ¡Texto sin PII fue modificado!")
        if name == "Con PII":
            if anonymized == cleaned: print("  [!] ADVERTENCIA: ¡Texto con PII NO fue modificado!")
            if "<PERSONA>" not in anonymized: print("  [!] ADVERTENCIA: ¡PERSONA no anonimizada!")
            if "<RUT>" not in anonymized: print("  [!] ADVERTENCIA: ¡RUT no anonimizado!")
            if "<TELEFONO>" not in anonymized: print("  [!] ADVERTENCIA: ¡TELEFONO no anonimizado!")  # Debería ser enmascarado
            if "<EMAIL>" not in anonymized: print("  [!] ADVERTENCIA: ¡EMAIL no anonimizado!")  # Debería ser enmascarado
        if name == "Con Markdown":
             if anonymized == cleaned: print("  [!] ADVERTENCIA: ¡Texto (Markdown) con PII NO fue modificado!")
             if "<PERSONA>" not in anonymized: print("  [!] ADVERTENCIA: ¡PERSONA (Markdown) no anonimizada!")
             if "<RUT>" not in anonymized: print("  [!] ADVERTENCIA: ¡RUT (Markdown) no anonimizado!")

    print("\n--- Fin Prueba Presidio ---")

# --- Punto de Entrada Principal ---
if __name__ == "__main__":
    main()
    # Descomenta la siguiente línea si quieres ejecutar las pruebas de Presidio al final
    #test_presidio()

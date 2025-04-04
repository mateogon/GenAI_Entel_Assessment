from presidio_analyzer import AnalyzerEngine, PatternRecognizer, Pattern
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig
import re

# Configurar el NLP engine para español usando spaCy
nlp_provider = NlpEngineProvider(nlp_configuration={
    "nlp_engine_name": "spacy",
    "models": [{"lang_code": "es", "model_name": "es_core_news_md"}]
})
nlp_engine = nlp_provider.create_engine()

# Inicializar el AnalyzerEngine con el NLP engine y soporte para "es"
analyzer = AnalyzerEngine(nlp_engine=nlp_engine, supported_languages=["es"])

# Cargar los reconocedores predefinidos para español (incluye PERSON, LOCATION, etc.)
analyzer.registry.load_predefined_recognizers(languages=["es"], nlp_engine=nlp_engine)
print("Recognizers predeterminados para 'es' cargados.")

# Definir el patrón para RUT Chileno (ejemplo: 12.345.678-9)
rut_pattern = Pattern(
    name="rut_pattern",
    regex=r"\b(\d{1,2}\.\d{3}\.\d{3}-[\dkK])\b",
    score=0.8
)

# Crear el reconocedor personalizado para RUT asignando explícitamente "es" como idioma
rut_recognizer = PatternRecognizer(
    supported_entity="CL_RUT",
    name="ChileanRUTRecognizer",
    patterns=[rut_pattern],
    context=["RUT", "rut", "Rol Único Tributario"],
    supported_language="es"
)
analyzer.registry.add_recognizer(rut_recognizer)
print("Reconocedor de RUT personalizado añadido.")

# Texto de prueba: contiene nombres (PERSON) y RUTs (CL_RUT)
test_text = """
Buenos días, le atiende Carlos.
Mi nombre es Juan Pérez y mi RUT es 12.345.678-9.
Además, mi colega María Gómez tiene el RUT 9.876.543-K,
y en el departamento se registra a Pedro Martínez con RUT 11.222.333-4.
"""

# Prueba independiente: buscar coincidencias del regex para RUT
pattern = re.compile(r"\b(\d{1,2}\.\d{3}\.\d{3}-[\dkK])\b")
print("\nResultados del regex independiente:")
print(pattern.findall(test_text))

# Analizar el texto buscando las entidades PERSON y CL_RUT
results_all = analyzer.analyze(
    text=test_text,
    language="es",
    entities=["PERSON", "CL_RUT"]
)
print("\nResultados del análisis (PERSON y CL_RUT):")
for r in results_all:
    print(r)

# Inicializar el AnonymizerEngine para reemplazar las entidades detectadas
anonymizer = AnonymizerEngine()
operators = {
    "PERSON": OperatorConfig("replace", {"new_value": "<PERSONA>"}),
    "CL_RUT": OperatorConfig("replace", {"new_value": "<RUT>"})
}


anonymized_result = anonymizer.anonymize(
    text=test_text,
    analyzer_results=results_all,
    operators=operators
)
print("\nTexto anonimizado:")
print(anonymized_result.text)

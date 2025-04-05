"""
Propósito: Cargar y parsear transcripciones en formato .txt, y gestionar la persistencia
de transcripciones procesadas en formato JSON.
"""

import os
import re
import orjson
from typing import List, Dict, Any, Optional

# Definición de rutas base y archivos del proyecto
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Directorio raíz del proyecto
DATA_DIR = os.path.join(BASE_DIR, 'data', 'raw')
PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')
EMBEDDINGS_FILE = os.path.join(BASE_DIR, 'data', 'embeddings.npy')
ID_MAP_FILE = os.path.join(BASE_DIR, 'data', 'id_map.pkl')

# Expresiones regulares para interpretar distintos formatos de línea en los archivos .txt
LINE_REGEX_SPEAKER = re.compile(r"\[(\d{2}:\d{2}:\d{2})\]\s*(\w+):\s*\.?\s*(.*)")
LINE_REGEX_END_CALL = re.compile(r"(?:\[(\d{2}:\d{2}:\d{2})\]\s*)?.*(?:LLAMADA FINALIZADA|\*\*\*|\[FIN DE LA LLAMADA\]|\[Llamada finalizada\]).*")
LINE_REGEX_LIST_ITEM = re.compile(r"^\s*-\s+(.*)")

def parse_transcript_line(line: str) -> Optional[Dict[str, str]]:
    """
    Interpreta una línea del archivo de transcripción.
    
    Se intenta identificar tres patrones:
      - Línea con timestamp, hablante y texto.
      - Línea que indica el fin de llamada.
      - Línea de item de lista.
    
    Si no se reconoce ningún patrón, se marca la línea como 'DESCONOCIDO'.
    """
    stripped_line = line.strip()
    if not stripped_line:
        return None  # Omitir líneas vacías

    match_speaker = LINE_REGEX_SPEAKER.match(stripped_line)
    if match_speaker:
        timestamp, speaker, text = match_speaker.groups()
        return {"timestamp": timestamp, "speaker": speaker.upper(), "text": text.strip()}

    match_end_call = LINE_REGEX_END_CALL.match(stripped_line)
    if match_end_call:
        # Usa el timestamp si está presente, o 'N/A'
        timestamp = match_end_call.group(1) or "N/A"
        return {"timestamp": timestamp, "speaker": "SISTEMA", "text": stripped_line}

    match_list_item = LINE_REGEX_LIST_ITEM.match(stripped_line)
    if match_list_item:
        text = match_list_item.group(1)
        return {"timestamp": "N/A", "speaker": "NOTA", "text": text.strip()}

    # Caso por defecto: línea sin patrón definido
    print(f"Advertencia: Línea no reconocida -> {stripped_line}")
    return {"timestamp": "N/A", "speaker": "DESCONOCIDO", "text": stripped_line}

def load_raw_transcripts(data_dir: str = DATA_DIR) -> List[Dict[str, Any]]:
    """
    Recorre el directorio de archivos .txt y parsea cada transcripción.
    
    Cada archivo se procesa línea por línea y se almacena solo si se obtienen datos útiles.
    """
    transcripts = []
    if not os.path.exists(data_dir):
        print(f"Error: El directorio de datos crudos no existe: {data_dir}")
        return []
    print(f"Cargando y parseando transcripciones .txt desde: {data_dir}")

    try:
        filenames = [f for f in os.listdir(data_dir) if f.endswith(".txt")]
    except FileNotFoundError:
        print(f"Error: Directorio no encontrado al listar archivos: {data_dir}")
        return []
    except Exception as e:
        print(f"Error listando archivos en {data_dir}: {e}")
        return []

    for filename in filenames:
        file_path = os.path.join(data_dir, filename)
        # Se extrae el ID del archivo (ej: "sample_01.txt" → "sample_01")
        transcript_id = os.path.splitext(filename)[0]
        parsed_lines = []
        line_number = 0
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line_number += 1
                    parsed_line = parse_transcript_line(line)
                    if parsed_line:
                        parsed_lines.append(parsed_line)
            if parsed_lines:
                transcripts.append({"id": transcript_id, "data": parsed_lines})
            else:
                print(f"Advertencia: No se parsearon líneas útiles de {filename}")
        except Exception as e:
            print(f"Error crítico procesando {filename} en línea {line_number}: {e}")

    print(f"Cargadas y parseadas {len(transcripts)} transcripciones.")
    return transcripts

def save_processed_transcript(transcript_data: Dict[str, Any], output_dir: str = PROCESSED_DIR):
    """
    Guarda una transcripción procesada en formato JSON usando orjson para mayor eficiencia.
    
    Se verifica la existencia del directorio de salida y se crea si es necesario.
    """
    if not transcript_data or 'id' not in transcript_data:
         print("Error: Datos inválidos para guardar transcripción procesada.")
         return

    if not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
        except OSError as e:
            print(f"Error creando directorio de salida {output_dir}: {e}")
            return

    output_filename = f"processed_{transcript_data['id']}.json"
    output_path = os.path.join(output_dir, output_filename)
    try:
        with open(output_path, 'wb') as f:
            f.write(orjson.dumps(transcript_data, option=orjson.OPT_INDENT_2 | orjson.OPT_APPEND_NEWLINE))
    except Exception as e:
        print(f"Error guardando {output_filename}: {e}")

def load_processed_transcripts(data_dir: str = PROCESSED_DIR) -> List[Dict[str, Any]]:
    """
    Carga todas las transcripciones procesadas (archivos JSON) desde el directorio especificado.
    """
    transcripts = []
    if not os.path.exists(data_dir):
        print(f"Error: El directorio de datos procesados no existe: {data_dir}")
        return []
    print(f"Cargando transcripciones procesadas (JSON) desde: {data_dir}")

    try:
        filenames = [f for f in os.listdir(data_dir) if f.startswith("processed_") and f.endswith(".json")]
    except FileNotFoundError:
        print(f"Error: Directorio no encontrado al listar archivos procesados: {data_dir}")
        return []
    except Exception as e:
        print(f"Error listando archivos procesados en {data_dir}: {e}")
        return []

    for filename in filenames:
        file_path = os.path.join(data_dir, filename)
        try:
            with open(file_path, 'rb') as f:
                data = orjson.loads(f.read())
                transcripts.append(data)
        except orjson.JSONDecodeError as e:
            print(f"Error decodificando JSON en {filename}: {e}")
        except Exception as e:
            print(f"Error cargando archivo procesado {filename}: {e}")

    print(f"Cargadas {len(transcripts)} transcripciones procesadas.")
    return transcripts


# Bloque de prueba para validar funciones de carga y guardado
if __name__ == "__main__":
    print("--- Ejecutando pruebas de load_data.py ---")
    raw_data = load_raw_transcripts()
    if raw_data:
        print(f"\nEjemplo de datos crudos parseados (ID: {raw_data[0]['id']}):")
        sample_data = raw_data[0]['data']
        print("Primeras 5 líneas parseadas:")
        print(orjson.dumps(sample_data[:5], option=orjson.OPT_INDENT_2).decode())
        if len(sample_data) > 5:
            print("\nÚltimas 5 líneas parseadas:")
            print(orjson.dumps(sample_data[-5:], option=orjson.OPT_INDENT_2).decode())

    # Prueba de guardado y posterior carga de transcripción procesada
    if raw_data:
        test_id = "test_load_save_001"
        test_processed_internal = [
            {"speaker": "AGENTE", "timestamp": "00:00:01", "original_text_preview": "...", "processed_text": "Hola"},
            {"speaker": "CLIENTE", "timestamp": "00:00:05", "original_text_preview": "...", "processed_text": "Adiós"}
        ]
        test_processed_to_save = {"id": test_id, "processed_data": test_processed_internal}
        print(f"\nProbando guardado para ID: {test_id}")
        save_processed_transcript(test_processed_to_save)
        print("Probando carga...")
        processed_data = load_processed_transcripts()
        loaded_test = any(t['id'] == test_id for t in processed_data)
        print(f"Transcripción de prueba cargada: {'Sí' if loaded_test else 'No'}")
        # Eliminación del archivo de prueba
        test_file_path = os.path.join(PROCESSED_DIR, f"processed_{test_id}.json")
        if os.path.exists(test_file_path):
            try:
                os.remove(test_file_path)
                print("Archivo de prueba eliminado.")
            except OSError as e:
                print(f"Error eliminando archivo de prueba: {e}")
    print("\n--- Fin pruebas load_data.py ---")

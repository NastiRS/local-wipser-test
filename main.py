from faster_whisper import WhisperModel
import time


def load_model(turbo: bool, low_power: bool) -> WhisperModel:
    print("Cargando modelo...")
    start_load = time.time()
    model_name = "large-v3-turbo" if turbo else "large-v3"
    compute_type = "int8" if low_power else "float16"

    model = WhisperModel(model_name, device="auto", compute_type=compute_type)
    end_load = time.time()
    print(
        f"Modelo cargado en {end_load - start_load:.2f} segundos."
    )  # Added print with timing
    return model


def transcribe(model: WhisperModel, audio_path: str, language: str = None) -> str:
    print(f"Iniciando transcripción de '{audio_path}'...")
    segments, info = model.transcribe(
        audio_path,
        beam_size=5,
        language=language,
        temperature=0.4,
        # vad_filter=True,
    )
    print(
        f"Detectado idioma: {info.language} con probabilidad {info.language_probability}"
    )
    print("Uniendo segmentos...")
    result = " ".join(segment.text for segment in segments)
    print("Segmentos unidos.")

    return result


if __name__ == "__main__":
    audio_file = ""
    turbo = True
    low_power = False
    language = "es"

    print("--- Inicio del script ---")  #
    start_transcribe = time.time()

    # Cargar el modelo una sola vez
    model = load_model(turbo=turbo, low_power=low_power)

    try:
        texto = transcribe(model=model, audio_path=audio_file, language=language)
        end_transcribe = time.time()

        print("--- Transcripción obtenida ---")
        print("Transcripción:", texto if texto else "[No se generó texto]")
        print(
            f"OK:Transcripción completada en {end_transcribe - start_transcribe:.2f} segundos."
        )

    except Exception as e:
        print(f"Ocurrió un error: {e}")
    print("--- Fin del script ---")

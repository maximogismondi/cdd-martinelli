import tensorflow as tf
from tensorflow import keras
import numpy as np
from pathlib import Path
import argparse
import json
import cv2


def load_sequence(frame_paths, sequence_length):
    """Carga una secuencia de frames"""
    sequence = []
    for path in frame_paths[-sequence_length:]:  # Últimos N frames
        img = tf.keras.preprocessing.image.load_img(path)
        img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
        sequence.append(img_array)

    # Si no hay suficientes frames, repetir el primero
    while len(sequence) < sequence_length:
        sequence.insert(0, sequence[0])

    return np.array([sequence])  # Batch de 1


def predict_clean_frame(model, dirty_frames_dir, frame_idx, sequence_length):
    """
    Predice el frame limpio usando contexto temporal.

    Args:
        model: Modelo entrenado
        dirty_frames_dir: Directorio con frames sucios
        frame_idx: Índice del frame a predecir
        sequence_length: Longitud de la secuencia temporal

    Returns:
        Frame limpio predicho (como array numpy 0-255)
    """
    # Cargar frames alrededor del frame objetivo
    all_frames = sorted(list(dirty_frames_dir.glob("*.jpg")))

    # Calcular rango de frames para la secuencia
    start_idx = max(0, frame_idx - sequence_length // 2)
    end_idx = min(len(all_frames), start_idx + sequence_length)
    start_idx = max(0, end_idx - sequence_length)

    frame_paths = all_frames[start_idx:end_idx]

    # Cargar secuencia
    X = load_sequence(frame_paths, sequence_length)

    # Predecir
    prediction = model.predict(X, verbose=0)[0]

    # Convertir a rango 0-255
    prediction = (prediction * 255).astype(np.uint8)

    return prediction


def process_video(
    model,
    dirty_video_dir,
    output_dir,
    sequence_length,
    save_comparison=False,
    clean_video_dir=None,
):
    """
    Procesa todos los frames de un video.

    Args:
        model: Modelo entrenado
        dirty_video_dir: Directorio con frames sucios del video
        output_dir: Directorio de salida
        sequence_length: Longitud de la secuencia temporal
        save_comparison: Si guardar comparación lado a lado
        clean_video_dir: Directorio con frames limpios originales (para comparación)
    """
    video_name = dirty_video_dir.name
    output_video_dir = output_dir / video_name
    output_video_dir.mkdir(parents=True, exist_ok=True)

    if save_comparison:
        comparison_dir = output_dir / f"{video_name}_comparison"
        comparison_dir.mkdir(parents=True, exist_ok=True)

    frames = sorted(list(dirty_video_dir.glob("*.jpg")))
    total_frames = len(frames)

    print(f"\nProcesando video: {video_name} ({total_frames} frames)")

    for i, frame_path in enumerate(frames):
        # Predecir frame limpio
        clean_pred = predict_clean_frame(model, dirty_video_dir, i, sequence_length)

        # Guardar predicción
        output_path = output_video_dir / frame_path.name
        cv2.imwrite(str(output_path), cv2.cvtColor(clean_pred, cv2.COLOR_RGB2BGR))

        # Guardar comparación si se solicita
        if save_comparison and clean_video_dir is not None:
            dirty_img = cv2.imread(str(frame_path))
            clean_original = cv2.imread(str(clean_video_dir / frame_path.name))
            clean_pred_bgr = cv2.cvtColor(clean_pred, cv2.COLOR_RGB2BGR)

            # Crear imagen de comparación (dirty | pred | clean)
            comparison = np.hstack([dirty_img, clean_pred_bgr, clean_original])
            comparison_path = comparison_dir / frame_path.name
            cv2.imwrite(str(comparison_path), comparison)

        # Progreso
        if (i + 1) % 50 == 0:
            print(f"  Procesados {i+1}/{total_frames} frames")

    print(f"  Completado: {total_frames} frames guardados en {output_video_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Inferencia: limpia frames usando el modelo entrenado"
    )
    parser.add_argument(
        "--model-path", type=str, required=True, help="Path al modelo .keras"
    )
    parser.add_argument(
        "--config-path",
        type=str,
        required=True,
        help="Path al archivo config.json del modelo",
    )
    parser.add_argument(
        "--dirty-frames-dir",
        type=str,
        default=".data/dirty_frames",
        help="Directorio con frames sucios",
    )
    parser.add_argument(
        "--clean-frames-dir",
        type=str,
        default=".data/clean_frames",
        help="Directorio con frames limpios (para comparación)",
    )
    parser.add_argument(
        "--output-dir", type=str, default="predictions", help="Directorio de salida"
    )
    parser.add_argument(
        "--save-comparison",
        action="store_true",
        help="Guardar comparaciones lado a lado",
    )
    parser.add_argument(
        "--video-name",
        type=str,
        default=None,
        help="Nombre de video específico a procesar (si no se especifica, procesa todos)",
    )

    args = parser.parse_args()

    # Cargar configuración
    with open(args.config_path, "r") as f:
        config = json.load(f)

    print("=== Configuración del modelo ===")
    print(json.dumps(config, indent=2))

    # Cargar modelo
    print(f"\n=== Cargando modelo desde {args.model_path} ===")
    model = keras.models.load_model(args.model_path)

    # Paths
    dirty_dir = Path(args.dirty_frames_dir)
    clean_dir = Path(args.clean_frames_dir) if args.save_comparison else None
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # Obtener videos a procesar
    if args.video_name:
        video_dirs = [dirty_dir / args.video_name]
        if not video_dirs[0].exists():
            print(f"Error: Video {args.video_name} no encontrado")
            return
    else:
        video_dirs = sorted([d for d in dirty_dir.iterdir() if d.is_dir()])

    print(f"\n=== Videos a procesar: {len(video_dirs)} ===")

    # Procesar cada video
    for video_dir in video_dirs:
        clean_video_dir = clean_dir / video_dir.name if clean_dir else None
        process_video(
            model,
            video_dir,
            output_dir,
            config["sequence_length"],
            args.save_comparison,
            clean_video_dir,
        )

    print(f"\n=== Procesamiento completado ===")
    print(f"Resultados guardados en: {output_dir}")


if __name__ == "__main__":
    main()

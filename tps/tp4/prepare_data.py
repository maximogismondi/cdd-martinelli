import cv2
import numpy as np
from pathlib import Path
import argparse


def generate_random_mask(
    height, width, seed, num_blobs=3, min_radius=30, max_radius=60, blur_size=21
):
    """
    Genera una máscara aleatoria con manchas circulares usando una seed.

    Args:
        height: Altura de la máscara
        width: Ancho de la máscara
        seed: Seed para el generador de números aleatorios
        num_blobs: Número de manchas a generar (default: 3 para ser más sutil)
        min_radius: Radio mínimo de las manchas (default: 30)
        max_radius: Radio máximo de las manchas (default: 60)
        blur_size: Tamaño del kernel de blur para suavizar bordes (default: 21)

    Returns:
        Máscara binaria (0s y 255s)
    """
    # Establecer la seed para reproducibilidad
    np.random.seed(seed)

    mask = np.zeros((height, width), dtype=np.uint8)

    for _ in range(num_blobs):
        # Posición aleatoria
        center_x = np.random.randint(0, width)
        center_y = np.random.randint(0, height)

        # Radio aleatorio
        radius = np.random.randint(min_radius, max_radius)

        # Dibujar círculo en la máscara
        cv2.circle(mask, (center_x, center_y), radius, 255, -1)

    # Aplicar blur para suavizar los bordes y hacer la mancha más sutil
    mask = cv2.GaussianBlur(mask, (blur_size, blur_size), 0)

    return mask


def apply_mask_to_frame(frame, mask):
    """
    Aplica la máscara al frame oscureciendo las áreas marcadas.

    Args:
        frame: Frame original (BGR)
        mask: Máscara binaria

    Returns:
        Frame con la máscara aplicada
    """
    # Normalizar la máscara a rango [0, 1]
    mask_normalized = mask.astype(np.float32) / 255.0

    # Invertir la máscara (queremos oscurecer donde hay mancha)
    mask_inverted = 1.0 - mask_normalized

    # Crear el efecto de mancha más sutil (oscurecer menos agresivamente)
    # Factor de oscurecimiento: 0.6 en vez de 0.3 (menos agresivo)
    dirty_frame = frame.copy().astype(np.float32)
    for c in range(3):  # Aplicar a cada canal BGR
        dirty_frame[:, :, c] = dirty_frame[:, :, c] * (0.6 + 0.4 * mask_inverted)

    return dirty_frame.astype(np.uint8)


def process_video(
    video_path,
    output_clean_dir,
    output_dirty_dir,
    frame_size=128,
    num_blobs=3,
    min_radius=30,
    max_radius=60,
    blur_size=21,
):
    """
    Procesa un video: extrae frames, aplica máscara y guarda versiones limpias y sucias.

    Args:
        video_path: Path al archivo de video
        output_clean_dir: Directorio para frames limpios
        output_dirty_dir: Directorio para frames sucios
        frame_size: Tamaño N x N para escalar los frames
        num_blobs: Número de manchas por video
        min_radius: Radio mínimo de las manchas
        max_radius: Radio máximo de las manchas
        blur_size: Tamaño del blur para suavizar
    """
    video_name = video_path.stem
    print(f"Procesando video: {video_name}")

    # Crear directorios de salida para este video
    clean_dir = output_clean_dir / video_name
    dirty_dir = output_dirty_dir / video_name
    clean_dir.mkdir(parents=True, exist_ok=True)
    dirty_dir.mkdir(parents=True, exist_ok=True)

    # Abrir el video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: No se pudo abrir el video {video_path}")
        return

    # Obtener dimensiones del video
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"  Dimensiones: {width}x{height}, Total frames: {total_frames}")

    # Generar seed a partir del nombre del video
    seed = hash(video_name) % (2**32)
    print(f"  Seed: {seed}, Manchas: {num_blobs}, Radio: [{min_radius}-{max_radius}]")

    # Generar una máscara aleatoria única para este video
    mask = generate_random_mask(
        height, width, seed, num_blobs, min_radius, max_radius, blur_size
    )

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Aplicar la máscara al frame
        dirty_frame = apply_mask_to_frame(frame, mask)

        # Escalar ambos frames a N x N
        clean_resized = cv2.resize(frame, (frame_size, frame_size))
        dirty_resized = cv2.resize(dirty_frame, (frame_size, frame_size))

        # Guardar los frames
        frame_filename = f"frame_{frame_count:06d}.jpg"
        cv2.imwrite(str(clean_dir / frame_filename), clean_resized)
        cv2.imwrite(str(dirty_dir / frame_filename), dirty_resized)

        frame_count += 1

        # Mostrar progreso cada 100 frames
        if frame_count % 100 == 0:
            print(f"  Procesados {frame_count}/{total_frames} frames")

    cap.release()
    print(f"  Completado: {frame_count} frames guardados\n")


def main():
    parser = argparse.ArgumentParser(
        description="Procesa videos para generar dataset de frames limpios y sucios"
    )
    parser.add_argument(
        "--raw-videos-dir",
        type=str,
        default=".data/raw_videos",
        help="Directorio con los videos originales",
    )
    parser.add_argument(
        "--clean-frames-dir",
        type=str,
        default=".data/clean_frames",
        help="Directorio de salida para frames limpios",
    )
    parser.add_argument(
        "--dirty-frames-dir",
        type=str,
        default=".data/dirty_frames",
        help="Directorio de salida para frames sucios",
    )
    parser.add_argument(
        "--frame-size",
        type=int,
        default=128,
        help="Tamaño N x N para escalar los frames (default: 128)",
    )
    parser.add_argument(
        "--num-blobs",
        type=int,
        default=3,
        help="Número de manchas por video (default: 3)",
    )
    parser.add_argument(
        "--min-radius",
        type=int,
        default=30,
        help="Radio mínimo de las manchas en píxeles (default: 30)",
    )
    parser.add_argument(
        "--max-radius",
        type=int,
        default=60,
        help="Radio máximo de las manchas en píxeles (default: 60)",
    )
    parser.add_argument(
        "--blur-size",
        type=int,
        default=21,
        help="Tamaño del blur para suavizar manchas (default: 21, debe ser impar)",
    )

    args = parser.parse_args()

    # Convertir a Path objects
    raw_videos_dir = Path(args.raw_videos_dir)
    clean_frames_dir = Path(args.clean_frames_dir)
    dirty_frames_dir = Path(args.dirty_frames_dir)

    # Verificar que existe el directorio de videos
    if not raw_videos_dir.exists():
        print(f"Error: El directorio {raw_videos_dir} no existe")
        return

    # Crear directorios de salida
    clean_frames_dir.mkdir(parents=True, exist_ok=True)
    dirty_frames_dir.mkdir(parents=True, exist_ok=True)

    # Buscar todos los videos en el directorio
    video_extensions = [".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv"]
    video_files = []
    for ext in video_extensions:
        video_files.extend(raw_videos_dir.glob(f"*{ext}"))
        video_files.extend(raw_videos_dir.glob(f"*{ext.upper()}"))

    if not video_files:
        print(f"No se encontraron videos en {raw_videos_dir}")
        return

    print(f"Encontrados {len(video_files)} videos para procesar\n")
    print(f"Tamaño de frames: {args.frame_size}x{args.frame_size}\n")

    # Procesar cada video
    for video_path in sorted(video_files):
        process_video(
            video_path,
            clean_frames_dir,
            dirty_frames_dir,
            args.frame_size,
            args.num_blobs,
            args.min_radius,
            args.max_radius,
            args.blur_size,
        )

    print("¡Procesamiento completado!")


if __name__ == "__main__":
    main()

"""
Script para procesar múltiples videos y generar comparaciones.
"""
import cv2
import numpy as np
import glob
import os
import tensorflow as tf
from inference_mobile_net import process_video


def create_comparison_video(video_name, dirty_dir=".data/dirty_frames", 
                           clean_dir=".data/clean_frames",
                           corrected_dir="corrected_frames",
                           output_name=None, fps=30):
    """
    Crea un video de comparación lado a lado mostrando:
    dirty | clean | corrected
    """
    if output_name is None:
        output_name = f"{video_name}_comparison.mp4"
    
    # Rutas
    dirty_path = os.path.join(dirty_dir, video_name)
    clean_path = os.path.join(clean_dir, video_name)
    corrected_path = os.path.join(corrected_dir, video_name)
    
    # Obtener frames
    dirty_frames = sorted(glob.glob(os.path.join(dirty_path, "*.jpg")))
    clean_frames = sorted(glob.glob(os.path.join(clean_path, "*.jpg")))
    corrected_frames = sorted(glob.glob(os.path.join(corrected_path, "*.jpg")))
    
    if not (dirty_frames and clean_frames and corrected_frames):
        raise ValueError("No se encontraron todos los frames necesarios")
    
    # Leer primer frame para obtener dimensiones
    sample = cv2.imread(dirty_frames[0])
    h, w = sample.shape[:2]
    
    # Crear video writer (3 frames lado a lado)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_name, fourcc, fps, (w * 3, h))
    
    print(f"Creando video de comparación para {video_name}...")
    
    for i, (d, c, cor) in enumerate(zip(dirty_frames, clean_frames, corrected_frames)):
        dirty_img = cv2.imread(d)
        clean_img = cv2.imread(c)
        corrected_img = cv2.imread(cor)
        
        # Agregar texto
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(dirty_img, "Dirty", (10, 30), font, 1, (0, 0, 255), 2)
        cv2.putText(clean_img, "Clean (GT)", (10, 30), font, 1, (0, 255, 0), 2)
        cv2.putText(corrected_img, "Corrected", (10, 30), font, 1, (255, 0, 0), 2)
        
        # Concatenar horizontalmente
        combined = np.hstack([dirty_img, clean_img, corrected_img])
        out.write(combined)
        
        if (i + 1) % 10 == 0:
            print(f"  Procesados {i + 1}/{len(dirty_frames)} frames")
    
    out.release()
    print(f"✓ Video de comparación guardado: {output_name}")
    return output_name


def process_all_videos(model_path="models/mobilenetv3_unet_stain_removal.keras"):
    """Procesa todos los videos del dataset."""
    # Cargar modelo
    print("Cargando modelo...")
    model = tf.keras.models.load_model(model_path)
    print("✓ Modelo cargado\n")
    
    # Obtener lista de videos
    dirty_dir = ".data/dirty_frames"
    videos = [d for d in os.listdir(dirty_dir) 
              if os.path.isdir(os.path.join(dirty_dir, d))]
    
    print(f"Se encontraron {len(videos)} videos para procesar")
    print("=" * 60)
    
    for i, video in enumerate(videos, 1):
        print(f"\n[{i}/{len(videos)}] Procesando: {video}")
        try:
            process_video(model, video, dirty_dir=dirty_dir, 
                         output_dir="corrected_frames", fps=30)
        except Exception as e:
            print(f"✗ Error procesando {video}: {e}")
            continue
    
    print("\n" + "=" * 60)
    print("✓ Procesamiento completado para todos los videos")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Modo: procesar un video específico y generar comparación
        video_name = sys.argv[1]
        
        # Cargar modelo y procesar
        print("Cargando modelo...")
        model = tf.keras.models.load_model("models/mobilenetv3_unet_stain_removal.keras")
        print("✓ Modelo cargado\n")
        
        # Procesar el video
        process_video(model, video_name)
        
        # Crear video de comparación
        create_comparison_video(video_name)
        
    else:
        # Modo: procesar todos los videos
        print("Procesando TODOS los videos del dataset")
        print("Para procesar solo uno, usa: python batch_inference.py <nombre_video>")
        print()
        
        response = input("¿Continuar? (y/n): ")
        if response.lower() == 'y':
            process_all_videos()
        else:
            print("Cancelado")

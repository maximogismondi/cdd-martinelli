import cv2
import numpy as np
import glob
import os
import tensorflow as tf
from pathlib import Path


def predict_mask(model, frame_path):
    """Predice la máscara de manchas para un frame."""
    frame = cv2.imread(frame_path)
    if frame is None:
        raise ValueError(f"No se pudo cargar el frame: {frame_path}")
    
    original_size = (frame.shape[1], frame.shape[0])  # (width, height)
    frame_resized = cv2.resize(frame, (256, 256)).astype("float32") / 255.
    
    pred = model.predict(np.expand_dims(frame_resized, 0), verbose=0)[0]
    mask = (pred > 0.5).astype("float32")
    
    # Redimensionar la máscara al tamaño original del frame
    mask_resized = cv2.resize(mask, original_size)
    mask_vis = (mask_resized * 255).astype("uint8")
    
    return mask_vis, original_size


def remove_stain_direct(model, dirty_path, out_path="corrected.png"):
    """Remueve manchas directamente sin guardar la máscara intermedia."""
    dirty = cv2.imread(dirty_path)
    if dirty is None:
        raise ValueError(f"No se pudo cargar el frame: {dirty_path}")
    
    original_size = (dirty.shape[1], dirty.shape[0])
    
    # Predecir máscara
    dirty_resized = cv2.resize(dirty, (256, 256)).astype("float32") / 255.
    pred = model.predict(np.expand_dims(dirty_resized, 0), verbose=0)[0]
    mask = (pred > 0.5).astype("float32")
    
    # Redimensionar máscara al tamaño original
    mask_resized = cv2.resize(mask, original_size)
    mask_3ch = cv2.merge([mask_resized, mask_resized, mask_resized])
    
    # Aplicar corrección
    dirty_float = dirty.astype("float32") / 255.
    corrected = dirty_float * (1 - mask_3ch)
    corrected = (corrected * 255).astype("uint8")
    
    cv2.imwrite(out_path, corrected)
    return corrected


def process_video(model, video_name, dirty_dir=".data/dirty_frames", 
                  output_dir="corrected_frames", fps=30):
    """Procesa todos los frames de un video y genera el video corregido."""
    
    # Rutas
    video_dirty_path = os.path.join(dirty_dir, video_name)
    video_output_path = os.path.join(output_dir, video_name)
    
    # Verificar que existe el directorio del video
    if not os.path.exists(video_dirty_path):
        raise ValueError(f"No existe el directorio: {video_dirty_path}")
    
    # Crear directorio de salida
    os.makedirs(video_output_path, exist_ok=True)
    
    # Obtener todos los frames
    frame_files = sorted(glob.glob(os.path.join(video_dirty_path, "*.jpg")) + 
                        glob.glob(os.path.join(video_dirty_path, "*.png")))
    
    if not frame_files:
        raise ValueError(f"No se encontraron frames en: {video_dirty_path}")
    
    print(f"Procesando {len(frame_files)} frames del video {video_name}...")
    
    # Procesar cada frame
    for i, frame_path in enumerate(frame_files):
        frame_name = os.path.basename(frame_path)
        out_path = os.path.join(video_output_path, frame_name)
        
        remove_stain_direct(model, frame_path, out_path)
        
        if (i + 1) % 10 == 0:
            print(f"  Procesados {i + 1}/{len(frame_files)} frames")
    
    print(f"✓ Todos los frames guardados en: {video_output_path}")
    
    # Generar video
    output_video_path = f"{video_name}_corrected.mp4"
    frames_to_video(video_output_path, output_video_path, fps)
    print(f"✓ Video generado: {output_video_path}")
    
    return output_video_path


def frames_to_video(frames_path, output_video="output.mp4", fps=30):
    """Convierte frames a video."""
    frame_files = sorted(glob.glob(os.path.join(frames_path, "*.png")) + 
                        glob.glob(os.path.join(frames_path, "*.jpg")))
    
    if not frame_files:
        raise ValueError(f"No se encontraron frames en: {frames_path}")
    
    frame = cv2.imread(frame_files[0])
    h, w = frame.shape[:2]
    
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_video, fourcc, fps, (w, h))
    
    print(f"Generando video con {len(frame_files)} frames...")
    for f in frame_files:
        img = cv2.imread(f)
        out.write(img)
    
    out.release()
    print(f"Video guardado: {output_video}")


if __name__ == "__main__":
    # Cargar el modelo
    print("Cargando modelo...")
    model = tf.keras.models.load_model("models/mobilenetv3_unet_stain_removal.keras")
    print("✓ Modelo cargado exitosamente")
    
    # Videos disponibles en el dataset de entrenamiento:
    # - Abuse001_x264_4
    # - Abuse003_x264_6
    # - Abuse004_x264_1
    # - Abuse006_x264_3
    # - Abuse008_x264_18
    # - Explosion004_x264_27
    # - RoadAccidents131_x264_6
    # - RoadAccidents132_x264_8
    # - Robbery142_x264_28
    # - Robbery145_x264_14
    
    # Elegir un video del dataset (cambia esto por el que quieras procesar)
    video_name = "Abuse003_x264_6"
    
    print(f"\nProcesando video: {video_name}")
    print("=" * 60)
    
    # Procesar el video completo
    process_video(model, video_name, dirty_dir=".data/dirty_frames", 
                  output_dir="corrected_frames", fps=30)
    
    print("\n" + "=" * 60)
    print("¡Procesamiento completado!")
    print(f"- Frames corregidos: corrected_frames/{video_name}/")
    print(f"- Video generado: {video_name}_corrected.mp4")
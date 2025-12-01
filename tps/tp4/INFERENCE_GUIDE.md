# Guía de Inferencia - MobileNetV3 U-Net para Remoción de Manchas

## Archivos Generados

### Scripts de Inferencia

- **`inference_mobile_net.py`**: Script principal para procesar un video individual
- **`batch_inference.py`**: Script para procesar múltiples videos y crear comparaciones

### Modelo Entrenado

- **`models/mobilenetv3_unet_stain_removal.keras`**: Modelo MobileNetV3-UNet entrenado (24 MB)

## Uso

### 1. Procesar un Video Individual

```bash
python3 inference_mobile_net.py
```

Por defecto procesa el video `Abuse001_x264_4`. Para cambiar el video, edita la variable `video_name` en el script.

**Videos disponibles en el dataset:**

- Abuse001_x264_4
- Abuse003_x264_6
- Abuse004_x264_1
- Abuse006_x264_3
- Abuse008_x264_18
- Explosion004_x264_27
- RoadAccidents131_x264_6
- RoadAccidents132_x264_8
- Robbery142_x264_28
- Robbery145_x264_14

**Salida:**

- Frames corregidos en: `corrected_frames/<video_name>/`
- Video MP4: `<video_name>_corrected.mp4`

### 2. Crear Video de Comparación (Dirty | Clean | Corrected)

```bash
python3 batch_inference.py Abuse001_x264_4
```

**Salida:**

- Video de comparación: `Abuse001_x264_4_comparison.mp4`
- Muestra lado a lado: frame dirty, ground truth clean, y frame corregido

### 3. Procesar TODOS los Videos

```bash
python3 batch_inference.py
```

Procesa todos los videos del dataset. Pide confirmación antes de comenzar.

## Estructura de Directorios

```
tps/tp4/
├── models/
│   └── mobilenetv3_unet_stain_removal.keras    # Modelo entrenado
├── .data/
│   ├── dirty_frames/                            # Dataset con manchas
│   │   ├── Abuse001_x264_4/
│   │   │   ├── frame_000000.jpg
│   │   │   └── ...
│   │   └── ...
│   └── clean_frames/                            # Ground truth (limpio)
│       └── ...
├── corrected_frames/                            # Frames procesados
│   └── Abuse001_x264_4/
│       ├── frame_000000.jpg
│       └── ...
├── Abuse001_x264_4_corrected.mp4               # Video corregido
└── Abuse001_x264_4_comparison.mp4              # Video comparación
```

## Detalles del Modelo

### Arquitectura

- **Encoder**: MobileNetV3-Small (pre-entrenado en ImageNet)
- **Decoder**: U-Net con skip connections
- **Entrada**: 256x256x3 (RGB)
- **Salida**: 256x256x1 (Máscara binaria de manchas)

### Entrenamiento

- Loss: Binary Cross-Entropy
- Optimizer: Adam (lr=1e-4)
- Epochs: 5
- Batch size: 4

### Inferencia

1. Redimensiona frame a 256x256
2. Normaliza a [0, 1]
3. Predice máscara de manchas (threshold=0.5)
4. Redimensiona máscara al tamaño original
5. Aplica corrección: `corrected = dirty * (1 - mask)`

## Funciones Principales

### `predict_mask(model, frame_path)`

Predice la máscara de manchas para un frame individual.

### `remove_stain_direct(model, dirty_path, out_path)`

Remueve manchas y guarda el frame corregido directamente.

### `process_video(model, video_name, ...)`

Procesa todos los frames de un video y genera el MP4 corregido.

### `create_comparison_video(video_name, ...)`

Crea un video de comparación lado a lado (Dirty | Clean | Corrected).

## Ejemplos de Uso en Python

```python
import tensorflow as tf
from inference_mobile_net import process_video, remove_stain_direct

# Cargar modelo
model = tf.keras.models.load_model("models/mobilenetv3_unet_stain_removal.keras")

# Procesar un video completo
process_video(model, "Robbery142_x264_28")

# Procesar un frame individual
remove_stain_direct(model,
                    "path/to/dirty_frame.jpg",
                    "output_corrected.jpg")
```

## Notas

- Los frames originales se redimensionan automáticamente al tamaño del modelo (256x256) y luego se restauran al tamaño original
- El video de salida mantiene el mismo FPS que el original (por defecto 30 FPS)
- La máscara se genera automáticamente comparando frames dirty vs clean con threshold de diferencia > 0.05

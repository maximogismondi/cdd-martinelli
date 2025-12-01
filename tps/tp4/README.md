# TP4 - Video Frame Cleaning con Deep Learning Temporal

Este proyecto implementa un sistema de limpieza de frames de video usando redes neuronales que aprovechan la dimensión temporal.

## Pipeline

### 1. Preparación de datos (`prepare_data.py`)

Genera el dataset a partir de videos:

- Extrae frames de videos
- Aplica manchas aleatorias (reproducibles por seed basada en nombre del video)
- Guarda frames limpios y sucios
- Escala frames a tamaño configurable

```bash
# Generar dataset con configuración por defecto
python3 prepare_data.py

# Personalizar manchas
python3 prepare_data.py --frame-size 128 --num-blobs 3 --min-radius 30 --max-radius 60

# Ver todas las opciones
python3 prepare_data.py --help
```

**Estructura de salida:**

```
.data/
├── raw_videos/          # Videos originales
├── clean_frames/        # Frames limpios
│   ├── video1/
│   │   ├── frame_000000.jpg
│   │   └── ...
│   └── video2/
└── dirty_frames/        # Frames con manchas
    ├── video1/
    └── video2/
```

### 2. Entrenamiento (`train_model.py`)

Entrena un modelo que considera la dimensión temporal usando secuencias de frames.

**Arquitecturas disponibles:**

1. **LSTM + CNN** (más ligero, recomendado):

   - CNN (TimeDistributed) extrae características espaciales de cada frame
   - LSTM captura dependencias temporales entre frames
   - Decoder reconstruye el frame limpio

2. **ConvLSTM** (más pesado, más expresivo):
   - ConvLSTM2D procesa patrones espaciales y temporales simultáneamente
   - Ideal si tienes GPU potente

```bash
# Entrenamiento básico con LSTM+CNN
python3 train_model.py --model-type lstm_cnn --epochs 50 --batch-size 8

# Entrenamiento con ConvLSTM
python3 train_model.py --model-type convlstm --epochs 50 --batch-size 4

# Personalizar secuencia temporal
python3 train_model.py --sequence-length 7 --learning-rate 0.0005

# Ver todas las opciones
python3 train_model.py --help
```

**Parámetros importantes:**

- `--sequence-length`: Número de frames consecutivos (default: 5)
  - Más frames = más contexto temporal pero más memoria
- `--model-type`: `lstm_cnn` (más rápido) o `convlstm` (más potente)
- `--batch-size`: Ajustar según memoria disponible
- `--val-split`: Proporción de videos para validación (default: 0.2)

**Salida:**

```
models/
├── best_model_lstm_cnn.keras       # Mejor modelo (checkpoint)
├── final_model_lstm_cnn.keras      # Modelo final
├── config_lstm_cnn.json            # Configuración
└── training_log_lstm_cnn.csv       # Historial de entrenamiento
```

### 3. Inferencia (`inference.py`)

Aplica el modelo entrenado para limpiar frames de videos.

```bash
# Procesar todos los videos
python3 inference.py \
  --model-path models/best_model_lstm_cnn.keras \
  --config-path models/config_lstm_cnn.json

# Procesar un video específico
python3 inference.py \
  --model-path models/best_model_lstm_cnn.keras \
  --config-path models/config_lstm_cnn.json \
  --video-name video1

# Guardar comparaciones (dirty | predicción | limpio)
python3 inference.py \
  --model-path models/best_model_lstm_cnn.keras \
  --config-path models/config_lstm_cnn.json \
  --save-comparison
```

**Salida:**

```
predictions/
├── video1/
│   ├── frame_000000.jpg     # Frames limpios predichos
│   └── ...
├── video1_comparison/       # Solo si --save-comparison
│   ├── frame_000000.jpg     # Comparación lado a lado
│   └── ...
└── video2/
```

## Cómo funciona la dimensión temporal

El modelo no procesa frames individuales, sino **secuencias de frames consecutivos**:

1. **Entrada**: Secuencia de N frames sucios consecutivos

   - Ejemplo con `sequence_length=5`: frames [t-2, t-1, t, t+1, t+2]

2. **Procesamiento**:

   - **CNN**: Extrae características espaciales de cada frame
   - **LSTM/ConvLSTM**: Captura patrones temporales (movimiento, cambios)

3. **Salida**: Frame limpio central de la secuencia
   - Predice el frame en tiempo `t` usando contexto temporal

**Ventajas de usar contexto temporal:**

- Reduce ruido usando información de frames vecinos
- Detecta mejor áreas estáticas vs. dinámicas
- Más robusto que procesar frames aislados
- Aprende patrones de movimiento de la cámara

## Requisitos

```bash
# Instalar dependencias
pip install tensorflow opencv-python numpy pillow

# O desde requirements.txt (si ya incluye tensorflow)
pip install -r requirements.txt
```

## Workflow completo ejemplo

```bash
# 1. Preparar datos (coloca videos en .data/raw_videos/)
python3 prepare_data.py --frame-size 128 --num-blobs 3

# 2. Entrenar modelo
python3 train_model.py \
  --model-type lstm_cnn \
  --sequence-length 5 \
  --epochs 50 \
  --batch-size 8

# 3. Hacer predicciones con comparaciones
python3 inference.py \
  --model-path models/best_model_lstm_cnn.keras \
  --config-path models/config_lstm_cnn.json \
  --save-comparison

# 4. Ver resultados en predictions/
```

## Tips y recomendaciones

### Ajuste de hiperparámetros

- **Sequence length**:

  - Corto (3-5): Menos memoria, entrenamiento más rápido
  - Largo (7-10): Más contexto temporal, mejor para escenas complejas

- **Batch size**:

  - Depende de la memoria GPU/CPU
  - ConvLSTM necesita más memoria que LSTM+CNN

- **Learning rate**:
  - Empezar con 0.001
  - Reducir si el loss oscila mucho

### Troubleshooting

**Error de memoria:**

```bash
# Reducir batch size o sequence length
python3 train_model.py --batch-size 4 --sequence-length 3
```

**Entrenamiento lento:**

```bash
# Usar LSTM+CNN en vez de ConvLSTM
python3 train_model.py --model-type lstm_cnn
```

**Resultados no satisfactorios:**

- Aumentar `--epochs` (50-100)
- Aumentar `--sequence-length` (7-10)
- Probar con manchas más/menos agresivas en `prepare_data.py`
- Agregar más videos de entrenamiento

## Estructura del proyecto

```
tp4/
├── README.md              # Este archivo
├── prepare_data.py        # Generación de dataset
├── train_model.py         # Entrenamiento del modelo
├── inference.py           # Predicciones
├── .data/                 # Datos (git-ignored)
│   ├── raw_videos/
│   ├── clean_frames/
│   └── dirty_frames/
├── models/                # Modelos entrenados (git-ignored)
└── predictions/           # Resultados (git-ignored)
```

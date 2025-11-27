import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from pathlib import Path
import argparse
from typing import Tuple, List
import json


class VideoFrameDataset(keras.utils.Sequence):
    """
    Dataset personalizado que carga secuencias de frames temporales.
    Cada muestra es una secuencia de N frames consecutivos.
    """

    def __init__(
        self,
        dirty_dirs: List[Path],
        clean_dirs: List[Path],
        sequence_length: int = 5,
        batch_size: int = 8,
        shuffle: bool = True,
    ):
        """
        Args:
            dirty_dirs: Lista de directorios con frames sucios
            clean_dirs: Lista de directorios con frames limpios
            sequence_length: Número de frames consecutivos por secuencia
            batch_size: Tamaño del batch
            shuffle: Si mezclar las secuencias
        """
        self.dirty_dirs = dirty_dirs
        self.clean_dirs = clean_dirs
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Construir índice de secuencias disponibles
        self.sequences = []
        for dirty_dir, clean_dir in zip(dirty_dirs, clean_dirs):
            frames = sorted(list(dirty_dir.glob("*.jpg")))
            num_sequences = max(0, len(frames) - sequence_length + 1)
            for i in range(num_sequences):
                self.sequences.append((dirty_dir, clean_dir, i))

        self.indexes = np.arange(len(self.sequences))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __len__(self):
        """Número de batches por época"""
        return len(self.sequences) // self.batch_size

    def __getitem__(self, index):
        """Genera un batch de datos"""
        # Seleccionar índices para este batch
        batch_indexes = self.indexes[
            index * self.batch_size : (index + 1) * self.batch_size
        ]

        # Generar datos
        X = []
        y = []

        for idx in batch_indexes:
            dirty_dir, clean_dir, start_idx = self.sequences[idx]

            # Cargar secuencia de frames sucios (entrada)
            dirty_sequence = []
            for i in range(self.sequence_length):
                frame_path = dirty_dir / f"frame_{start_idx + i:06d}.jpg"
                img = tf.keras.preprocessing.image.load_img(frame_path)
                img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
                dirty_sequence.append(img_array)

            # Cargar frame limpio central (objetivo)
            # Usamos el frame del medio de la secuencia
            middle_idx = start_idx + self.sequence_length // 2
            clean_path = clean_dir / f"frame_{middle_idx:06d}.jpg"
            clean_img = tf.keras.preprocessing.image.load_img(clean_path)
            clean_array = tf.keras.preprocessing.image.img_to_array(clean_img) / 255.0

            X.append(dirty_sequence)
            y.append(clean_array)

        return np.array(X), np.array(y)

    def on_epoch_end(self):
        """Mezclar después de cada época"""
        if self.shuffle:
            np.random.shuffle(self.indexes)


def build_convlstm_autoencoder(input_shape: Tuple, learning_rate: float = 0.001):
    """
    Construye un autoencoder con ConvLSTM2D para procesar secuencias temporales.

    Args:
        input_shape: (sequence_length, height, width, channels)
        learning_rate: Tasa de aprendizaje

    Returns:
        Modelo compilado
    """
    inputs = keras.Input(shape=input_shape)

    # Encoder con ConvLSTM2D
    # Captura patrones espaciales y temporales
    x = layers.ConvLSTM2D(
        filters=64,
        kernel_size=(3, 3),
        padding="same",
        return_sequences=True,
        activation="relu",
    )(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.TimeDistributed(layers.MaxPooling2D((2, 2)))(x)

    x = layers.ConvLSTM2D(
        filters=128,
        kernel_size=(3, 3),
        padding="same",
        return_sequences=True,
        activation="relu",
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.TimeDistributed(layers.MaxPooling2D((2, 2)))(x)

    x = layers.ConvLSTM2D(
        filters=256,
        kernel_size=(3, 3),
        padding="same",
        return_sequences=False,  # Solo queremos el estado final
        activation="relu",
    )(x)
    x = layers.BatchNormalization()(x)

    # Bottleneck
    # Expandir dimensión temporal para el decoder
    x = layers.Reshape((1, x.shape[1], x.shape[2], x.shape[3]))(x)

    # Decoder con Conv2DTranspose
    x = layers.TimeDistributed(
        layers.Conv2DTranspose(
            128, (3, 3), strides=(2, 2), padding="same", activation="relu"
        )
    )(x)
    x = layers.TimeDistributed(layers.BatchNormalization())(x)

    x = layers.TimeDistributed(
        layers.Conv2DTranspose(
            64, (3, 3), strides=(2, 2), padding="same", activation="relu"
        )
    )(x)
    x = layers.TimeDistributed(layers.BatchNormalization())(x)

    # Capa de salida: reconstruir el frame limpio
    outputs = layers.TimeDistributed(
        layers.Conv2D(3, (3, 3), padding="same", activation="sigmoid")
    )(x)

    # Remover dimensión temporal (solo predecimos 1 frame)
    outputs = layers.Reshape((input_shape[1], input_shape[2], 3))(outputs)

    model = keras.Model(inputs=inputs, outputs=outputs, name="ConvLSTM_Cleaner")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=["mae", tf.keras.metrics.RootMeanSquaredError(name="rmse")],
    )

    return model


def build_lstm_cnn_autoencoder(input_shape: Tuple, learning_rate: float = 0.001):
    """
    Construye un autoencoder con CNN + LSTM (alternativa más ligera).

    Args:
        input_shape: (sequence_length, height, width, channels)
        learning_rate: Tasa de aprendizaje

    Returns:
        Modelo compilado
    """
    inputs = keras.Input(shape=input_shape)

    # Encoder CNN para cada frame (TimeDistributed)
    x = layers.TimeDistributed(
        layers.Conv2D(32, (3, 3), activation="relu", padding="same")
    )(inputs)
    x = layers.TimeDistributed(layers.MaxPooling2D((2, 2)))(x)
    x = layers.TimeDistributed(layers.BatchNormalization())(x)

    x = layers.TimeDistributed(
        layers.Conv2D(64, (3, 3), activation="relu", padding="same")
    )(x)
    x = layers.TimeDistributed(layers.MaxPooling2D((2, 2)))(x)
    x = layers.TimeDistributed(layers.BatchNormalization())(x)

    x = layers.TimeDistributed(
        layers.Conv2D(128, (3, 3), activation="relu", padding="same")
    )(x)
    x = layers.TimeDistributed(layers.MaxPooling2D((2, 2)))(x)
    x = layers.TimeDistributed(layers.BatchNormalization())(x)

    # Flatten para LSTM
    shape_before_flatten = x.shape
    x = layers.TimeDistributed(layers.Flatten())(x)

    # LSTM para capturar dependencias temporales
    x = layers.LSTM(512, activation="relu", return_sequences=False)(x)
    x = layers.Dropout(0.3)(x)

    # Reshape para decoder
    flatten_size = (
        shape_before_flatten[2] * shape_before_flatten[3] * shape_before_flatten[4]
    )
    x = layers.Dense(flatten_size, activation="relu")(x)
    x = layers.Reshape(
        (shape_before_flatten[2], shape_before_flatten[3], shape_before_flatten[4])
    )(x)

    # Decoder con Conv2DTranspose
    x = layers.Conv2DTranspose(
        64, (3, 3), strides=(2, 2), padding="same", activation="relu"
    )(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv2DTranspose(
        32, (3, 3), strides=(2, 2), padding="same", activation="relu"
    )(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv2DTranspose(
        16, (3, 3), strides=(2, 2), padding="same", activation="relu"
    )(x)
    x = layers.BatchNormalization()(x)

    # Salida
    outputs = layers.Conv2D(3, (3, 3), padding="same", activation="sigmoid")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="LSTM_CNN_Cleaner")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=["mae", tf.keras.metrics.RootMeanSquaredError(name="rmse")],
    )

    return model


def load_datasets(dirty_dir: Path, clean_dir: Path, val_split: float = 0.2):
    """
    Carga y divide los datasets en entrenamiento y validación.

    Args:
        dirty_dir: Directorio base con frames sucios
        clean_dir: Directorio base con frames limpios
        val_split: Proporción para validación

    Returns:
        Tupla con listas de directorios (train_dirty, train_clean, val_dirty, val_clean)
    """
    # Obtener todos los subdirectorios (un subdirectorio por video)
    dirty_videos = sorted([d for d in dirty_dir.iterdir() if d.is_dir()])
    clean_videos = sorted([d for d in clean_dir.iterdir() if d.is_dir()])

    assert len(dirty_videos) == len(
        clean_videos
    ), "Número diferente de videos en dirty y clean"

    print(f"Total de videos encontrados: {len(dirty_videos)}")

    # Split train/val
    num_val = max(1, int(len(dirty_videos) * val_split))
    num_train = len(dirty_videos) - num_val

    train_dirty = dirty_videos[:num_train]
    train_clean = clean_videos[:num_train]
    val_dirty = dirty_videos[num_train:]
    val_clean = clean_videos[num_train:]

    print(f"Videos de entrenamiento: {num_train}")
    print(f"Videos de validación: {num_val}")

    return train_dirty, train_clean, val_dirty, val_clean


def main():
    parser = argparse.ArgumentParser(
        description="Entrena modelo para limpiar frames con manchas usando contexto temporal"
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
        help="Directorio con frames limpios",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="lstm_cnn",
        choices=["convlstm", "lstm_cnn"],
        help="Tipo de arquitectura (convlstm o lstm_cnn)",
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=5,
        help="Número de frames consecutivos en cada secuencia (default: 5)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=8, help="Tamaño del batch (default: 8)"
    )
    parser.add_argument(
        "--epochs", type=int, default=50, help="Número de épocas (default: 50)"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
        help="Tasa de aprendizaje (default: 0.001)",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.2,
        help="Proporción de videos para validación (default: 0.2)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models",
        help="Directorio para guardar el modelo (default: models)",
    )

    args = parser.parse_args()

    # Configurar GPU si está disponible
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        print(f"GPU(s) detectada(s): {len(gpus)}")
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(f"Error configurando GPU: {e}")
    else:
        print("No se detectó GPU, usando CPU")

    # Paths
    dirty_dir = Path(args.dirty_frames_dir)
    clean_dir = Path(args.clean_frames_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # Verificar directorios
    if not dirty_dir.exists() or not clean_dir.exists():
        print(f"Error: Los directorios {dirty_dir} o {clean_dir} no existen")
        return

    # Cargar datasets
    print("\n=== Cargando datasets ===")
    train_dirty, train_clean, val_dirty, val_clean = load_datasets(
        dirty_dir, clean_dir, args.val_split
    )

    # Obtener dimensiones de imagen del primer frame
    sample_frame = list(train_dirty[0].glob("*.jpg"))[0]
    sample_img = tf.keras.preprocessing.image.load_img(sample_frame)
    img_height, img_width = sample_img.size[1], sample_img.size[0]

    print(f"\nDimensiones de frames: {img_height}x{img_width}")
    print(f"Secuencia temporal: {args.sequence_length} frames")

    # Crear datasets
    print("\n=== Creando generadores de datos ===")
    train_dataset = VideoFrameDataset(
        train_dirty,
        train_clean,
        sequence_length=args.sequence_length,
        batch_size=args.batch_size,
        shuffle=True,
    )

    val_dataset = VideoFrameDataset(
        val_dirty,
        val_clean,
        sequence_length=args.sequence_length,
        batch_size=args.batch_size,
        shuffle=False,
    )

    print(f"Batches de entrenamiento: {len(train_dataset)}")
    print(f"Batches de validación: {len(val_dataset)}")

    # Construir modelo
    print(f"\n=== Construyendo modelo: {args.model_type} ===")
    input_shape = (args.sequence_length, img_height, img_width, 3)

    if args.model_type == "convlstm":
        model = build_convlstm_autoencoder(input_shape, args.learning_rate)
    else:
        model = build_lstm_cnn_autoencoder(input_shape, args.learning_rate)

    model.summary()

    # Callbacks
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            output_dir / f"best_model_{args.model_type}.keras",
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=10, restore_best_weights=True, verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=5, min_lr=1e-7, verbose=1
        ),
        keras.callbacks.CSVLogger(output_dir / f"training_log_{args.model_type}.csv"),
    ]

    # Entrenar
    print(f"\n=== Iniciando entrenamiento ===")
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=1,
    )

    # Guardar modelo final y configuración
    model.save(output_dir / f"final_model_{args.model_type}.keras")

    config = {
        "model_type": args.model_type,
        "sequence_length": args.sequence_length,
        "input_shape": input_shape,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "final_val_loss": float(history.history["val_loss"][-1]),
        "best_val_loss": float(min(history.history["val_loss"])),
    }

    with open(output_dir / f"config_{args.model_type}.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"\n=== Entrenamiento completado ===")
    print(f"Mejor val_loss: {config['best_val_loss']:.6f}")
    print(f"Modelo guardado en: {output_dir}")


if __name__ == "__main__":
    main()

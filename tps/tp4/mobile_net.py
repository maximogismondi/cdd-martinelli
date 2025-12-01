import tensorflow as tf
from tensorflow.keras import layers, models

def mobilenetv3_unet(input_shape=(256, 256, 3), num_classes=1):
    # === ENCODER ===
    base_model = tf.keras.applications.MobileNetV3Small(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet"
    )

    # Extraemos los mapas de características para skip-connections
    encoder_outputs = [
        base_model.get_layer("expanded_conv_project").output,     # 64x64
    base_model.get_layer("expanded_conv_2_project").output,   # 32x32
        base_model.get_layer("expanded_conv_6_project").output,   # 16x16
        base_model.get_layer("conv_1").output                     # 8x8 · deepest
    ]

    # Creamos un modelo del encoder
    encoder = models.Model(inputs=base_model.input, outputs=encoder_outputs)

    # === DECODER (UNET) ===
    inputs = tf.keras.Input(shape=input_shape)
    skips = encoder(inputs)

    skip1, skip2, skip3, bottleneck = skips  # from shallow → deep

    x = bottleneck  # start decoding

    # Bloque de upsampling + conv (helper)
    def up_block(x, skip, filters):
        x = layers.UpSampling2D(size=(2, 2), interpolation="bilinear")(x)
        x = layers.Concatenate()([x, skip])
        x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
        x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
        return x

    # Decoder path
    x = up_block(x, skip3, 128)  # 8×8 → 16×16
    x = up_block(x, skip2, 64)   # 16×16 → 32×32
    x = up_block(x, skip1, 32)   # 32×32 → 64×64

    # Upsample to 128x128
    x = layers.UpSampling2D(size=(2, 2), interpolation="bilinear")(x)  # 64→128
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    
    # Último upsampling hasta el tamaño original 256x256
    x = layers.UpSampling2D(size=(2, 2), interpolation="bilinear")(x)  # 128→256
    x = layers.Conv2D(16, 3, padding="same", activation="relu")(x)

    # Capa de salida
    outputs = layers.Conv2D(num_classes, 1, activation="sigmoid")(x)

    model = models.Model(inputs, outputs)
    return model



if __name__ == "__main__":
    import numpy as np
    import cv2
    import os

    class StainDataset(tf.keras.utils.Sequence):
        def __init__(self, clean_dir, dirty_dir, batch_size=8, img_size=(256,256), **kwargs):
            super().__init__(**kwargs)
            self.clean_dir = clean_dir
            self.dirty_dir = dirty_dir
            self.batch_size = batch_size
            self.img_h, self.img_w = img_size
            
            # Collect all image paths from subdirectories
            self.image_pairs = []
            video_dirs = sorted(os.listdir(clean_dir))
            
            for video_dir in video_dirs:
                clean_video_path = os.path.join(clean_dir, video_dir)
                dirty_video_path = os.path.join(dirty_dir, video_dir)
                
                # Skip if not a directory
                if not os.path.isdir(clean_video_path):
                    continue
                    
                # Get all frames from this video
                frames = sorted([f for f in os.listdir(clean_video_path) if f.endswith(('.jpg', '.png', '.jpeg'))])
                
                for frame in frames:
                    clean_frame_path = os.path.join(clean_video_path, frame)
                    dirty_frame_path = os.path.join(dirty_video_path, frame)
                    self.image_pairs.append((clean_frame_path, dirty_frame_path))

        def __len__(self):
            return len(self.image_pairs) // self.batch_size

        def __getitem__(self, idx):
            batch_x = []
            batch_y = []

            batch_pairs = self.image_pairs[idx*self.batch_size:(idx+1)*self.batch_size]

            for clean_path, dirty_path in batch_pairs:
                clean = cv2.imread(clean_path)
                dirty = cv2.imread(dirty_path)
                
                # Check if images were loaded successfully
                if clean is None:
                    raise ValueError(f"Failed to load clean image: {clean_path}")
                if dirty is None:
                    raise ValueError(f"Failed to load dirty image: {dirty_path}")

                # Resize
                clean = cv2.resize(clean, (self.img_w, self.img_h))
                dirty = cv2.resize(dirty, (self.img_w, self.img_h))

                clean = clean.astype("float32") / 255.
                dirty = dirty.astype("float32") / 255.

                # === Generar MÁSCARA DE MANCHAS automát. ===
                diff = np.abs(dirty - clean).mean(axis=2, keepdims=True)
                mask = (diff > 0.05).astype("float32")  # umbral

                batch_x.append(dirty)
                batch_y.append(mask)

            return np.array(batch_x), np.array(batch_y)


    train_ds = StainDataset(
        clean_dir=".data/clean_frames",
        dirty_dir=".data/dirty_frames",
        batch_size=4,
        img_size=(256,256)
    )


    model = mobilenetv3_unet(
        input_shape=(256, 256, 3),
        num_classes=1
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    model.summary()


    model.fit(train_ds, epochs=5)

    model.save("models/mobilenetv3_unet_stain_removal.keras")




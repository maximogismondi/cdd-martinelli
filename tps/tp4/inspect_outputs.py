import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV3Small

base = MobileNetV3Small(input_shape=(256,256,3), include_top=False, weights='imagenet')

layer_names = [
    'expanded_conv_project',
    'expanded_conv_3_project',
    'expanded_conv_6_project',
    'conv_1'
]

layers_found = []
for name in layer_names:
    try:
        layers_found.append(base.get_layer(name).output)
    except Exception as e:
        print(f"Layer {name} not found: {e}")

sub = tf.keras.Model(inputs=base.input, outputs=layers_found)

x = np.zeros((1,256,256,3), dtype=np.float32)
outs = sub.predict(x)

for name, out in zip(layer_names, outs):
    print(f"{name}: {out.shape}")

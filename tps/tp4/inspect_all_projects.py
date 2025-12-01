import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV3Small

base = MobileNetV3Small(input_shape=(256,256,3), include_top=False, weights='imagenet')

project_layers = [layer.name for layer in base.layers if 'project' in layer.name]
print('Found project layers:', project_layers)

outputs = [base.get_layer(name).output for name in project_layers]
sub = tf.keras.Model(inputs=base.input, outputs=outputs)

x = np.zeros((1,256,256,3), dtype=np.float32)
outs = sub.predict(x)

for name, out in zip(project_layers, outs):
    print(f"{name}: {out.shape}")

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV3Small

model = MobileNetV3Small(input_shape=(256,256,3), include_top=False, weights='imagenet')

for i, layer in enumerate(model.layers):
    try:
        shape = layer.output_shape
    except Exception:
        shape = None
    print(f"{i:03d}: {layer.name} -> {shape}")

# Also print a short filtered list of layers with 'project' or 'Conv' in name
print('\nFiltered layers:')
for layer in model.layers:
    if 'project' in layer.name or 'conv' in layer.name.lower():
        try:
            shape = layer.output_shape
        except Exception:
            shape = None
        print(f"{layer.name}: {shape}")

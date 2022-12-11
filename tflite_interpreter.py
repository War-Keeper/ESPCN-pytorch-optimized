###########################################################
# HAVE TO RUN THIS IN GOOGLE COLLAB
# does not work on local
###########################################################

import tensorflow as tf
import numpy as np
import time

# Load the TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="other_models/X4.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Test the model on random input data.
input_shape = input_details[0]['shape']
input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
interpreter.set_tensor(input_details[0]['index'], input_data)

t = time.time()
interpreter.invoke()
t = time.time() - t
print(t)

# The function `get_tensor()` returns a copy of the tensor data.
# Use `tensor()` in order to get a pointer to the tensor.
output_data = interpreter.get_tensor(output_details[0]['index'])
print(output_data)

# example
# [[[[0.13277414]
#    [0.12438094]
#    [0.12190379]
#    [0.09487992]]

#   [[0.1226983 ]
#    [0.15142697]
#    [0.12933587]
#    [0.11702529]]

#   [[0.13736868]
#    [0.15062958]
#    [0.14164498]
#    [0.13684091]]

#   [[0.12234847]
#    [0.16925363]
#    [0.17504045]
#    [0.15958051]]]]
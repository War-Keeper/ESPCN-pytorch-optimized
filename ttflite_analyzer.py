###########################################################
# HAVE TO RUN THIS IN GOOGLE COLLAB
# does not work on local
###########################################################

import tensorflow as tf

# No quantization
h5_model = model = tf.keras.models.load_model('X4.h5')

converter = tf.lite.TFLiteConverter.from_keras_model(model)

model = converter.convert()

tf.lite.experimental.Analyzer.analyze(model_path=None,
                                      model_content=model,
                                      gpu_compatibility=False)


# With dynamic range quantization
h5_model = model = tf.keras.models.load_model('X4.h5')

converter = tf.lite.TFLiteConverter.from_keras_model(model)

converter.optimizations = [tf.lite.Optimize.DEFAULT]

model = converter.convert()

tf.lite.experimental.Analyzer.analyze(model_path=None,
                                      model_content=model,
                                      gpu_compatibility=False)


###########################################################
# Output 
# === TFLite ModelAnalyzer ===

# Your TFLite model has '1' subgraph(s). In the subgraph description below,
# T# represents the Tensor numbers. For example, in Subgraph#0, the CONV_2D op takes
# tensor #0 and tensor #5 and tensor #4 as input and produces tensor #9 as output.

# Subgraph#0 main(T#0) -> [T#13]
#   Op#0 CONV_2D(T#0, T#5, T#4) -> [T#9]
#   Op#1 CONV_2D(T#9, T#6, T#3) -> [T#10]
#   Op#2 CONV_2D(T#10, T#7, T#2) -> [T#11]
#   Op#3 CONV_2D(T#11, T#8, T#1) -> [T#12]
#   Op#4 DEPTH_TO_SPACE(T#12) -> [T#13]

# Tensors of Subgraph#0
#   T#0(serving_default_input_2:0) shape_signature:[-1, -1, -1, 1], type:FLOAT32
#   T#1(model_1/conv2d_7/BiasAdd/ReadVariableOp) shape:[16], type:FLOAT32 RO 64 bytes
#   T#2(model_1/conv2d_6/BiasAdd/ReadVariableOp) shape:[32], type:FLOAT32 RO 128 bytes
#   T#3(model_1/conv2d_5/BiasAdd/ReadVariableOp) shape:[64], type:FLOAT32 RO 256 bytes
#   T#4(model_1/conv2d_4/BiasAdd/ReadVariableOp) shape:[64], type:FLOAT32 RO 256 bytes
#   T#5(model_1/conv2d_4/Conv2D) shape:[64, 5, 5, 1], type:FLOAT32 RO 6400 bytes
#   T#6(model_1/conv2d_5/Conv2D) shape:[64, 3, 3, 64], type:FLOAT32 RO 147456 bytes
#   T#7(model_1/conv2d_6/Conv2D) shape:[32, 3, 3, 64], type:FLOAT32 RO 73728 bytes
#   T#8(model_1/conv2d_7/Conv2D) shape:[16, 3, 3, 32], type:FLOAT32 RO 18432 bytes
#   T#9(model_1/conv2d_4/Relu;model_1/conv2d_4/BiasAdd;model_1/conv2d_5/Conv2D;model_1/conv2d_4/Conv2D;model_1/conv2d_4/BiasAdd/ReadVariableOp) shape_signature:[-1, -1, -1, 64], type:FLOAT32
#   T#10(model_1/conv2d_5/Relu;model_1/conv2d_5/BiasAdd;model_1/conv2d_5/Conv2D;model_1/conv2d_5/BiasAdd/ReadVariableOp) shape_signature:[-1, -1, -1, 64], type:FLOAT32
#   T#11(model_1/conv2d_6/Relu;model_1/conv2d_6/BiasAdd;model_1/conv2d_6/Conv2D;model_1/conv2d_6/BiasAdd/ReadVariableOp) shape_signature:[-1, -1, -1, 32], type:FLOAT32
#   T#12(model_1/conv2d_7/Relu;model_1/conv2d_7/BiasAdd;model_1/conv2d_7/Conv2D;model_1/conv2d_7/BiasAdd/ReadVariableOp) shape_signature:[-1, -1, -1, 16], type:FLOAT32
#   T#13(StatefulPartitionedCall:0) shape_signature:[-1, -1, -1, 1], type:FLOAT32

# ---------------------------------------------------------------
# Your TFLite model has ‘1’ signature_def(s).

# Signature#0 key: 'serving_default'
# - Subgraph: Subgraph#0
# - Inputs: 
#     'input_2' : T#0
# - Outputs: 
#     'tf.nn.depth_to_space_1' : T#13

# ---------------------------------------------------------------
#               Model size:     249484 bytes
#     Non-data buffer size:       2664 bytes (01.07 %)
#   Total data buffer size:     246820 bytes (98.93 %)
#     (Zero value buffers):          0 bytes (00.00 %)

# * Buffers of TFLite model are mostly used for constant tensors.
#   And zero value buffers are buffers filled with zeros.
#   Non-data buffers area are used to store operators, subgraphs and etc.
#   You can find more details from https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/schema/schema.fbs


##########################
# With dynamic range quantization
# ---------------------------------------------------------------
# Your TFLite model has ‘1’ signature_def(s).

# Signature#0 key: 'serving_default'
# - Subgraph: Subgraph#0
# - Inputs: 
#     'input_2' : T#0
# - Outputs: 
#     'tf.nn.depth_to_space_1' : T#13

# ---------------------------------------------------------------
#               Model size:      67224 bytes
#     Non-data buffer size:       4912 bytes (07.31 %)
#   Total data buffer size:      62312 bytes (92.69 %)
#     (Zero value buffers):          0 bytes (00.00 %)
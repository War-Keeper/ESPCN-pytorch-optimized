###########################################################
# HAVE TO RUN THIS IN GOOGLE COLLAB
# does not work on local
###########################################################
import tensorflow as tf

model = tf.keras.models.load_model('X4.h5')
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Dynamic range Quantization
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Float16 quantization
# converter.target_spec.supported_types = [tf.float16]

model = converter.convert()
file = open( 'X4.tflite' , 'wb' ) 
file.write( model )
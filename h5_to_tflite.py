###########################################################
# HAVE TO RUN THIS IN GOOGLE COLLAB
# does not work on local
###########################################################
import tensorflow as tf

model = tf.keras.models.load_model('X4.h5')
converter = tf.lite.TFLiteConverter.from_keras_model(model)
model = converter.convert()
file = open( 'X4.tflite' , 'wb' ) 
file.write( model )
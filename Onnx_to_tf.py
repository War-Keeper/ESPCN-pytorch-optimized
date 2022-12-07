###########################################################
# HAVE TO RUN THIS IN GOOGLE COLLAB
# does not work on local
###########################################################

# !pip install onnx
# !pip install onnx-tf

import onnx

from onnx_tf.backend import prepare

# from google.colab import files
# uploaded = files.upload()

# from google.colab import files
# uploaded = files.upload()

def convert_to_frozen_pb(file_in, file_out):
    onnx_model = onnx.load(file_in)  # load onnx model
    tf_rep = prepare(onnx_model)  # prepare tf representation
    tf_rep.export_graph(file_out)  # export the model

convert_to_frozen_pb("model/best.onnx", "model/best.pb")
convert_to_frozen_pb("model/pruned_best_149.onnx", "model/pruned_best_149.pb")

# run this for colabl, not the 2 above
# convert_to_pb("best.onnx", "best.pb")
# convert_to_pb("pruned_best_149.onnx", "pruned_best_149.pb")


from onnx2keras import onnx_to_keras

def convert_to_keras(file_in, file_out):
    # Load ONNX model
    onnx_model = onnx.load('best.onnx')
    print(onnx_model)

    # Call the converter (input - is the main model input name, can be different for your model)
    k_model = onnx_to_keras(onnx_model, ['input'])
    k_model.save("best.h5")
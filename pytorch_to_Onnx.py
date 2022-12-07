import torch

def convert_to_ONNX(file_in, file_out):

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    model = torch.load(file_in).to(device)
    dummy_input=torch.randn([256, 1, 28, 28]).to(device)
    torch.onnx.export(model, dummy_input, file_out)

convert_to_ONNX("outputs/pruned_x8/pruned_best_149.pth", "model/pruned_best_149.onnx")
convert_to_ONNX("model/best.pth", "model/best.onnx")

# model = torch.load("model/best.pth")
# model.eval()
# print('Finished loading model!')
# print(model)

# device = torch.device("cuda")
# model = model.to(device)

# input_names = ["input"]
# output_names = ["output"]
# inputs = torch.randn([256, 1, 28, 28]).to(device)

# torch_out = torch.onnx._export(model, inputs, 'best2.pth', export_params=True, verbose=False,
#                                input_names=input_names, output_names=output_names)
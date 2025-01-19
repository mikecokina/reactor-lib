import torch
from reactorlib.reswapper.stf_128 import StyleTransferModel

import onnx
from onnx import numpy_helper

onnx_model = onnx.load("inswapper_128.onnx")
INTIALIZERS = onnx_model.graph.initializer
onnx_weights = {}

for initializer in INTIALIZERS:
    W = numpy_helper.to_array(initializer)
    onnx_weights[initializer.name] = W

model = StyleTransferModel()
weight_shapes = []
for n, p in model.named_parameters():
    weight_shapes.append((n, '-'.join([str(x) for x in list(p.shape)])))

replacement_dict = {
    'styles': 'style_blocks',
    'conv1.1': 'conv1.conv',
    'style1.linear': 'conv1.adain.style',
    'conv2.1': 'conv2.conv',
    'style2.linear': 'conv2.adain.style',
    'up0.1': 'up.8',
    'onnx::Conv_833': 'down.0.weight',
    'onnx::Conv_834': 'down.0.bias',
    'onnx::Conv_836': 'down.2.weight',
    'onnx::Conv_837': 'down.2.bias',
    'onnx::Conv_839': 'down.4.weight',
    'onnx::Conv_840': 'down.4.bias',
    'onnx::Conv_842': 'down.6.weight',
    'onnx::Conv_843': 'down.6.bias',
    'onnx::Conv_845': 'up.1.weight',
    'onnx::Conv_846': 'up.1.bias',
    'onnx::Conv_848': 'up.4.weight',
    'onnx::Conv_849': 'up.4.bias',
    'onnx::Conv_851': 'up.6.weight',
    'onnx::Conv_852': 'up.6.bias',
    # 'initializer': 'initializer.weight'
}

# noinspection DuplicatedCode
renamed_weights = {}

for k, v in onnx_weights.items():
    orig_k = k
    for name, replacement in replacement_dict.items():
        k = k.replace(name, replacement)

    if k == orig_k:
        shape_name = '-'.join([str(x) for x in v.shape])
        replacements = []
        for weight_shape in weight_shapes:
            if shape_name == weight_shape[-1]:
                replacements.append(weight_shape[0])
        if len(replacements) == 1:
            k = replacements[0]

    if k != orig_k:
        renamed_weights[k] = v

state_dict = {k: torch.from_numpy(v) for k, v in renamed_weights.items()}
model.load_state_dict(state_dict)

torch.save(model.state_dict(), 'inswapper_256-newarch.pth')

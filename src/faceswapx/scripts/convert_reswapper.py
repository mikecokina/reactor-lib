import torch
from faceswapx.reswapper.stf_128 import StyleTransferModel

path = "reswapper_256-1399500.pth"
transfer_weights = torch.load(path)

model = StyleTransferModel()
weight_shapes = []
for n, p in model.named_parameters():
    weight_shapes.append((n, '-'.join([str(x) for x in list(p.shape)])))

replacement_dict = {
    'target_encoder': 'down',
    'conv1': 'conv1.conv',
    'style1': 'conv1.adain.style',
    'conv2': 'conv2.conv',
    'style2': 'conv2.adain.style',
    'decoder.0': 'up.1',
    'decoderPart1.0': 'up.4',
    'decoderPart1.2': 'up.6',
    'decoderPart2.0': 'up.8'
}

# noinspection DuplicatedCode
renamed_weights = {}

for k, v in transfer_weights.items():
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

state_dict = {k: torch.tensor(v) for k, v in renamed_weights.items()}
model.load_state_dict(state_dict)


torch.save(model.state_dict(), 'reswapper_256-1399500-newarch.pth')

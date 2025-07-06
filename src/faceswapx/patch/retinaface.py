import inspect

import torchvision.models as models
from facexlib.detection.align_trans import get_reference_facial_points
from facexlib.detection.retinaface import RetinaFace, generate_config

import torch
import warnings

from facexlib.detection.retinaface_net import MobileNetV1, FPN, SSH, make_class_head, make_bbox_head, make_landmark_head
# noinspection PyProtectedMember
from torchvision.models._utils import IntermediateLayerGetter

# Save original __init__ so we can reuse structure
_original_init = RetinaFace.__init__


# Define our patched __init__
def _patched_init(self, network_name='resnet50', half=False, phase='test', device=None):
    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device is None else device
    super(RetinaFace, self).__init__()
    self.half_inference = half
    cfg = generate_config(network_name)
    self.backbone = cfg['name']

    self.model_name = f'retinaface_{network_name}'
    self.cfg = cfg
    self.phase = phase
    self.target_size, self.max_size = 1600, 2150
    self.resize, self.scale, self.scale1 = 1., None, None
    self.mean_tensor = torch.tensor([[[[104.]], [[117.]], [[123.]]]], device=self.device)
    self.reference = get_reference_facial_points(default_square=True)

    # Build network
    if cfg['name'] == 'mobilenet0.25':
        backbone = MobileNetV1()
        self.body = IntermediateLayerGetter(backbone, cfg['return_layers'])
    elif cfg['name'] == 'Resnet50':
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)

            resnet50_sig = inspect.signature(models.resnet50)
            if "weights" in resnet50_sig.parameters:
                # torchvision >= 0.13
                backbone = models.resnet50(weights=None)
            else:
                # torchvision < 0.13
                backbone = models.resnet50(pretrained=False)

        self.body = IntermediateLayerGetter(backbone, cfg['return_layers'])

    in_channels_stage2 = cfg['in_channel']
    in_channels_list = [
        in_channels_stage2 * 2,
        in_channels_stage2 * 4,
        in_channels_stage2 * 8,
    ]

    out_channels = cfg['out_channel']
    self.fpn = FPN(in_channels_list, out_channels)
    self.ssh1 = SSH(out_channels, out_channels)
    self.ssh2 = SSH(out_channels, out_channels)
    self.ssh3 = SSH(out_channels, out_channels)

    self.ClassHead = make_class_head(fpn_num=3, inchannels=cfg['out_channel'])
    self.BboxHead = make_bbox_head(fpn_num=3, inchannels=cfg['out_channel'])
    self.LandmarkHead = make_landmark_head(fpn_num=3, inchannels=cfg['out_channel'])

    self.to(self.device)
    self.eval()
    if self.half_inference:
        self.half()


# Apply the patch
RetinaFace.__init__ = _patched_init

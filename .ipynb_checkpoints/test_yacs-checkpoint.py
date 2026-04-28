import yaml
from yacs.config import CfgNode as CN

_C = CN()
_C.MODEL = CN()
_C.MODEL.DEVICE_ID = '0'
# Add other keys to avoid key errors before DEVICE_ID
_C.MODEL.NAME = 'resnet50'
_C.MODEL.STRIDE_SIZE = [16, 16]
_C.MODEL.NECK = 'bnneck'
_C.MODEL.COS_LAYER = False
_C.MODEL.SIE_CAMERA = False
_C.MODEL.SIE_VIEW = False

try:
    _C.merge_from_file('configs/au/vit_base_au.yaml')
    print("Successfully merged!")
except Exception as e:
    import traceback
    traceback.print_exc()

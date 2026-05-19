from typing import Union, Tuple, Optional, Literal, Dict, Any
import torch

ModalityType = Literal["image", "video", "sequence"]
DeviceType = Union[str, torch.device]
DtypeType = torch.dtype

SpatialShape = Tuple[int, int]  # (H, W)
SpatioTemporalShape = Tuple[int, int, int]  # (T, H, W)
ShapeType = Union[SpatialShape, SpatioTemporalShape]

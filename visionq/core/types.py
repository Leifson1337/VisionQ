from typing import Literal

import torch

ModalityType = Literal["image", "video", "sequence"]
DeviceType = str | torch.device
DtypeType = torch.dtype

SpatialShape = tuple[int, int]
SpatioTemporalShape = tuple[int, int, int]
ShapeType = SpatialShape | SpatioTemporalShape

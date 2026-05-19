import torch

from visionq.models import VisionBackbone


def test_readme_quickstart_smoke():
    x = torch.randn(1, 8, 16)
    model = VisionBackbone(depth=1, dim=16, num_heads=4, backend_override="flash")
    out = model(x)
    assert out.flatten_all().shape == (1, 8, 16)

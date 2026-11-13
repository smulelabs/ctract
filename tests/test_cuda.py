"""Optional device parity; never selects or probes a GPU automatically."""

import os

import pytest
import torch
from ctract_chimera import NEUTRAL, ExactGeometry


@pytest.mark.cuda
@pytest.mark.skipif(os.environ.get("CTRACT_TEST_CUDA") != "1", reason="explicit GPU opt-in required")
def test_geometry_cuda_parity() -> None:
    pose = torch.tensor([NEUTRAL], dtype=torch.float64)
    cpu = ExactGeometry()(pose)
    gpu = ExactGeometry().to("cuda")(pose.to("cuda"))
    for key in cpu:
        torch.testing.assert_close(cpu[key], gpu[key].cpu(), rtol=1e-8, atol=1e-9)

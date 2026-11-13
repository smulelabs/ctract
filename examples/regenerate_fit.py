"""Create the public four-parameter fitting target from CHIMERA's Rust stepper."""

import json
import subprocess
from pathlib import Path

import torch
from ctract_chimera import NEUTRAL


def regenerate() -> None:
    package = Path(__file__).parents[1]
    samples = 12288
    time = torch.arange(samples, dtype=torch.float64) / 96000
    frequencies = torch.arange(1, 25, dtype=torch.float64) * 250
    phase = torch.arange(24, dtype=torch.float64) * 0.41
    # The finite source and its first derivative vanish at the boundaries.
    # Tones land on the FFT grid; no real data or stored model weights are used.
    flow = 0.0001 * torch.sin(torch.pi * torch.arange(samples, dtype=torch.float64) / samples).square()
    flow = flow * torch.sin(2 * torch.pi * time[:, None] * frequencies + phase).sum(-1) / (24**0.5)
    request = {"anatomy": [1.0, 1.0, 2.25, 4.6], "artic": NEUTRAL, "source": flow.tolist(), "fs": 96000.0, "freqs": []}
    result = subprocess.run(  # noqa: S603 -- fixed executable built from committed oracle source; no shell.
        [str(package / "oracle/target/release/ctract-reference-oracle")],
        input=json.dumps(request),
        text=True,
        capture_output=True,
        check=True,
    )
    reference = json.loads(result.stdout)
    row = {
        "artic": NEUTRAL,
        "anatomy": [1.0, 1.0, 2.25, 4.6],
        "source": reference["source"][::6],
        "audio": reference["audio"][::6],
        "output_fs": 16000.0,
        "revision": "eed4a5ff2695",
        "expected": {"jaw": 0.6, "lip_aperture": 0.7, "larynx": 0.0, "flow_gain": 1.0},
    }
    (package / "tests/fixtures/fit.json").write_text(json.dumps(row, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    regenerate()

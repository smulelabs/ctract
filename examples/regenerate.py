"""Regenerate the small stationary synthetic fixture with the Rust oracle.

Run from this package after building oracle/Cargo.toml. No input audio is used.
The saved audio is a public synthetic signal, never a user's media.
"""

import json
import subprocess
from pathlib import Path

import torch
from ctract_chimera import NEUTRAL

PACKAGE = Path(__file__).resolve().parents[1]


def regenerate() -> None:
    """Produce deterministic multisine audio from CHIMERA's actual time stepper."""
    t = torch.arange(12288, dtype=torch.float64) / 96000
    source = (
        1e-3
        * (1 - torch.exp(-t / 0.005))
        * (
            torch.sin(2 * torch.pi * 137 * t)
            + 0.3 * torch.sin(2 * torch.pi * 823 * t)
            + 0.1 * torch.sin(2 * torch.pi * 2173 * t)
        )
    )
    request = {
        "anatomy": [1.0, 1.0, 2.25, 4.6],
        "artic": NEUTRAL,
        "source": source.tolist(),
        "fs": 96000.0,
        "freqs": [100.0, 500.0, 1000.0, 3000.0],
    }
    # Fixed local executable built from the committed Rust source; no shell.
    result = subprocess.run(  # noqa: S603
        [str(PACKAGE / "oracle/target/release/ctract-reference-oracle")],
        input=json.dumps(request),
        text=True,
        capture_output=True,
        check=True,
    )
    oracle = json.loads(result.stdout)
    native = {key: oracle[key] for key in ("source", "audio", "artic", "ph", "orl", "ph_len", "or_len", "velum")}
    native["revision"] = "eed4a5ff2695"
    (PACKAGE / "tests/fixtures/stationary_native.json").write_text(json.dumps(native, separators=(",", ":")) + "\n")
    # Reference audio is point-sampled at 16 kHz. The input multisine contains no
    # components above 8 kHz; residual startup transients are excluded in tests.
    oracle["source"] = oracle["source"][::6]
    oracle["audio"] = oracle["audio"][::6]
    oracle["output_fs"] = 16000.0
    oracle["provenance"] = "CHIMERA eed4a5ff2695; oracle/src/main.rs; ideal-flow, noise disabled"
    (PACKAGE / "tests/fixtures/stationary.json").write_text(json.dumps(oracle, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    regenerate()

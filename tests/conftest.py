"""Keep small CPU tensor tests predictable on many-core research machines."""

import torch


def pytest_sessionstart(session) -> None:
    torch.set_num_threads(2)

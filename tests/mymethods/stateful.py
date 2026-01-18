"""Test fixture module for user-defined functions.

This module provides simple user-defined functions that should be serialized
by source code (not by reference) since 'mymethods' is not in the whitelist.
"""

import torch


def normalize(tensor):
    """Normalize a tensor to unit norm along the last dimension."""
    return torch.nn.functional.normalize(tensor, dim=-1)

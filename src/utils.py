"""Utility functions."""

import math
from typing import Callable, Optional, TypeVar, Union

import torch
from torch import Tensor

T = TypeVar('T')


def default(val: Optional[T], default_val: Union[T, Callable[[], T]]) -> T:
    if val is not None:
        return val
    return default_val() if callable(default_val) else default_val


def extract(a: Tensor, t: Tensor, x_shape: tuple) -> Tensor:
    """Extract values from 1-D tensor `a` at indices `t`, reshaped for broadcasting."""
    batch_size = t.shape[0]
    out = a.gather(-1, t)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

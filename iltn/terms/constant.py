from __future__ import annotations

import torch
import iltn.terms.base as base

class Constant(base.Trajectory):
    def __init__(self, value: torch.Tensor, trainable: bool) -> None:
        """A temporal constant. It is interpreted as a single trajectory.

        Args:
            value (torch.Tensor): A tensor of shape [T,...], where the first axis is for the time 
                    dimension.
            trainable (bool): Whether the constant variables should be trainable.
        """
        self._trainable = trainable
        if self._trainable:
            tensor = torch.nn.Parameter(value.float(), requires_grad=True)
        else:
            tensor = value.clone().detach().float()

        if len(tensor.shape) == 0:
            raise ValueError("t-LTN Constants are trajectories and must have a time axis. "
                             "If you want to create a constant from a static value, use "
                             "`Constant.from_static`.")
        elif len(tensor.shape) == 1:  # Add feature dims
            tensor = tensor.unsqueeze(0)

        free_vars = ["time"]
        super().__init__(tensor, free_vars)

    @classmethod
    def from_static(cls: Constant, value: torch.Tensor, trainable: bool, trace_size: int) -> Constant:
        value = value.unsqueeze(0)
        value = value.repeat(trace_size, 1)
        return cls(value, trainable)

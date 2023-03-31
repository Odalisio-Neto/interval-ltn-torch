from __future__ import annotations
import dataclasses

import tensorflow as tf
import ltn


TimeLabel = str
AxLabel = TimeLabel | ltn.core.VarLabel

class Trajectory(ltn.core.Term):
    def __init__(self, tensor: tf.Tensor, free_vars: list[AxLabel]) -> None:
        super().__init__(tensor, free_vars)
        if "time" not in self.free_vars:
            raise ValueError("Trajectory tensors should have a dimension for `time`.")

    def _copy(self) -> Trajectory:
        return Trajectory(self.tensor, self.free_vars.copy())
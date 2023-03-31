from __future__ import annotations
import abc

import numpy as np
from numpy.typing import ArrayLike
import tensorflow as tf

class Event(abc.ABC):
    def __init__(self, label: str) -> None:
        super().__init__()
        self.label = label

    @abc.abstractmethod
    def mf(self, x: float | ArrayLike) -> float | ArrayLike:
        """Membership function.

        Args:
            x (float | ArrayLike): Input values for which to compute membership values, specified as a scalar or vector.

        Returns:
            float | ArrayLike: Membership value returned as a scalar or a vector. 
                    The dimensions of y match the dimensions of x. Each element of y is the membership value computed for the corresponding element of x.
        """
        pass

    @abc.abstractproperty
    def trainable_variables(self) -> list[tf.Variable]:
        pass


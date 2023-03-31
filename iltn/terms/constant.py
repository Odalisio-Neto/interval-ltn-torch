from __future__ import annotations

import tensorflow as tf
import iltn.terms.base as base

class Constant(base.Trajectory):
    def __init__(self, value: tf.Tensor, trainable: bool) -> None:
        """A temporal constant. It is interepreted as a single trajectory.

        Args:
            value (tf.Tensor): A tensor of shape [T,...], where the first axis is for the time 
                    dimension.
            trainable (bool): Whether the constant variables should be trainable.
        """
        self._trainable = trainable
        if self._trainable:
            tensor = tf.Variable(value, trainable=True, dtype=tf.float32)
        else:
            try:
                tensor = tf.constant(value, dtype=tf.float32)
            except TypeError:
                tensor = tf.convert_to_tensor(tf.cast(value,tf.float32), dtype=tf.float32)
        if len(tensor.shape) == 0:
            raise ValueError("t-LTN Constants are trajectories and must have a time axis. "
                    "If you want to create a constant from a static value, use "
                    "`Constant.from_static`.")
        elif len(tensor.shape) == 1: # add feature dims
            tensor = tensor[tf.newaxis]
        free_vars = ["time"]
        super().__init__(tensor, free_vars)

    @classmethod
    def from_static(cls: Constant, value: tf.Tensor, trainable: bool, trace_size: int) -> None:
        value = tf.expand_dims(value, axis=0)
        value = tf.repeat(value, trace_size, axis=0)
        cls(value, trainable)
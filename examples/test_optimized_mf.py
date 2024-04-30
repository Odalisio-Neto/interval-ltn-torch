import timeit
from typing import Callable

import dataclasses
import tensorflow as tf
import matplotlib.pyplot as plt

@tf.custom_gradient
def zero_with_softplus_grads(x: float, beta: float = 1.) -> float:
    def grad(dy):
        return dy * tf.sigmoid(x*beta)
    return tf.zeros_like(x), grad

def tf_masked_assignment(x: tf.Tensor, mask: tf.Tensor, updates: tf.Tensor) -> tf.Tensor:
    original_shape = tf.shape(x)
    x = tf.reshape(x, -1)
    mask = tf.reshape(mask, -1)
    indices = tf.where(mask)
    res = tf.tensor_scatter_nd_update(x, indices, updates)
    res = tf.reshape(res, original_shape)
    return res

def tf_conditional_vectorized_map(
        map_fn: Callable,
        x: tf.Tensor, 
        where: tf.Tensor
    ) -> tf.Tensor:
    results = map_fn(x[where])
    return tf_masked_assignment(x, where, results)

@dataclasses.dataclass
class TrapzMf:
    a: float
    b: float
    c: float
    d: float

    def v1(self, x: tf.Tensor, beta: float = 1.) -> tf.Tensor:
        return tf.where(
                x<=self.a,
                zero_with_softplus_grads(x-self.a, beta=beta),
                tf.where(
                    x<=self.b,
                    (x-self.a)/tf.maximum(self.b-self.a,1e-9),
                    tf.where(
                        x<=self.c,
                        1.-zero_with_softplus_grads(tf.maximum(self.b-x,x-self.c), beta=beta),
                        tf.where(
                            x<=self.d,
                            (x-self.d)/tf.minimum(self.c-self.d,-1e-9), 
                            zero_with_softplus_grads(self.d-x, beta=beta)
                        )
                    )
                )
            )
    
    def v2(self, x: tf.Tensor, beta: float = 1.) -> tf.Tensor:
        x = tf_conditional_vectorized_map(
                lambda t: zero_with_softplus_grads(t-self.a, beta=beta),
                x, 
                x<=self.a)
        x = tf_conditional_vectorized_map(
                lambda t: (t-self.a)/tf.maximum(self.b-self.a,1e-9),
                x, 
                tf.logical_and(x>self.a,x<=self.b))
        x = tf_conditional_vectorized_map(
                lambda t: 1.-zero_with_softplus_grads(tf.maximum(self.b-t,t-self.c), beta=beta),
                x, 
                tf.logical_and(x>self.b, x<=self.c))
        x = tf_conditional_vectorized_map(
                lambda t: (t-self.d)/tf.minimum(self.c-self.d,-1e-9),
                x,
                tf.logical_and(x>self.c, x<=self.d))
        x = tf_conditional_vectorized_map(
                lambda t: zero_with_softplus_grads(self.d-t, beta=beta),
                x, 
                x>self.d)
        return x

A = TrapzMf(1.,3.,6.,8.)
x = tf.range(0.,1000,0.01)
# fig, ax = plt.subplots(1,1)
# ax.plot(x, A.v2(x))
# plt.show()

print(timeit.timeit(lambda: A.v1(x), number=5))
print(timeit.timeit(lambda: A.v2(x), number=5))
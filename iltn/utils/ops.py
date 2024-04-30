import tensorflow as tf
import numpy as np

def softplus(x: float, beta: float = 1.) -> float:
    return (1/beta) * tf.math.softplus(beta*x)


@tf.custom_gradient
def zero_with_softplus_grads(x: float, beta: float = 1.) -> float:
    def grad(dy):
        return dy * tf.sigmoid(x*beta)
    return 0., grad

# def softplus_inverse(x: float)-> float:
#     return tf.math.log(tf.math.exp(x) - 1.)

def as_numpy_dtype(dtype):
  """Returns a `np.dtype` based on this `dtype`."""
  dtype = tf.as_dtype(dtype)
  if hasattr(dtype, 'as_numpy_dtype'):
    return dtype.as_numpy_dtype
  return dtype

def softplus_inverse(x, name=None):
  """EXCERPT FROM TENSORFLOW PROBABILITIES
  Computes the inverse softplus, i.e., x = softplus_inverse(softplus(x)).
  Mathematically this op is equivalent to:
  ```none
  softplus_inverse = log(exp(x) - 1.)
  ```
  Args:
    x: `Tensor`. Non-negative (not enforced), floating-point.
    name: A name for the operation (optional).
  Returns:
    `Tensor`. Has the same type/shape as input `x`.
  """
  with tf.name_scope(name or 'softplus_inverse'):
    x = tf.convert_to_tensor(x, name='x')
    # We begin by deriving a more numerically stable softplus_inverse:
    # x = softplus(y) = Log[1 + exp{y}], (which means x > 0).
    # ==> exp{x} = 1 + exp{y}                                (1)
    # ==> y = Log[exp{x} - 1]                                (2)
    #       = Log[(exp{x} - 1) / exp{x}] + Log[exp{x}]
    #       = Log[(1 - exp{-x}) / 1] + Log[exp{x}]
    #       = Log[1 - exp{-x}] + x                           (3)
    # (2) is the "obvious" inverse, but (3) is more stable than (2) for large x.
    # For small x (e.g. x = 1e-10), (3) will become -inf since 1 - exp{-x} will
    # be zero. To fix this, we use 1 - exp{-x} approx x for small x > 0.
    #
    # In addition to the numerically stable derivation above, we clamp
    # small/large values to be congruent with the logic in:
    # tensorflow/core/kernels/softplus_op.h
    #
    # Finally, we set the input to one whenever the input is too large or too
    # small. This ensures that no unchosen codepath is +/- inf. This is
    # necessary to ensure the gradient doesn't get NaNs. Recall that the
    # gradient of `where` behaves like `pred*pred_true + (1-pred)*pred_false`
    # thus an `inf` in an unselected path results in `0*inf=nan`. We are careful
    # to overwrite `x` with ones only when we will never actually use this
    # value. Note that we use ones and not zeros since `log(expm1(0.)) = -inf`.
    threshold = np.log(np.finfo(as_numpy_dtype(x.dtype)).eps) + 2.
    is_too_small = x < np.exp(threshold)
    is_too_large = x > -threshold
    too_small_value = tf.math.log(x)
    too_large_value = x
    # This `where` will ultimately be a NOP because we won't select this
    # codepath whenever we used the surrogate `ones_like`.
    x = tf.where(is_too_small | is_too_large, tf.ones([], x.dtype), x)
    y = x + tf.math.log(-tf.math.expm1(-x))  # == log(expm1(x))
    return tf.where(is_too_small,
                    too_small_value,
                    tf.where(is_too_large, too_large_value, y))

@tf.custom_gradient
def norm(x, axis=None, keepdims=None, name=None):
    y = tf.norm(x, axis=None, keepdims=None, name=None)
    def grad(dy):
        return dy * (x / (y + 1e-19))
    return y, grad

def smooth_equal(x: tf.Tensor, y: tf.Tensor, alpha=1.):
    x = tf.expand_dims(x,-1) if tf.rank(x) == 0 else x
    y = tf.expand_dims(y,-1) if tf.rank(y) == 0 else y
    return tf.exp(-alpha*norm(x-y,axis=0))

def smooth_equal_inv(x: tf.Tensor, y: tf.Tensor, alpha:float=1.):
    return 1/(1+alpha*norm(x-y,axis=0))
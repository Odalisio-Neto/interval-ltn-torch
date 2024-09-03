import torch
import numpy as np

def softplus(x: float, beta: float = 1.) -> float:
    return (1/beta) * torch.nn.functional.softplus(beta*x)

def zero_with_softplus_grads(x: float, beta: float = 1.) -> float:
    return 0., torch.sigmoid(x*beta)

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
    threshold = np.log(np.finfo(x.dtype).eps) + 2.
    is_too_small = x < np.exp(threshold)
    is_too_large = x > -threshold
    too_small_value = torch.log(x)
    too_large_value = x
    # This `where` will ultimately be a NOP because we won't select this
    # codepath whenever we used the surrogate `ones_like`.
    x = torch.where(is_too_small | is_too_large, torch.ones_like(x), x)
    y = x + torch.log(-torch.expm1(-x))  # == log(expm1(x))
    return torch.where(is_too_small,
                    too_small_value,
                    torch.where(is_too_large, too_large_value, y))

def norm(x, axis=None, keepdims=None, name=None):
    y = torch.norm(x, dim=axis, keepdim=keepdims)
    def grad(dy):
        return dy * (x / (y + 1e-19))
    return y, grad

def smooth_equal(x: torch.Tensor, y: torch.Tensor, alpha=1.):
    x = x.unsqueeze(-1) if x.dim() == 0 else x
    y = y.unsqueeze(-1) if y.dim() == 0 else y
    return torch.exp(-alpha*torch.norm(x-y,dim=0))

def smooth_equal_inv(x: torch.Tensor, y: torch.Tensor, alpha:float=1.):
    return 1/(1+alpha*torch.norm(x-y,dim=0))
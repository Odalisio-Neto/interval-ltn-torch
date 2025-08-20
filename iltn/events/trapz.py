from __future__ import annotations
import dataclasses
import warnings

import numpy as np
from numpy.typing import ArrayLike
import torch
import torch.nn as nn
import torch.nn.functional as F

from iltn.events.event import Event
from iltn.utils.ops import softplus, softplus_inverse, zero_with_softplus_grads


class LeftInfiniteTrapzEvent(Event):
    def __init__(self, label: str, params: tuple[float, float], trainable: bool = True, beta: float = 1.) -> None:
        """params = [c,d]"""
        super().__init__(label)
        params = [float(param) for param in params]
        self._trainable = trainable
        if trainable:
            self.cp_param = SoftplusParameter(params[0])
            self.dp_param = SoftplusParameter(params[1] - params[0])
        else:
            self._c = params[0]
            self._d = params[1]
        self._optimized_mode = False
        self.beta = beta
    
    @classmethod
    def from_tensors(cls, label: str, params: tuple[torch.Tensor, torch.Tensor], beta: float = 1.) -> LeftInfiniteTrapzEvent:
        dummy_params = [0, 1]
        event = cls(label, dummy_params, trainable=False, beta=beta)
        event._c = params[0]
        event._d = params[1]
        return event

    def mf_map_fn(self, x: float, smooth: bool = True, beta: float = None) -> float:
        beta = self.beta if beta is None else beta
        if x <= self.c:
            return 1. if not smooth else 1. - zero_with_softplus_grads(x - self.c, beta=beta)
        elif x <= self.d:
            return (x - self.d) / (self.c - self.d)
        else:
            return 0. if not smooth else zero_with_softplus_grads(self.d - x, beta=beta)

    def mf_opti(self, x: float | ArrayLike, smooth: bool = True, beta: float = None) -> float:
        """Work in progress"""

        try:
            return torch.tensor([self.mf_map_fn(t, smooth=smooth, beta=beta) for t in x])
        except:
            try:
                return self.mf_map_fn(x, smooth=smooth, beta=beta)
            except:
                return self.mf(x, smooth=smooth, beta=beta)

    def mf(self, x: float | ArrayLike, smooth: bool = True, beta: float = None) -> float | ArrayLike:
        beta = self.beta if beta is None else beta
        if x <= self.c:
            return 1. if not smooth else 1. - zero_with_softplus_grads(x - self.c, beta=beta)
        elif x <= self.d:
            return (x - self.d) / (self.c - self.d)
        else:
            return 0. if not smooth else zero_with_softplus_grads(self.d - x, beta=beta)

    @property
    def trainable_variables(self) -> list[nn.Parameter]:
        if self._trainable:
            return self.cp_param.trainable_variables + self.dp_param.trainable_variables
        else:
            return []

    def start_optimized_step(self, tape: torch.autograd.grad = None) -> None:
        if tape is None:
            warnings.warn("Make sure that a gradient tape is watching the optimization step.")
        self._optimized_mode = True
        if self._trainable:
            self._cp = self.cp_param.eval()
            self._dp = self.dp_param.eval()

    def end_optimized_step(self) -> None:
        self._optimized_mode = False

    @property
    def cp(self) -> float | torch.Tensor:
        if not self._trainable:
            return self.c
        else:
            return self.cp_param.eval() if not self._optimized_mode else self._cp
    
    @property
    def dp(self) -> float | torch.Tensor:
        if not self._trainable:
            return self.d - self.c
        else:
            return self.dp_param.eval() if not self._optimized_mode else self._dp

    @property
    def c(self) -> float | torch.Tensor:
        return self._c if not self._trainable else self.cp

    @property
    def d(self) -> float | torch.Tensor:
        return self._d if not self._trainable else self.cp + self.dp

    @property 
    def area(self) -> float | torch.Tensor:
        return float('inf')
    
    
class RightInfiniteTrapzEvent(Event):
    def __init__(self, label: str, params: tuple[float, float], trainable: bool = True, beta: float = 1.) -> None:
        """params = [a,b]"""
        super().__init__(label)
        params = [float(param) for param in params]
        self._trainable = trainable
        if trainable:
            self.ap_param = SoftplusParameter(params[0])
            self.bp_param = SoftplusParameter(params[1] - params[0])
        else:
            self._a = params[0]
            self._b = params[1]
        self._optimized_mode = False
        self.beta = beta

    @classmethod
    def from_tensors(cls, label: str, params: tuple[torch.Tensor, torch.Tensor], beta: float = 1.) -> RightInfiniteTrapzEvent:
        dummy_params = [0, 1]
        event = cls(label, dummy_params, trainable=False, beta=beta)
        event._a = params[0]
        event._b = params[1]
        return event

    def mf_map_fn(self, x: float, smooth: bool = True, beta: float = None) -> float:
        beta = self.beta if beta is None else beta
        if x <= self.a:
            return 0. if not smooth else zero_with_softplus_grads(x - self.a, beta=beta)
        elif x <= self.b:
            return (x - self.a) / (self.b - self.a)
        else:
            return 1. if not smooth else 1. - zero_with_softplus_grads(self.b - x, beta=beta)
        
    def mf_opti(self, x: float | ArrayLike, smooth: bool = True, beta: float = None) -> float:
        """Work in progress"""
        try:
            return torch.tensor([self.mf_map_fn(t, smooth=smooth, beta=beta) for t in x])
        except:
            try:
                return self.mf_map_fn(x, smooth=smooth, beta=beta)
            except:
                return self.mf(x, smooth=smooth, beta=beta)

    def mf(self, x: float | ArrayLike, smooth: bool = True, beta: float = None) -> float | ArrayLike:
        beta = self.beta if beta is None else beta
        if x <= self.a:
            return 0. if not smooth else zero_with_softplus_grads(x - self.a, beta=beta)
        elif x <= self.b:
            return (x - self.a) / (self.b - self.a)
        else:
            return 1. if not smooth else 1. - zero_with_softplus_grads(self.b - x, beta=beta)

    @property
    def trainable_variables(self) -> list[nn.Parameter]:
        if self._trainable:
            return self.ap_param.trainable_variables + self.bp_param.trainable_variables
        else:
            return []

    def start_optimized_step(self, tape: torch.autograd.grad = None) -> None:
        if tape is None:
            warnings.warn("Make sure that a gradient tape is watching the optimization step.")
        self._optimized_mode = True
        if self._trainable:
            self._ap = self.ap_param.eval()
            self._bp = self.bp_param.eval()

    def end_optimized_step(self) -> None:
        self._optimized_mode = False

    @property
    def ap(self) -> float | torch.Tensor:
        if not self._trainable:
            return self.a
        else:
            return self.ap_param.eval() if not self._optimized_mode else self._ap
    
    @property
    def bp(self) -> float | torch.Tensor:
        if not self._trainable:
            return self.b - self.a
        else:
            return self.bp_param.eval() if not self._optimized_mode else self._bp

    @property
    def a(self) -> float | torch.Tensor:
        return self._a if not self._trainable else self.ap

    @property
    def b(self) -> float | torch.Tensor:
        return self._b if not self._trainable else self.ap + self.bp

    @property 
    def area(self) -> float | torch.Tensor:
        return float('inf')


class TrapzEvent(Event):
    """Finite trapezoidal event"""
    def __init__(self, label: str, params: tuple[float, float, float, float], trainable: bool = False,
                 beta: float = 1.) -> None:
        super().__init__(label=label)
        params = [float(param) for param in params]
        self._trainable = trainable
        if trainable:
            self.ap_param = SoftplusParameter(params[0])
            self.bp_param = SoftplusParameter(params[1] - params[0])
            self.cp_param = SoftplusParameter(params[2] - params[1])
            self.dp_param = SoftplusParameter(params[3] - params[2])
        else:
            self._a, self._b, self._c, self._d = params
        self._optimized_mode = False
        self.beta = beta
    
    @property
    def trainable_variables(self) -> list[torch.nn.Parameter]:
        """Returns the trainable variables of the event."""
        if self._trainable:
            return (self.ap_param.trainable_variables + 
                   self.bp_param.trainable_variables + 
                   self.cp_param.trainable_variables + 
                   self.dp_param.trainable_variables)
        else:
            return []

    @classmethod
    def from_tensors(cls, label: str, params: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], 
                     beta: float = 1.) -> TrapzEvent:
        """ Makes sure that the new event points directly to the input tensors without 
        creating new variables, and that the gradients will flow back to the inputs."""
        dummy_params = [0, 1, 2, 3]
        event = cls(label, dummy_params, trainable=False, beta=beta)
        event._a = params[0]
        event._b = params[1]
        event._c = params[2]
        event._d = params[3]
        return event

    @classmethod
    def from_model(cls, model: "tltnModel") -> None:
        pass  # Implementation would depend on the tltnModel structure

    def mf_map_fn(self, x: float, smooth: bool = True, beta: float = None) -> float:
        beta = self.beta if beta is None else beta
        if x <= self.a:
            return 0. if not smooth else zero_with_softplus_grads(x - self.a, beta=beta)
        elif x <= self.b:
            return (x - self.a) / torch.maximum(self.b - self.a, torch.tensor(1e-9))
        elif x <= self.c:
            return 1.
        elif x <= self.d:
            return (self.d - x) / torch.maximum(self.d - self.c, torch.tensor(1e-9))
        else:
            return 0. if not smooth else zero_with_softplus_grads(self.d - x, beta=beta)

    def mf_opti(self, x: float | ArrayLike, smooth: bool = True, beta: float = None) -> float:
        """Work in progress"""

        try:
            return torch.tensor([self.mf_map_fn(t, smooth=smooth, beta=beta) for t in x])
        except:
            try:
                return self.mf_map_fn(x, smooth=smooth, beta=beta)
            except:
                return self.mf(x, smooth=smooth, beta=beta)

    def mf(self, x: float | ArrayLike, smooth: bool = True, beta: float = None) -> float | ArrayLike:
        beta = self.beta if beta is None else beta
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32)
        elif not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        
        a_val = self.a if isinstance(self.a, torch.Tensor) else torch.tensor(self.a, dtype=torch.float32)
        b_val = self.b if isinstance(self.b, torch.Tensor) else torch.tensor(self.b, dtype=torch.float32)
        c_val = self.c if isinstance(self.c, torch.Tensor) else torch.tensor(self.c, dtype=torch.float32)
        d_val = self.d if isinstance(self.d, torch.Tensor) else torch.tensor(self.d, dtype=torch.float32)
        
        if not smooth:
            cond1 = x <= a_val
            cond2 = x <= b_val
            cond3 = x <= c_val
            cond4 = x <= d_val
            
            res = torch.where(cond1, torch.tensor(0.),
                    torch.where(cond2, (x - a_val) / torch.maximum(b_val - a_val, torch.tensor(1e-9)),
                    torch.where(cond3, torch.tensor(1.),
                    torch.where(cond4, (d_val - x) / torch.maximum(d_val - c_val, torch.tensor(1e-9)),
                    torch.tensor(0.)))))
        else:
            cond1 = x <= a_val
            cond2 = x <= b_val
            cond3 = x <= c_val
            cond4 = x <= d_val
            
            res = torch.where(cond1, zero_with_softplus_grads(x - a_val, beta=beta),
                    torch.where(cond2, (x - a_val) / torch.maximum(b_val - a_val, torch.tensor(1e-9)),
                    torch.where(cond3, torch.tensor(1.),
                    torch.where(cond4, (d_val - x) / torch.maximum(d_val - c_val, torch.tensor(1e-9)),
                    zero_with_softplus_grads(d_val - x, beta=beta)))))
        return res

    def start_optimized_step(self, tape: torch.autograd.grad = None) -> None:
        if tape is None:
            warnings.warn("Make sure that a gradient tape is watching the optimization step.")
        self._optimized_mode = True
        if self._trainable:
            self._ap = self.ap_param.eval()
            self._bp = self.bp_param.eval()
            self._cp = self.cp_param.eval()
            self._dp = self.dp_param.eval()

    def end_optimized_step(self) -> None:
        self._optimized_mode = False

    @property
    def ap(self) -> float | torch.Tensor:
        if not self._trainable:
            return self.a
        else:
            return self.ap_param.eval() if not self._optimized_mode else self._ap
    
    @property
    def bp(self) -> float | torch.Tensor:
        if not self._trainable:
            return self.b - self.a
        else:
            return self.bp_param.eval() if not self._optimized_mode else self._bp

    @property
    def cp(self) -> float | torch.Tensor:
        if not self._trainable:
            return self.c - self.b
        else:
            return self.cp_param.eval() if not self._optimized_mode else self._cp

    @property
    def dp(self) -> float | torch.Tensor:
        if not self._trainable:
            return self.d - self.c
        else:
            return self.dp_param.eval() if not self._optimized_mode else self._dp

    @property
    def a(self) -> float | torch.Tensor:
        return self._a if not self._trainable else self.ap

    @property
    def b(self) -> float | torch.Tensor:
        return self._b if not self._trainable else self.ap + self.bp

    @property
    def c(self) -> float | torch.Tensor:
        return self._c if not self._trainable else self.ap + self.bp + self.cp

    @property
    def d(self) -> float | torch.Tensor:
        return self._d if not self._trainable else self.ap + self.bp + self.cp + self.dp

    @property
    def area(self) -> float | torch.Tensor:
        # Trapz area: left triangle + rectangle + right triangle
        # (b-a)/2 + (c-b) + (d-c)/2 = (b-a + 2(c-b) + d-c)/2 = (d-a + c-b)/2
        return (self.d - self.a + self.c - self.b) / 2



class SoftplusParameter:
    def __init__(self, initial_value: float) -> None:
        float_value = float(initial_value)
        initial_tensor = torch.tensor(float_value + 1e-9, dtype=torch.float32)
        self.logit = torch.nn.Parameter(softplus_inverse(initial_tensor))

    def eval(self) -> torch.Tensor:
        return torch.nn.functional.softplus(self.logit)

    @property
    def trainable_variables(self) -> list[torch.nn.Parameter]:
        return [self.logit]

LeftFiniteTrapezoidalEvent = RightInfiniteTrapzEvent | TrapzEvent
RightFiniteTrapezoidalEvent = LeftInfiniteTrapzEvent | TrapzEvent

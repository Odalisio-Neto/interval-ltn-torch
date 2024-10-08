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
        if not smooth:
            res = torch.where(x <= self.c,
                              torch.tensor(1.),
                              torch.where(x <= self.d,
                                          (x - self.d) / torch.minimum(self.c - self.d, torch.tensor(-1e-9)),
                                          torch.tensor(0.)))
        else:
            res = torch.where(x <= self.c,
                              1. - zero_with_softplus_grads(x - self.c, beta=beta),
                              torch.where(x <= self.d,
                                          (x - self.d) / torch.minimum(self.c - self.d, torch.tensor(-1e-9)),
                                          zero_with_softplus_grads(self.d - x, beta=beta)))
        return res

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
        if not smooth:
            res = torch.where(x <= self.a,
                              torch.tensor(0.),
                              torch.where(x <= self.b,
                                          (x - self.a) / torch.maximum(self.b - self.a, torch.tensor(1e-9)),
                                          torch.tensor(1.)))
        else:
            res = torch.where(x <= self.a,
                              zero_with_softplus_grads(x - self.a, beta=beta),
                              torch.where(x <= self.b,
                                          (x - self.a) / torch.maximum(self.b - self.a, torch.tensor(1e-9)),
                                          1. - zero_with_softplus_grads(self.b - x, beta=beta)))
        return res

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
        if not smooth:
            res = torch.where(x <= self.a,
                              torch.tensor(0.),
                              torch.where(x <= self.b,
                                          (x - self.a) / torch.maximum(self.b - self.a, torch.tensor(1e-9)),
                                          torch.where(x <= self.c,
                                                      torch.tensor(1.),
                                                      torch.where(x <= self.d,
                                                                  (self.d - x) / torch.maximum(self.d - self.c, torch.tensor(1e-9)),
                                                                  torch.tensor(0.)))))
        else:
            res = torch.where(x <= self.a,
                              zero_with_softplus_grads(x - self.a, beta=beta),
                              torch.where(x <= self.b,
                                          (x - self.a) / torch.maximum(self.b - self.a, torch.tensor(1e-9)),
                                          torch.where(x <= self.c,
                                                      torch.tensor(1.),
                                                      torch.where(x <= self.d,
                                                                  (self.d - x) / torch.maximum(self.d - self.c, torch.tensor(1e-9)),
                                                                  zero_with_softplus_grads(self.d - x, beta=beta)))))
        return res

    def start_optimized_step(self, tape: torch.autograd.grad = None) -> None:
        if tape is None:
            warnings.warn("Make sure that a gradient tape is watching the optimization step.")
        self._optimized_mode = True
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



class SoftplusParameter:
    def __init__(self, initial_value: float) -> None:
        self.logit = torch.nn.Parameter(softplus_inverse(initial_value+1e-9))

    def eval(self) -> torch.Tensor:
        return torch.math.softplus(self.logit)

    @property
    def trainable_variables(self) -> list[torch.nn.Parameter]:
        return [self.logit]

LeftFiniteTrapezoidalEvent = RightInfiniteTrapzEvent | TrapzEvent
RightFiniteTrapezoidalEvent = LeftInfiniteTrapzEvent | TrapzEvent

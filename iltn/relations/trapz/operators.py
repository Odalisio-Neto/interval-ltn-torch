from __future__ import annotations

import torch

import iltn
from iltn.events.trapz import LeftFiniteTrapezoidalEvent, RightFiniteTrapezoidalEvent


class Duration:
    def __call__(self, A: iltn.events.TrapzEvent) -> float | torch.Tensor:
        return A.area
    

class Between:
    def __call__(self, A: RightFiniteTrapezoidalEvent, B: LeftFiniteTrapezoidalEvent) -> iltn.events.TrapzEvent:
        return iltn.events.TrapzEvent.from_tensors(
            label=f"between_{A.label}_{B.label}", params=[A.c, A.d, B.a, B.b], beta=A.beta)


class Before:
    def __call__(self, A: LeftFiniteTrapezoidalEvent) -> iltn.events.LeftInfiniteTrapzEvent:
        return iltn.events.LeftInfiniteTrapzEvent.from_tensors(
            label=f"before_{A.label}", params=[A.a, A.b], beta=A.beta)


class After:
    def __call__(self, A: RightFiniteTrapezoidalEvent) -> iltn.events.RightInfiniteTrapzEvent:
        return iltn.events.RightInfiniteTrapzEvent.from_tensors(
            label=f"after_{A.label}", params=[A.c, A.d], beta=A.beta)


class Fuzzify:
    def __call__(self, x: float | torch.Tensor, core: float, support: float, label: str = None, 
                 beta: float = 1.) -> iltn.events.TrapzEvent:
        if support < core:
            raise ValueError("The core parameter 'core' of fuzzify should be"
                             " lower or equal than the support parameter 'support'.")
        label = f"fuzzify_{x}" if label is None else label
        return iltn.events.TrapzEvent.from_tensors(
            label=label, params=[x - support / 2., x - core / 2., x + core / 2., x + support / 2.], beta=beta)


class Start:
    def __init__(self, delta: float = 0.1) -> None:
        self.delta = delta
        self.fuzzify = Fuzzify()

    def __call__(self, A: LeftFiniteTrapezoidalEvent, delta: float = None) -> iltn.events.TrapzEvent:
        delta = self.delta if delta is None else delta
        a_val = A.a if isinstance(A.a, torch.Tensor) else torch.tensor(A.a, dtype=torch.float32)
        b_val = A.b if isinstance(A.b, torch.Tensor) else torch.tensor(A.b, dtype=torch.float32)
        delta_tensor = torch.tensor(delta, dtype=torch.float32)
        support = torch.max(delta_tensor, b_val - a_val)
        return self.fuzzify((a_val + b_val) / 2., core=0, support=support, 
                            label=f"start_{A.label}", beta=A.beta)


class End:
    def __init__(self, delta: float = 0.1) -> None:
        self.delta = delta
        self.fuzzify = Fuzzify()

    def __call__(self, A: RightFiniteTrapezoidalEvent, delta: float = None) -> iltn.events.TrapzEvent:
        delta = self.delta if delta is None else delta
        c_val = A.c if isinstance(A.c, torch.Tensor) else torch.tensor(A.c, dtype=torch.float32)
        d_val = A.d if isinstance(A.d, torch.Tensor) else torch.tensor(A.d, dtype=torch.float32)
        delta_tensor = torch.tensor(delta, dtype=torch.float32)
        support = torch.max(delta_tensor, d_val - c_val)
        return self.fuzzify((c_val + d_val) / 2., core=0, support=support,
                            label=f"end_{A.label}", beta=A.beta)

from __future__ import annotations

import tensorflow as tf

import iltn
from iltn.events.trapz import LeftFiniteTrapezoidalEvent, RightFiniteTrapezoidalEvent


class Duration:
    def __call__(self, A: iltn.events.TrapzEvent) -> float | tf.Tensor:
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
    def __call__(self, x: float | tf.Tensor, core: float, support: float, label: str = None, 
                 beta: float = 1.) -> iltn.events.TrapzEvent:
        if support < core:
            raise ValueError("The core parameter 'core' of fuzzify should be"
                             " lower or equal than the support parameter 'support'.")
        label = f"fuzzify_{x}" if label is None else label
        return iltn.events.TrapzEvent.from_tensors(
            label=label, params=[x-support/2., x-core/2., x+core/2., x+support/2.], beta=beta)


class Start:
    def __init__(self, delta: float = .1) -> None:
        self.delta = delta
        self.fuzzify = Fuzzify()

    def __call__(self, A: LeftFiniteTrapezoidalEvent, delta: float = None) -> iltn.events.TrapzEvent:
        delta = self.delta if delta is None else delta
        return self.fuzzify((A.a+A.b)/2., core=0, support=tf.maximum(delta, A.b-A.a), 
            label=f"start_{A.label}", beta=A.beta)


class End:
    def __init__(self, delta: float = .1) -> None:
        self.delta = delta
        self.fuzzify = Fuzzify()

    def __call__(self, A: RightFiniteTrapezoidalEvent, delta: float = None) -> iltn.events.TrapzEvent:
        delta = self.delta if delta is None else delta
        return self.fuzzify((A.c+A.d)/2., core=0, support=tf.maximum(delta, A.d-A.c),
            label=f"end_{A.label}", beta=A.beta)
    
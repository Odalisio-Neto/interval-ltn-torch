from __future__ import annotations
import ltn
import tensorflow as tf

import iltn
from iltn.utils.ops import softplus, zero_with_softplus_grads
from iltn.relations.trapz.intersection import area_intersection
from iltn.events.trapz import (TrapzEvent, LeftInfiniteTrapzEvent, RightInfiniteTrapzEvent,
    LeftFiniteTrapezoidalEvent, RightFiniteTrapezoidalEvent)
import iltn.relations.trapz.operators as basic_op


class Contains:
    """Contains(A,B): the portion of the area of B that is contained in A."""
    def __init__(self, smooth: bool = True, beta: float = 1.) -> None:
        self.smooth = smooth
        self.beta = beta

    def __call__(self, 
            A: TrapzEvent | LeftInfiniteTrapzEvent | RightInfiniteTrapzEvent, 
            B: TrapzEvent, 
            smooth: bool = None,
            beta: float = None
            ) -> float | tf.Tensor:
        smooth = self.smooth if smooth is None else smooth
        beta = self.beta if beta is None else beta
        # Turn A finite
        if isinstance(A, LeftInfiniteTrapzEvent):
            bound = tf.stop_gradient(B.a)
            A = TrapzEvent.from_tensors(f"finite_{A.label}", [bound, bound, A.c, A.d])
        elif isinstance(A, RightInfiniteTrapzEvent):
            bound = tf.stop_gradient(B.d)
            A = TrapzEvent.from_tensors(f"finite_{A.label}", [A.a, A.b, bound, bound])
        # Discontinuous cases
        # * Empty intersection
        if A.d <= B.a:
            res = 0. if not smooth else zero_with_softplus_grads(A.d-B.a, beta=beta)
        elif B.d <= A.a:
            res = 0. if not smooth else zero_with_softplus_grads(B.d-A.a, beta=beta)
        # * B Fully in A
        elif A.a <= B.a and A.b <= B.b and A.c >= B.c and A.d >= B.d:
            res = 1. if not smooth else 1.-zero_with_softplus_grads(A.a-B.a+B.d-A.d, beta=beta)/2.
        # Continuous case
        else:
            res = area_intersection(A, B)/B.area
        return res


class Equals:
    def __init__(self, smooth: bool = True, beta: float = 1.) -> None:
        self.smooth = smooth
        self.beta = beta
        self.fuzzy_and = ltn.fuzzy_ops.And_Prod()
        self.contains = Contains(smooth=smooth,beta=beta)

    def __call__(self, A: TrapzEvent, B: TrapzEvent, smooth: bool = None, beta: float = None
            ) -> float | tf.Tensor:
        smooth = self.smooth if smooth is None else smooth
        beta = self.beta if beta is None else beta
        return self.fuzzy_and(self.contains(A,B), self.contains(B,A))


class Before:
    def __init__(self, op_before: basic_op.Before = None, contains: Contains = None, 
                 smooth: bool = True) -> None:
        self.smooth = smooth
        self.op_before = basic_op.Before() if op_before is None else op_before
        self.contains = Contains() if contains is None else contains

    def __call__(self, A: TrapzEvent, B: LeftFiniteTrapezoidalEvent, smooth: bool = None
            ) -> float | tf.Tensor:
        smooth = self.smooth if smooth is None else smooth
        return self.contains(self.op_before(B), A, smooth=smooth)
    

class After:
    def __init__(self, op_after: basic_op.After = None, contains: Contains = None, 
                 smooth: bool = True) -> None:
        self.smooth = smooth
        self.op_after = basic_op.After() if op_after is None else op_after
        self.contains = Contains() if contains is None else contains

    def __call__(self, A: TrapzEvent, B: RightFiniteTrapezoidalEvent, smooth: bool = None
            ) -> float | tf.Tensor:
        smooth = self.smooth if smooth is None else smooth
        return self.contains(self.op_after(B), A, smooth=smooth)


class Starts:
    def __init__(self, op_start: basic_op.Start = None, op_end: basic_op.End = None,
            equals: Equals = None, before: Before = None, fuzzy_and: ltn.fuzzy_ops.And_Prod = None,
            smooth: bool = True) -> None:
        self.smooth = smooth
        self.op_start = basic_op.Start() if op_start is None else op_start
        self.op_end = basic_op.End() if op_end is None else op_end
        self.equals = Equals() if equals is None else equals
        self.before = Before() if before is None else before
        self.fuzzy_and = ltn.fuzzy_ops.And_Prod() if fuzzy_and is None else fuzzy_and

    def __call__(self, A: TrapzEvent, B: LeftFiniteTrapezoidalEvent, smooth: bool = None
            ) -> float | tf.Tensor:
        smooth = self.smooth if smooth is None else smooth
        if isinstance(B, RightInfiniteTrapzEvent):
            return self.equals(self.op_start(A),self.op_start(B),smooth=smooth)
        else:
            return self.fuzzy_and(self.equals(self.op_start(A),self.op_start(B),smooth=smooth), 
                              self.before(self.op_end(A), self.op_end(B),smooth=smooth))


class During:
    def __init__(self, op_start: basic_op.Start = None, op_end: basic_op.End = None,
            equals: Equals = None, before: Before = None, after: After = None,
            fuzzy_and: ltn.fuzzy_ops.And_Prod = None,
            smooth: bool = True) -> None:
        self.smooth = smooth
        self.op_start = basic_op.Start() if op_start is None else op_start
        self.op_end = basic_op.End() if op_end is None else op_end
        self.equals = Equals() if equals is None else equals
        self.before = Before() if before is None else before
        self.after = After() if after is None else after
        self.fuzzy_and = ltn.fuzzy_ops.And_Prod() if fuzzy_and is None else fuzzy_and

    def __call__(self, A: TrapzEvent, B: LeftFiniteTrapezoidalEvent | RightFiniteTrapezoidalEvent, 
                smooth: bool = None) -> float | tf.Tensor:
        smooth = self.smooth if smooth is None else smooth        
        if isinstance(B, RightInfiniteTrapzEvent):
            return self.after(self.op_start(A),self.op_start(B),smooth=smooth)
        elif isinstance(B, LeftInfiniteTrapzEvent):
            return self.before(self.op_end(A), self.op_end(B),smooth=smooth)
        else:
            return self.fuzzy_and(
                self.after(self.op_start(A),self.op_start(B),smooth=smooth),
                self.before(self.op_end(A), self.op_end(B),smooth=smooth)
            )


class Overlaps:
    def __init__(self, op_start: basic_op.Start = None, op_end: basic_op.End = None,
            equals: Equals = None, before: Before = None, 
            fuzzy_and_aggreg: ltn.fuzzy_ops.Aggreg_Prod = None,
            smooth: bool = True) -> None:
        self.smooth = smooth
        self.op_start = basic_op.Start() if op_start is None else op_start
        self.op_end = basic_op.End() if op_end is None else op_end
        self.equals = Equals() if equals is None else equals
        self.before = Before() if before is None else before
        self.fuzzy_and_aggreg = ltn.fuzzy_ops.Aggreg_Prod() if fuzzy_and_aggreg is None else fuzzy_and_aggreg

    def __call__(self, A: RightFiniteTrapezoidalEvent, B: LeftFiniteTrapezoidalEvent, 
                smooth: bool = None) -> float | tf.Tensor:
        smooth = self.smooth if smooth is None else smooth        
        if isinstance(A, LeftInfiniteTrapzEvent) and isinstance(B,RightInfiniteTrapzEvent):
            return self.before(self.op_start(B),self.op_end(A),smooth=smooth)
        elif isinstance(A, LeftInfiniteTrapzEvent):
            return self.fuzzy_and_aggreg([
                self.before(self.op_start(B),self.op_end(A),smooth=smooth),
                self.before(self.op_end(A),self.op_end(B),smooth=smooth)
            ])
        elif isinstance(B, RightInfiniteTrapzEvent):
            return self.fuzzy_and_aggreg([
                self.before(self.op_start(A),self.op_start(B),smooth=smooth),
                self.before(self.op_start(B),self.op_end(A),smooth=smooth)
            ])
        else:
            return self.fuzzy_and_aggreg([
                self.before(self.op_start(A),self.op_start(B),smooth=smooth),
                self.before(self.op_start(B),self.op_end(A),smooth=smooth),
                self.before(self.op_end(A),self.op_end(B),smooth=smooth)
            ])


class BeforeAreaIntersection:
    def __init__(self, smooth: bool = True, beta1: float = 0.75, beta2: float = 2.) -> None:
        """Implementation of Before(A,B) using another formula that doesn't depend 
        on the duration of A or B.

        Args:
            smooth (bool, optional): Whether to use a softplus activation to smoothen the 
                zero values and avoid vanishing gradients. Defaults to True. If False,
                the parameters `beta1` and `beta2` are not used.
            beta1 (float, optional): Parameter of the smooth zero when B is before A. 
                We recommend a value lower or equal to 1. Defaults to 0.75.
            beta2 (float, optional): Parameter of the smooth zero when A is before B. 
                We recommend a value higher or equal to 1. Defaults to 2.
        """
        super().__init__()
        self.smooth = smooth
        self.beta1 = beta1
        self.beta2 = beta2

    def __call__(
            self, 
            A: TrapzEvent, 
            B: TrapzEvent,
            smooth: bool = None
            ) -> float:
        smooth = self.smooth if smooth is None else smooth
        # TODO: with terms instead of events, and free vars
        if B.b <= A.c:
            res = softplus(B.b-A.c, self.beta1) if (smooth) else 0
        elif A.d <= B.a:
            res = 1. - softplus(A.d-B.a, self.beta2) if (smooth) else 1
        else:
            res = (B.b-A.c)**2/((A.d-B.a)**2 + (B.b-A.c)**2)
        return res

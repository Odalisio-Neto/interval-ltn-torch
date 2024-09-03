from __future__ import annotations
import dataclasses

import torch

from iltn.events.trapz import TrapzEvent, LeftInfiniteTrapzEvent, RightInfiniteTrapzEvent 

@dataclasses.dataclass
class Point:
    x: float | torch.Tensor
    y: float | torch.Tensor

@dataclasses.dataclass
class EdgeLine:
    """A line represented with x= inverse_slope * y + xintercept"""
    inv_slope: float | torch.Tensor
    xintercept: float | torch.Tensor

    def is_vertical(self) -> bool:
        return self.inv_slope == 0
    
    def is_parallel(self, other: EdgeLine) -> bool:
        return self.inv_slope == other.inv_slope

    def get_x(self, y: float | torch.Tensor) -> float | torch.Tensor:
        return self.inv_slope * y + self.xintercept

    def get_y(self, x: float | torch.Tensor) -> float | torch.Tensor:
        if self.is_vertical():
            raise ValueError("Cannot calculate y for a vertical line.")
        return (x - self.xintercept) / self.inv_slope

    def intersect(self, other: EdgeLine) -> list[Point]:
        if self.is_parallel(other):
            return []
        if self.is_vertical():
            x = self.xintercept
            y = other.get_y(x)
        elif other.is_vertical():
            x = other.xintercept
            y = self.get_y(x)
        else:
            denominator = (self.inv_slope - other.inv_slope)
            y = (other.xintercept - self.xintercept) / denominator
            x = (self.inv_slope * other.xintercept - other.inv_slope * self.xintercept) / denominator
        return [Point(x, y)]

def turn_finite(
        A: LeftInfiniteTrapzEvent | RightInfiniteTrapzEvent, 
        using_bounds_of_B: TrapzEvent
    ) -> TrapzEvent:
    if isinstance(A, LeftInfiniteTrapzEvent):
        bound = using_bounds_of_B.a.detach()
        finite_A = TrapzEvent.from_tensors(f"finite_{A.label}", [bound, bound, A.c, A.d])
    elif isinstance(A, RightInfiniteTrapzEvent):
        bound = using_bounds_of_B.d.detach()
        finite_A = TrapzEvent.from_tensors(f"finite_{A.label}", [A.a, A.b, bound, bound])
    else:
        raise ValueError("Event is already finite.")
    return finite_A

def area_intersection(
        A: TrapzEvent | LeftInfiniteTrapzEvent | RightInfiniteTrapzEvent, 
        B: TrapzEvent | LeftInfiniteTrapzEvent | RightInfiniteTrapzEvent
    ) -> float | torch.Tensor:
    if ((isinstance(A, LeftInfiniteTrapzEvent) and isinstance(B, LeftInfiniteTrapzEvent)) 
            or (isinstance(A, RightInfiniteTrapzEvent) and isinstance(B, RightInfiniteTrapzEvent))):
        raise ValueError("Cannot calculate area intersection for events that share the same-side "
                         "infinity.")
    if isinstance(A, (LeftInfiniteTrapzEvent, RightInfiniteTrapzEvent)):
        turn_finite(A, B)
    if isinstance(B, (LeftInfiniteTrapzEvent, RightInfiniteTrapzEvent)):
        turn_finite(B, A)
    vertices = find_intersection_vertices(A, B)
    if len(vertices) <= 2:
        area = 0.
    else:
        area = shoelace_formula(vertices)
    return area

def shoelace_formula(vertices: list[Point]) -> float | torch.Tensor:
    """
    Args:
        vertices (list[Point]): Must be in counter-clockwise order
    """
    xs = torch.stack([v.x for v in vertices])
    ys = torch.stack([v.y for v in vertices])
    s1 = torch.sum(xs * torch.roll(ys, -1, dims=0))
    s2 = torch.sum(ys * torch.roll(xs, -1, dims=0))
    area = 0.5 * torch.abs(s1 - s2)
    return area

def find_intersection_vertices(A: TrapzEvent, B: TrapzEvent) -> list[tuple[float, float]]:
    """Return a list of vertices delimiting A inter B, 
    in a counter-clockwise order, starting from the bottom left."""
    # Make sure A is left of B
    if A.a > B.a:
        A, B = B, A
    # Empty intersection
    if A.d <= B.a:
        return []
    # Bottom vertices
    bottom_vertices = [Point(B.a, 0.), Point(torch.min(A.d, B.d), 0.)]
    # Top vertices
    if A.c < B.b or A.b > B.c:
        top_vertices = []
    elif A.b == B.c:
        top_vertices = [Point((A.b + B.c) / 2, 1.)]
    elif A.c == B.b:
        top_vertices = [Point((A.c + B.b) / 2, 1.)]
    else:
        top_vertices = [Point(torch.min(A.c, B.c), 1.), Point(torch.max(A.b, B.b), 1.)] # order matters
    # side vertices
    left_A = EdgeLine(A.b - A.a, A.a)
    right_A = EdgeLine(A.c - A.d, A.d)
    left_B = EdgeLine(B.b - B.a, B.a)
    right_B = EdgeLine(B.c - B.d, B.d)
    imprecision_threshold = 1e-8
    left_side_vertices = [point for point in left_A.intersect(right_B) + left_A.intersect(left_B)
                          if point.y < 1 - imprecision_threshold and point.y > 0 + imprecision_threshold] # order matters
    right_side_vertices = [point for point in right_A.intersect(right_B) + right_A.intersect(left_B)
                           if point.y < 1 - imprecision_threshold and point.y > 0 + imprecision_threshold] # order matters
    return bottom_vertices + right_side_vertices + top_vertices + left_side_vertices

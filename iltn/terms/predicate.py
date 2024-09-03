from __future__ import annotations
import torch
import iltn.terms.base as base
import iltn.terms.variable as var
import iltn.terms.constant as const
import iltn.relations.trapz.operators as ops

class Predicate:
    def __init__(self, func: callable, smooth: bool = True) -> None:
        self.func = func
        self.smooth = smooth

    def evaluate(self, *args, **kwargs) -> torch.Tensor:
        """Evaluate the predicate."""
        return self.func(*args, **kwargs)

    def evaluate_with_constant(self, constant: const.Constant, *args, **kwargs) -> torch.Tensor:
        """Evaluate the predicate using a constant."""
        return self.func(constant.tensor, *args, **kwargs)

    def evaluate_with_variable(self, variable: var.Variable, *args, **kwargs) -> torch.Tensor:
        """Evaluate the predicate using a variable."""
        return self.func(variable.evaluate(), *args, **kwargs)

    def _copy(self) -> Predicate:
        return Predicate(self.func, self.smooth)

# Trying to define a temporal predicate
class TemporalPredicate(Predicate):
    def __init__(self, func: callable, smooth: bool = True) -> None:
        super().__init__(func, smooth)

    def evaluate_over_time(self, tensor: torch.Tensor) -> torch.Tensor:
        """Evaluate the predicate over a time-trajectory."""
        results = []
        for t in range(tensor.shape[0]):
            results.append(self.func(tensor[t]))
        return torch.stack(results)

    def evaluate_over_constant_time(self, constant: const.Constant) -> torch.Tensor:
        """Evaluate the predicate at a specific time using a constant."""
        return self.func(constant.tensor[0])

    def evaluate_over_variable_time(self, tensor: torch.Tensor, variable: var.Variable) -> torch.Tensor:
        """Evaluate the predicate over a specific time slice determined by a variable."""
        time_value = int(variable.evaluate().item())  # Assuming the variable represents an integer time step
        return self.func(tensor[time_value])

# Usage with a simplest function i could imagine...
class EqualsPredicate(TemporalPredicate):
    def __init__(self, smooth: bool = True) -> None:
        def equals_func(tensor: torch.Tensor) -> torch.Tensor:
            return ops.smooth_equal(tensor, torch.ones_like(tensor))  # 
        super().__init__(equals_func, smooth)

from __future__ import annotations
import torch
import iltn.terms.base as base
import iltn.terms.variable as var
import iltn.terms.constant as const

class Function(base.Trajectory):
    def __init__(self, tensor: torch.Tensor, free_vars: list[base.AxLabel], func: callable) -> None:
        super().__init__(tensor, free_vars)
        self.func = func

    def apply(self, *args, **kwargs) -> torch.Tensor:
        """Apply the function to the tensor."""
        return self.func(self.tensor, *args, **kwargs)

    def apply_to_constant(self, constant: const.Constant) -> torch.Tensor:
        """Apply the function using the value of a Constant."""
        return self.func(constant.tensor)

    def apply_to_variable(self, variable: var.Variable) -> torch.Tensor:
        """Apply the function using the value of a Variable."""
        return self.func(variable.evaluate())

    def _copy(self) -> Function:
        return Function(self.tensor, self.free_vars.copy(), self.func)

# Example of a specialized function
class TemporalFunction(Function):
    def __init__(self, tensor: torch.Tensor, free_vars: list[base.AxLabel], func: callable) -> None:
        super().__init__(tensor, free_vars, func)

    def apply_to_timestep(self, timestep: int) -> torch.Tensor:
        """Apply the function to a specific time slice of the tensor."""
        return self.func(self.tensor[timestep])

    def apply_to_constant_time(self, constant: const.Constant) -> torch.Tensor:
        """Apply the function using the time value from a Constant."""
        return self.apply_to_time(0)  # Assuming the constant is time-invariant

    def apply_to_variable_time(self, variable: var.Variable) -> torch.Tensor:
        """Apply the function using the time value from a Variable."""
        time_value = int(variable.evaluate().item())  # Assuming the variable represents an integer time step
        return self.apply_to_time(time_value)

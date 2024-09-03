from __future__ import annotations
import torch
import iltn.terms.base as base

class Variable(base.Trajectory):
    def __init__(self, name: base.AxLabel, tensor: torch.Tensor) -> None:
        """
        Initialize a variable in the t-LTN framework.

        Args:
            name (AxLabel): The label/name of the variable (e.g., 'time', 'x', etc.).
            tensor (torch.Tensor): The tensor representing the variable's data.
        """
        super().__init__(tensor, [name])
        self.name = name

    def _copy(self) -> Variable:
        return Variable(self.name, self.tensor.clone())

    def assign(self, new_value: torch.Tensor) -> None:
        """
        Assign a new value to the variable.

        Args:
            new_value (torch.Tensor): The new tensor to assign.
        """
        if new_value.shape != self.tensor.shape:
            raise ValueError("The shape of the new value must match the variable's shape.")
        self.tensor = new_value.clone()

    def evaluate(self) -> torch.Tensor:
        """
        Return the current value of the variable.
        
        Returns:
            torch.Tensor: The tensor value of the variable.
        """
        return self.tensor

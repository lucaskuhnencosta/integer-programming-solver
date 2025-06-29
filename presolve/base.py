from abc import ABC, abstractmethod
from typing import List

class Reduction:
    """
    Represents a change to apply to the MIPInstance: fixing a variable, tightening bounds, removing a constraint, etc.
    """
    def __init__(self, kind: str, target: str, value):
        self.kind = kind  # e.g., 'fix_variable', 'remove_constraint', 'tighten_bound'
        self.target = target  # variable or constraint name
        self.value = value  # value to fix to, or bounds tuple

    def __repr__(self):
        return f"Reduction(kind={self.kind}, target={self.target}, value={self.value})"


class Presolver(ABC):
    """
    Abstract base class for presolvers.
    """

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def apply(self, instance) -> List[Reduction]:
        """
        Analyze the instance and return a list of reductions.
        Do not modify the instance directly.
        """
        pass

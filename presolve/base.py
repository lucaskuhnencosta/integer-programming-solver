from abc import ABC, abstractmethod
from typing import List

class Reduction:
    def __init__(self, kind: str, target: str, value):
        self.kind = kind
        self.target = target
        self.value = value

    def __repr__(self):
        return f"Reduction(kind={self.kind}, target={self.target}, value={self.value})"



class Presolver(ABC):
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def apply(self, instance) -> List[Reduction]:
        pass
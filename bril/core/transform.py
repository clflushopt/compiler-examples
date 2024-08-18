"""
Definition of a code transformation that can be applied to functions
for intra-procedural optimizations.
"""

from abc import abstractmethod, ABC
from bril.core.ir import Function


class Transform(ABC):
    """
    A transform is any class that implements the `run` method to process
    Bril functions.
    """

    @abstractmethod
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def run(self, function: Function):
        pass

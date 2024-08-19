"""
Definition of a code transformation that can be applied to functions
for intra-procedural optimizations.
"""

from abc import ABC, abstractmethod

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


class Identity(Transform):
    """
    Identity transformation is a no op pass.
    """

    def __init__(self):
        super().__init__("identity")

    def run(self, _: Function):
        pass

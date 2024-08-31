"""
Definition of a code transformation that can be applied to functions
for intra-procedural optimizations.
"""

from abc import ABC, abstractmethod

from bril.core.ir import Function

# Name assigned to the dead code elimination transform.
DEAD_CODE_ELIMINATION: str = "dce"
# Name assigned to the global dead code elimination transform.
GLOBAL_DEAD_CODE_ELIMINATION: str = "global-dce"
# Name assigned to the redundant store elimination transform.
REDUNDANT_STORE_ELIMINATION: str = "rse"
# Name assigned to local value numbering.
LOCAL_VALUE_NUMBERING: str = "lvn"

# Whether or not to enable debug mode when running optimizations to display
# verbose information.
ENABLE_OPTIMIZATION_DEBUG_MODE: bool = False


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

"""
Implemention of an abstract `Analysis` class used to represent a program
analysis implementation such as `Liveness` or `ReachingDefinitions`.
"""

from abc import ABC, abstractmethod

from bril.core.ir import Function

# Name assigned to the liveness analysis pass.
LIVENESS_ANALYSIS: str = "liveness"

# Enable analyses debug mode which prints extra meta-information during an
# analysis pass.
ENABLE_ANALYSIS_DEBUG_MODE: bool = True


class Analysis(ABC):
    """
    An analysis pass is any class that implements a `run` method to run custom
    program analysis passes.
    """

    @abstractmethod
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def run(self, function: Function):
        pass


class Identity(Analysis):
    """
    Identity analysis is a no op pass.
    """

    def __init__(self):
        super().__init__("identity")

    def run(self, _: Function):
        pass


"""
Implementation of liveness analysis.
"""

from typing import Dict

from bril.core.analyses import Analysis
from bril.core.cfg import ControlFlowGraph
from bril.core.ir import BasicBlock, Function


class Liveness(Analysis):
    name: str = "liveness"
    # Set of upward-exposed variables in a block.
    uevar: Dict[str, set[str]] = {}
    # Set of liveout variables in a block.
    liveout: Dict[str, set[str]] = {}
    # Set of all variables defined in a block.
    varkill: Dict[str, set[str]] = {}
    # Set of all variables in the function being analysed.
    variables: set[str] = set()

    def __init__(self):
        super().__init__(self.name)

    def run(self, function: Function):
        # Build the function's control flow graph.
        cfg: ControlFlowGraph = ControlFlowGraph(function)
        # Worklist of blocks to work on.
        for block in cfg.basic_blocks:
            self.gather(block)
        self.solve(cfg)
        print("Finished computing live out set.")

        print("Liveout set: ")
        for label, liveout in self.liveout.items():
            print(f"Block {label} liveout set is {liveout}")

        pass

    def gather(self, block: BasicBlock):
        """
        Gather initial information to populate the liveness analysis state.
        """
        self.uevar[block.label] = set()
        self.liveout[block.label] = set()
        self.varkill[block.label] = set()

        # Local liveness analysis state for this block.
        uevar: set[str] = set()
        liveout: set[str] = set()
        varkill: set[str] = set()

        for instr in block.instructions:
            # If the arguments belong to the varkill set
            # then they were defined upwards so add them to uevar.
            if instr.get_args() is not None:
                for arg in instr.get_args():
                    uevar.add(arg)
            # Add the destination  to the varkill set.
            if instr.get_dest() is not None:
                varkill.add(instr.get_dest())
                # All variables must be initialized somewhere at the IR level.
                self.variables.add(instr.get_dest())

        # Populate the shared state.
        self.uevar[block.label] = uevar
        self.liveout[block.label] = liveout
        self.varkill[block.label] = varkill

    def complement(self, label: str) -> set[str]:
        """
        Compute the complement of the varkill set of a basic block.
        """
        complement = self.variables.difference(self.varkill[label])
        return complement

    def compute(self, block: BasicBlock):
        """
        Compute the liveout set for a given basic block.

        - `U`: set union.
        - `SUCC`: set of successors for a node.
        - `C(..)`: complement of the set, in this case the set of all
            variables not defined in a block.

        The equation itself, the set of liveout variables of a node `n`:

        LIVEOUT(n) = UNION(UEVAR(m) U (LIVEOUT(m) U C(VARKILL(m)))) for m in SUCC(n)
        """
        liveout_n: set[str] = set()

        for m in block.successors:
            uevar_m = self.uevar[m.label]
            liveout_m = self.liveout[m.label]
            complement_varkill_m = self.complement(m.label)
            liveout_n = liveout_n.union(
                uevar_m.union(liveout_m.union(complement_varkill_m))
            )
        self.liveout[block.label] = liveout_n

    def solve(self, cfg: ControlFlowGraph):
        """
        Iteratively solve the data-flow equation defined as for a basic block
        or cfg node `n` from Cooper & Torczon Chapter 8.

        for block in cfg.blocks:
            liveout[block.label] = ()

        changed := true
        while changed:
            changed = false
            for block in cfg.blocks:
                compute_liveout(blcok)
                if liveout(block) changed:
                    changed = true
        """
        for block in cfg.basic_blocks:
            # Initialize the liveout for each block as the empty set.
            self.liveout[block.label] = set()

        # Iteration stops when `changed` is no longer true.
        changed: bool = True

        while changed:
            changed = False
            for block in cfg.basic_blocks:
                old_liveout = self.liveout[block.label]
                # Recompute LIVEOUT(i)
                self.compute(block)
                new_liveout = self.liveout[block.label]
                # Compute the set difference, if it's not empty then the blocks
                # have not converged yet.
                set_diff = new_liveout.difference(old_liveout)
                if len(set_diff) != 0:
                    changed = True

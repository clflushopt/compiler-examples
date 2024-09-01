"""
Implemention of an abstract `Analysis` class used to represent a program
analysis implementation such as `Liveness` or `ReachingDefinitions`.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Set

from bril.core.ir import Function, Instruction, OPCode

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
    """
    Liveness analysis is a backward analysis that computes for each block
    the set of live variables when we exist the block.

    Variables are considered live at a point p in a program if there exist
    a path from p to a use of the variable along the control flow graph.
    """

    name: str = "liveness"
    # Set of upward-exposed variables in a block.
    uevar: Dict[str, Set[str]] = {}
    # Set of liveout variables in a block.
    liveout: Dict[str, Set[str]] = {}
    # Set of all variables defined in a block.
    varkill: Dict[str, Set[str]] = {}
    # Set of all variables in the function being analysed.
    variables: Set[str] = set()

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
        uevar: Set[str] = set()
        liveout: Set[str] = set()
        varkill: Set[str] = set()

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

    def complement(self, label: str) -> Set[str]:
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
        liveout_n: Set[str] = set()

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
                old_liveout = self.liveout[block.label].copy()
                # Recompute LIVEOUT(i)
                self.compute(block)
                new_liveout = self.liveout[block.label]
                # Compute the set difference, if it's not empty then the blocks
                # have not converged yet.
                set_diff = new_liveout.difference(old_liveout)
                if len(set_diff) != 0:
                    changed = True


@dataclass
class Expression:
    instr: Instruction

    def __hash__(self):
        return hash((self.instr.op, tuple(self.instr.get_args())))

    def __eq__(self, other):
        if not isinstance(other, Expression):
            return False
        return (
            self.instr.op == other.instr.op
            and self.instr.get_args() == other.instr.get_args()
        )

    def __str__(self):
        return f"{self.instr.op.name} {self.instr.get_args()[0] if self.instr.get_args() is not None else ""} {self.instr.get_args()[1] if self.instr.get_args() is not None and self.instr.get_args() > 0 else ""}"


class AvailableExpressions(Analysis):
    """
    Available Expressions analysis is a forward data flow analysis that determines
    which expressions are already computed and still valid at a given program point.
    An expression is "available" if:
    - It has been computed on some path to the current program point.
    - The variables used in the expression haven't been redefined since its last computation.
    """

    name: str = "available_expressions"
    # Expressions generated in each block
    gen: Dict[str, Set[Expression]] = {}
    # Expressions killed in each block
    kill: Dict[str, Set[Expression]] = {}
    # Available expressions at the end of each block
    avail_out: Dict[str, Set[Expression]] = {}
    # All expressions in the function
    all_expressions: Set[Expression] = set()

    def __init__(self):
        pass

    def run(self, function):
        cfg = ControlFlowGraph(function)
        for block in cfg.basic_blocks:
            self.gather(block)
        self.solve(cfg)
        print("Finished computing available expressions.")

        print("Available expressions at block exits:")
        for label, exprs in self.avail_out.items():
            print(f"Block {label}: {exprs}")

    def gather(self, block: BasicBlock):
        self.gen[block.label] = set()
        self.kill[block.label] = set()

        for instr in block.instructions:
            # If this a computed expression build an expression object
            # out of it.
            if instr.op in [OPCode.ADD, OPCode.SUB, OPCode.MUL, OPCode.DIV, OPCode.LAND, OPCode.LOR]:
                expr = Expression(instr)
                self.all_expressions.add(expr)
                self.gen[block.label].add(expr)
                # Kill all expressions using the destination variable
                self.kill[block.label] |= {
                    e for e in self.all_expressions if instr.get_dest() in e.instr.get_args()
                }
            else:
                # For non-arithmetic instructions, kill expressions using the destination
                if instr.get_dest():
                    self.kill[block.label] |= {e for e in self.all_expressions if instr.get_dest() in e.instr.get_args()}

    def compute(self, block: BasicBlock):
        """
        Compute available expressions at the exit of a block.
        AVAIL_OUT(n) = GEN(n) ∪ (AVAIL_IN(n) - KILL(n))
        where AVAIL_IN(n) = ∩ AVAIL_OUT(p) for all predecessors p of n
        """
        if not block.predecessors:
            avail_in = set()
        else:
            avail_in = set.intersection(
                *(self.avail_out[p.label] for p in block.predecessors)
            )

        self.avail_out[block.label] = self.gen[block.label].union(
            avail_in .difference(self.kill[block.label])
        )

    def solve(self, cfg: ControlFlowGraph):
        # Initialize AVAIL_OUT for all blocks to the set of all expressions
        for block in cfg.basic_blocks:
            self.avail_out[block.label] = self.all_expressions.copy()

        changed = True
        while changed:
            changed = False
            for block in cfg.basic_blocks:
                old_avail_out = self.avail_out[block.label].copy()
                self.compute(block)
                if self.avail_out[block.label] != old_avail_out:
                    changed = True

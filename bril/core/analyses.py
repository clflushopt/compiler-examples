"""
Implemention of an abstract `Analysis` class used to represent a program
analysis implementation such as `Liveness` or `ReachingDefinitions`.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Set

from bril.core.ir import Function, Instruction, OPCode

# Name assigned to the liveness analysis pass.
LIVENESS_ANALYSIS: str = "liveness"
# Name assigned to the available expressions pass.
AVAILABLE_EXPRESSIONS: str = "available-expressions"
# Name assigned to the reaching definitions pass.
REACHING_DEFINITIONS: str = "reaching-definitions"

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

    def __init__(self):
        super().__init__(LIVENESS_ANALYSIS)
        self.name: str = LIVENESS_ANALYSIS
        # Set of upward-exposed variables in a block.
        self.uevar: Dict[str, Set[str]] = {}
        # Set of liveout variables in a block.
        self.liveout: Dict[str, Set[str]] = {}
        # Set of all variables defined in a block.
        self.varkill: Dict[str, Set[str]] = {}
        # Set of all variables in the function being analysed.
        self.variables: Set[str] = set()

    def run(self, function: Function):
        # Build the function's control flow graph.
        cfg: ControlFlowGraph = ControlFlowGraph(function)
        # Worklist of blocks to work on.
        for block in cfg.basic_blocks:
            self.gather(block)
        self.solve(cfg)
        if ENABLE_ANALYSIS_DEBUG_MODE:
            for label, liveout in self.liveout.items():
                print(f"Block {label} liveout set is {liveout}")

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
    versions: Dict[str, int]

    def __hash__(self):
        return hash(
            (self.instr.op, tuple(self.instr.get_args()), tuple(self.versions.items()))
        )

    def __eq__(self, other):
        if not isinstance(other, Expression):
            return False
        return (
            self.instr.op == other.instr.op
            and self.instr.get_args() == other.instr.get_args()
            and self.versions == other.versions
        )

    def __str__(self):
        args = self.instr.get_args()
        lhs = f"{args[0]}_{self.versions[args[0]]}" if args is not None else ""
        rhs = f"{args[1]}_{self.versions[args[1]]}" if args is not None else ""
        return f"{self.instr.op.name} {lhs} {rhs}"


class AvailableExpressions(Analysis):
    """
    Available Expressions analysis is a forward data flow analysis that determines
    which expressions are already computed and still valid at a given program point.
    An expression is "available" if:
    - It has been computed on some path to the current program point.
    - The variables used in the expression haven't been redefined since its last computation.
    """

    def __init__(self):
        super().__init__(AVAILABLE_EXPRESSIONS)
        self.name: str = AVAILABLE_EXPRESSIONS
        # Expressions generated in each block
        self.gen: Dict[str, Set[Expression]] = {}
        # Expressions killed in each block
        self.kill: Dict[str, Set[Expression]] = {}
        # Available expressions at the end of each block
        self.avail_out: Dict[str, Set[Expression]] = {}
        # All expressions in the function
        self.all_expressions: Set[Expression] = set()

    def run(self, function):
        cfg = ControlFlowGraph(function)
        for block in cfg.basic_blocks:
            self.gather(block)
        self.solve(cfg)
        if ENABLE_ANALYSIS_DEBUG_MODE:
            print("Available expressions at block exits:")
            for label, exprs in self.avail_out.items():
                print(f"Block {label}: {exprs}")

    def gather(self, block: BasicBlock):
        self.gen[block.label] = set()
        self.kill[block.label] = set()
        versions = {}

        for instr in block.instructions:
            # If this a computed expression build an expression object
            # out of it.
            if instr.op in [
                OPCode.ADD,
                OPCode.SUB,
                OPCode.MUL,
                OPCode.DIV,
                OPCode.LAND,
                OPCode.LOR,
            ]:
                for arg in instr.get_args():
                    if arg not in versions:
                        versions[arg] = 0
                expr = Expression(instr, versions.copy())
                self.all_expressions.add(expr)
                self.gen[block.label].add(expr)
                # Kill all expressions using the destination variable
                self.kill[block.label] |= {
                    e
                    for e in self.all_expressions
                    if instr.get_dest() in e.instr.get_args()
                }
            else:
                # For non-arithmetic instructions, kill expressions using the destination
                if instr.get_dest():
                    versions[instr.get_dest()] = versions.get(instr.get_dest(), 0) + 1
                    self.kill[block.label] |= {
                        e
                        for e in self.all_expressions
                        if instr.get_dest() in e.instr.get_args()
                    }

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
            avail_in.difference(self.kill[block.label])
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


@dataclass
class Definition:
    # Variable name.
    variable: str
    # Block name or label.
    block: str
    # Index in the list of instructions.
    index: int

    def __hash__(self):
        return hash((self.variable, self.block, self.index))

    def __eq__(self, other):
        if not isinstance(other, Definition):
            return False
        return (
            self.variable == other.variable
            and self.block == other.block
            and self.index == other.index
        )

    def __str__(self):
        return f"{self.variable} defined in block {self.block} at index {self.index}"


class ReachingDefinitions(Analysis):
    """
    Reaching definitions determines which definitions of variables may reach
    a given point in the program.

    Similar to liveness and available expressions analysis, the goal is to
    collect facts about the program and use the facts to solve the data-flow
    equation.

    The facts collected from the program are:

    - GEN: Set of definitions generated in the block.
    - KILL: Set of definitions killed (overwritten) in the block.
    - IN: Set of definitions reaching the start of the block.
    - OUT: Set of definitions reaching the end of the block.

    For each block we want to solve the following data-flow equations:

    - IN[B] = UNION(OUT[P] for P in PRED(B))
    - OUT[B] = GEN[B] U (IN[B] - KILL[B])
    """

    def __init__(self):
        super().__init__(REACHING_DEFINITIONS)
        self.name: str = REACHING_DEFINITIONS
        self.gen: Dict[str, Set[Definition]] = {}
        self.kill: Dict[str, Set[Definition]] = {}
        self.in_defs: Dict[str, Set[Definition]] = {}
        self.out_defs: Dict[str, Set[Definition]] = {}
        self.all_defs: Set[Definition] = set()

    def run(self, function: Function):
        cfg: ControlFlowGraph = ControlFlowGraph(function)
        worklist: List[BasicBlock] = cfg.basic_blocks

        for block in worklist:
            self.gather(block)

        # Initialize the solves sets.
        for block in worklist:
            self.in_defs[block.label] = set()
            self.out_defs[block.label] = set()

        changed: bool = True
        while changed:
            changed = False
            for block in worklist:
                # Compute IN[B].
                new_in = set().union(
                    *(self.out_defs[p.label] for p in block.predecessors)
                )

                if new_in != self.in_defs[block.label]:
                    self.in_defs[block.label] = new_in
                    changed = True

                # Compute OUT[B]
                new_out = self.gen[block.label].union(
                    self.in_defs[block.label].difference(self.kill[block.label])
                )

                if new_out != self.out_defs[block.label]:
                    self.out_defs[block.label] = new_out
                    changed = True

        if ENABLE_ANALYSIS_DEBUG_MODE:
            print("Reaching Definitions Analysis Results:")
            for block_label, out_set in self.out_defs.items():
                print(f"Block {block_label} OUT:")
                for definition in out_set:
                    print(
                        f"  {definition.variable} defined in block {definition.block} at index {definition.index}"
                    )

    def gather(self, block: BasicBlock):
        """
        Gather facts for the block.
        """
        self.gen[block.label] = set()
        self.kill[block.label] = set()

        for ii, instr in enumerate(block.instructions):
            if instr.get_dest() is not None:
                new_def = Definition(instr.get_dest(), block.label, ii)
                self.all_defs.add(new_def)
                self.gen[block.label].add(new_def)
                # Kill previous definitions.
                self.kill[block.label] |= {
                    d for d in self.all_defs if d.variable == instr.get_dest()
                }

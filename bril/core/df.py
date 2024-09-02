"""
Implementation of a data-flow framework.

Data-flow frameworks are an approach that generalizes analyses based on
data-flow to a unified framework that uses lattices and a single global
solver to solve all equations. The equations of each analysis can be
reduced to a single `JOIN` function also described as `MEET` in other
references.

Most of this is based on algorithms described in Chapter 5 of SPA[1] by Møller
and Schwartzbach.

One common theme to all analyses we will look at are the Join, Meet and Transfer
functions. The Join function combines the facts from the predecessors of a node
in the CFG, the Transfer function works in the analysis direction and computes
the local output facts for the node in the CFG. Finally, the Meet function is
used to combine the facts from the successors of a node in the CFG.

Join and Meet are equivalent, Join is the terminology used for backward analyses
and Meet is the terminology used for forward analyses.

[1] SPA: Static Program Analysis
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from typing import Callable, Dict, Generic, List, Set, TypeVar

from bril.core.analyses import Analysis
from bril.core.cfg import ControlFlowGraph
from bril.core.ir import BasicBlock, Function, Instruction, OPCode

# Type variable for all latice elements.
T = TypeVar("T")


class Direction(Enum):
    """
    Direction of the analysis, backwards or forwards.
    """

    FORWARD = auto()
    BACKWARD = auto()


class Lattice(ABC, Generic[T]):
    """
    Lattice class that we will inherit the set-based analyses from.
    """

    @abstractmethod
    def bottom(self) -> T:
        """
        Returns the bottom element of the lattice.
        """
        pass

    @abstractmethod
    def top(self) -> T:
        """
        Returns the top element of the lattice.
        """
        pass

    @abstractmethod
    def join(self, *elements: T) -> T:
        """Perform the join operation on the given elements."""
        pass

    @abstractmethod
    def meet(self, *elements: T) -> T:
        """Perform the meet operation on the given elements."""
        pass


class SetLattice:
    def __init__(self, universe: Set):
        self.universe = universe

    def bottom(self) -> Set:
        return set()

    def top(self) -> Set:
        return self.universe.copy()

    def meet(self, *sets: Set) -> Set:
        return set.intersection(*sets) if sets else self.top()

    def join(self, *sets: Set) -> Set:
        return set.union(*sets) if sets else self.bottom()


class DataAnalysisFramework(Generic[T]):
    def __init__(
        self,
        cfg: ControlFlowGraph,
        direction: Direction,
        lattice: Lattice[T],
        transfer_function: Callable[[BasicBlock, T], T],
        initial_value: T,
    ):
        self.cfg: ControlFlowGraph = cfg
        self.direction: Direction = direction
        self.lattice: Lattice[T] = lattice
        self.transfer: Callable[[BasicBlock, T], T] = transfer_function
        self.initial_value: T = initial_value
        self.in_facts: Dict[str, T] = {}
        self.out_facts: Dict[str, T] = {}

    def solve(self):
        """
        Solve the data-flow equations using the worklist algorithm.
        """
        for block in self.cfg.basic_blocks:
            # Initialize all facts with the lattice bottom value.
            self.in_facts[block.label] = self.lattice.bottom()
            self.out_facts[block.label] = self.lattice.bottom()

            # Set the initial value for the entry and exit point.
            match self.direction:
                case Direction.FORWARD:
                    self.in_facts[self.cfg.entry.label] = self.initial_value
                case Direction.BACKWARD:
                    self.out_facts[self.cfg.exit.label] = self.initial_value

            # Build the worklist, the worklist must be worked in post order
            # if the analysis direction backwards, otherwise its reverse postorder.
            worklist: List[BasicBlock] = (
                self.cfg.reverse_postorder()
                if self.direction == Direction.FORWARD
                else self.cfg.postorder()
            )

            # Iteratively solve.
            while worklist:
                block = worklist.pop(0)

                match self.direction:
                    case Direction.FORWARD:
                        # Meet function for forward analysis.
                        in_fact = self.lattice.meet(
                            *(self.out_facts[pred.label] for pred in block.predecessors)
                        )
                        out_fact = self.transfer(block, in_fact)
                        # If the input or output facts for the block changed
                        # they need to be updated with the new facts and then
                        # propagated by adding all successors to the worklist.
                        #
                        # This is where we know if iteration converges or not.
                        if (
                            in_fact != self.in_facts[block.label]
                            or out_fact != self.out_facts[block.label]
                        ):
                            self.in_facts[block.label] = in_fact
                            self.out_facts[block.label] = out_fact
                            worklist.extend(block.successors)
                    case Direction.BACKWARD:
                        # Join operation for backward analysis.
                        out_fact = self.lattice.join(
                            *(self.in_facts[succ.label] for succ in block.successors)
                        )
                        in_fact = self.transfer(block, out_fact)
                        # Similar to forward analysis except this time we
                        # propagate analysis information by appending
                        # the predecessors to the worklist.
                        if (
                            out_fact != self.out_facts[block.label]
                            or in_fact != self.in_facts[block.label]
                        ):
                            self.out_facts[block.label] = out_fact
                            self.in_facts[block.label] = in_fact
                            worklist.extend(block.predecessors)

    def __repr__(self) -> str:
        s = "\n"
        for block in self.cfg.basic_blocks:
            s += f"Block {block.label}:"
            s += f"  IN:  {self.in_facts[block.label]}"
            s += f"  OUT: {self.out_facts[block.label]}"
        return s

    def results(self):
        print(self)


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


class AvailableExpressionsAnalysis:
    def __init__(self, cfg: ControlFlowGraph):
        self.cfg: ControlFlowGraph = cfg
        self.expressions: Set[Expression] = set()
        self.gen: Dict[str, Set[Expression]] = dict()
        self.kill: Dict[str, set[Expression]] = dict()

        self.gather()

    def gather(self):
        for block in self.cfg.basic_blocks:
            self.gen[block.label] = set()
            self.kill[block.label] = set()
            for instr in block.instructions:
                if instr.op in [OPCode.ADD, OPCode.SUB, OPCode.MUL, OPCode.DIV]:
                    expr = Expression(instr, dict())
                    self.expressions.add(expr)
                    self.gen[block.label].add(expr)
                    # Kill all expressions using the destination of this instruction
                    self.kill[block.label] |= {
                        e
                        for e in self.expressions
                        if instr.get_dest() in e.instr.get_args()
                    }

    def transfer(self, block: BasicBlock, in_fact: Set[Expression]) -> Set[Expression]:
        return (in_fact.difference(self.kill[block.label])).union(self.gen[block.label])

    def framework(self) -> "DataAnalysisFramework":
        """
        Instantiate the data analysis framework.
        """
        return DataAnalysisFramework(
            cfg=self.cfg,
            direction=Direction.FORWARD,
            lattice=SetLattice(self.expressions),
            transfer_function=self.transfer,
            initial_value=set(),
        )


@dataclass(frozen=True)
class Definition:
    variable: str
    block: str
    index: int

    def __str__(self):
        return f"{self.variable} @ {self.block}:{self.index}"


class ReachingDefinitionsAnalysis:
    def __init__(self, cfg: ControlFlowGraph):
        self.cfg: ControlFlowGraph = cfg
        self.definitions: Set[Definition] = set()
        self.gen: Dict[str, Set[Definition]] = {}
        self.kill: Dict[str, Set[Definition]] = {}
        self.gather()

    def gather(self):
        for block in self.cfg.basic_blocks:
            self.gen[block.label] = set()
            self.kill[block.label] = set()
            for i, instr in enumerate(block.instructions):
                if instr.get_dest() is not None:
                    new_def = Definition(instr.get_dest(), block.label, i)
                    self.definitions.add(new_def)
                    self.gen[block.label].add(new_def)
                    self.kill[block.label] |= {
                        d for d in self.definitions if d.variable == instr.get_dest()
                    }

    def transfer(self, block: BasicBlock, in_fact: Set[Definition]) -> Set[Definition]:
        return (in_fact - self.kill[block.label]) | self.gen[block.label]

    def framework(self) -> "DataAnalysisFramework":
        return DataAnalysisFramework(
            cfg=self.cfg,
            direction=Direction.FORWARD,
            lattice=SetLattice(self.definitions),
            transfer_function=self.transfer,
            initial_value=set(),
        )


class LiveVariablesAnalysis:
    def __init__(self, cfg: ControlFlowGraph):
        self.cfg: ControlFlowGraph = cfg
        self.variables: Set[str] = set()
        self.use: Dict[str, Set[str]] = {}
        self.def_: Dict[str, Set[str]] = {}
        self.gather()

    def gather(self):
        for block in self.cfg.basic_blocks:
            self.use[block.label] = set()
            self.def_[block.label] = set()
            for instr in block.instructions:
                if instr.get_dest() is not None:
                    self.def_[block.label].add(instr.get_dest())
                    self.variables.add(instr.get_dest())
                if instr.get_args() is not None:
                    self.use[block.label].update(
                        arg
                        for arg in instr.get_args()
                        if arg not in self.def_[block.label]
                    )
                    self.variables.update(instr.get_args())

    def transfer(self, block: BasicBlock, out_fact: Set[str]) -> Set[str]:
        return self.use[block.label] | (out_fact - self.def_[block.label])

    def framework(self) -> "DataAnalysisFramework":
        return DataAnalysisFramework(
            cfg=self.cfg,
            direction=Direction.BACKWARD,
            lattice=SetLattice(self.variables),
            transfer_function=self.transfer,
            initial_value=set(),
        )


class DataFlow(Analysis):
    """
    Data-flow analysis using the data-flow framework.
    """

    def __init__(self, function: Function):
        super().__init__("data-flow")
        self.cfg: ControlFlowGraph = ControlFlowGraph(function=function)
        self.analyses: List[DataAnalysisFramework] = [
            AvailableExpressionsAnalysis(cfg=self.cfg.copy()).framework(),
            ReachingDefinitionsAnalysis(cfg=self.cfg.copy()).framework(),
            LiveVariablesAnalysis(cfg=self.cfg.copy()).framework(),
        ]

    def run(self, function: Function):
        for analysis in self.analyses:
            analysis.solve()
            analysis.results()

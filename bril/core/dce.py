"""
Implementation of dead code elimination on basic blocks.
"""

from typing import List, Set

from bril.core.ir import (
    BasicBlock,
    ConstOperation,
    ControlFlowGraph,
    EffectOperation,
    Function,
    Instruction,
    ValueOperation,
)
from bril.core.transform import Transform


class DeadCodeElimination(Transform):
    """
    Dead code elimination pass as a combination of running global trivial
    dead code elimination and local dead code elimination.
    """

    def __init__(self):
        super().__init__("dce")

    def run(self, function: Function):
        while TrivialDeadCodeElimination()._run(
            function
        ) or RedundantStoreElimination()._run(function):
            pass


class TrivialDeadCodeElimination(Transform):
    """
    Implementation of trivial dead code elimination of instructions per basic
    block.

    The algorithm for dead code elimination proceeds by forming def-use sets
    where def-set is the set of all variables defined in the basic block and
    the use-set is the set of all variables used in a basic block. Sets are
    merged before the next step starts.

    Once both sets are defined then any instruction defined and not used in
    the function is a candidate for deletion.
    """

    def __init__(self):
        super().__init__("trivial-dce")

    def run(self, function: Function):
        while self._run(function):
            pass

    def _run(self, function: Function) -> bool:
        super().run(function)
        # Worklist is the list of basic blocks we will work on.
        worklist: List[BasicBlock] = ControlFlowGraph(function).basic_blocks
        # Defs is the set of defined names.
        defs: Set[str] = set()
        # Uses is the set of used names.
        uses: Set[str] = set()
        # Whether or not we eliminated any dead instructions.
        eliminated: bool = False

        # Build the sef of `defs`.
        for block in worklist:
            for inst in block.instructions:
                if isinstance(inst, ConstOperation) and inst.dest is not None:
                    defs.add(inst.dest)
                if isinstance(inst, ValueOperation) and inst.dest is not None:
                    defs.add(inst.dest)

        # Build the set of `uses`.
        for block in worklist:
            for inst in block.instructions:
                if isinstance(inst, ValueOperation) and inst.args is not None:
                    uses.update(inst.args)
                if isinstance(inst, EffectOperation) and inst.args is not None:
                    uses.update(inst.args)

        # Intersect both sets, eliminating all defs that have been used.
        candidates = defs.difference(uses)
        print(
            f"Found {len(candidates)} candidates for deletion : {[name for name in candidates]}"
        )

        # Iterate over the basic blocks, marking all instructions that are
        # candidates for deletion.
        for block in worklist:
            new_block = []
            for inst in block.instructions:
                assert (
                    isinstance(inst, ConstOperation)
                    or isinstance(inst, ValueOperation)
                    or isinstance(inst, EffectOperation)
                )
                if isinstance(inst, ConstOperation) and inst.dest not in candidates:
                    new_block.append(inst)
                if isinstance(inst, ValueOperation) and inst.dest not in candidates:
                    new_block.append(inst)
                if isinstance(inst, EffectOperation):
                    new_block.append(inst)

            eliminated |= len(new_block) != len(block.instructions)
            block.instructions[:] = new_block
        function.instructions[:] = ControlFlowGraph.reassemble(worklist)
        return eliminated


class RedundantStoreElimination(Transform):
    """
    Implementation of redundant store elimination, where unused assignment
    which are assigned to but unused are considered as candidates for deletion
    during the same basic block if they are assigned again.
    """

    def __init__(self):
        super().__init__("redundant-store-elimination")

    def run(self, function: Function):
        while self._run(function):
            pass

    def _run(self, function: Function) -> bool:
        super().run(function)
        # Worklist is the list of basic blocks we will work on.
        worklist: List[BasicBlock] = ControlFlowGraph(function).basic_blocks
        # Whether or not any redundant stores were found and eliminated.
        eliminated: bool = False

        # Iterate over the blocks in the worklist.
        for block in worklist:
            eliminated |= self.eliminate_dead_stores(block)
        function.instructions[:] = ControlFlowGraph.reassemble(worklist)
        return eliminated

    def eliminate_dead_stores(self, block: BasicBlock) -> bool:
        # Defs is the set of defined names.
        defs: Set[str] = set()
        # Uses is the set of used names.
        uses: Set[str] = set()
        # List of candidates for deletion.
        candidates: Set[str] = set()
        # Whether or not any redundant stores were found and eliminated.
        eliminated: bool = False
        for instr in block.instructions:
            # Check if the current instruction references a name that was
            # defined, if that is the cast then it is no longer a candidate
            # for deletion.
            if (
                isinstance(instr, ValueOperation)
                or isinstance(instr, EffectOperation)
                and instr.args is not None
            ):
                for name in instr.args:
                    # The argument to this value operation is defined
                    # so we can no longer consider it as a candidate.
                    if name in defs:
                        defs.remove(name)
            # Check if the current instruct ions assigns to a definition
            # that has already been defined before, in that case it is
            # considered a candidate for deletion.
            if (
                isinstance(instr, ConstOperation) or isinstance(instr, ValueOperation)
            ) and instr.dest is not None:
                dest = instr.dest
                if dest in defs:
                    # This definition assigns to a new unused definition
                    # the last definition is then a candidate for deletion.
                    candidates.add(dest)
                defs.add(instr.dest)
        print(f"Found {len(candidates)} redundant stores ", candidates)
        # Iterate over the instructions dropping the first assignment that
        # is considered redundant.
        instructions: List[Instruction] = []
        for instr in block.instructions:
            if (
                isinstance(instr, ConstOperation) or isinstance(instr, ValueOperation)
            ) and instr.dest in candidates:
                candidates.remove(instr.dest)
            else:
                instructions.append(instr)
        changed = len(block.instructions) != len(instructions)
        block.instructions[:] = instructions
        return changed

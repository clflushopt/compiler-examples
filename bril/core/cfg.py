"""
Implementation of the control flow graph class and methods.
"""

from typing import List, OrderedDict, Set

import bril.core.ir as ir
from bril.core.ir import BasicBlock, Function, Instruction, Label


class ControlFlowGraph:
    """
    Control flow graph for a function is a graph where nodes are basic blocks
    and edges are control flow instructions.
    """

    def __init__(self, function):
        self.function: Function = function
        self.basic_blocks: List[BasicBlock] = []
        self.block_map: OrderedDict[str, BasicBlock] = OrderedDict()
        self.entry: BasicBlock = None
        self.exit: BasicBlock = None

        # List of blocks we are going to build.
        blocks: List[BasicBlock] = []
        # Current block being processed.
        current_block: BasicBlock = []

        # The algorithm for forming basic blocks iterates over the function's
        # instruction and for each instruction, if it's not a terminator or a
        # label it gets appended to `current_block`.
        #
        # If the instruction is a terminator it gets appended to `current_block`
        # and the `current_block` is appended to `blocks`.
        #
        # In the case of labels the `current_block` is considered complete and
        # is appended to `blocks` with the label as its name.
        for instr in self.function.instructions:
            if not isinstance(instr, Label):
                # The current instruction is not label so it is appended to the
                # current block.
                current_block.append(instr)
                # If the current instruction is a terminator then the current
                # block is considered complete.
                if instr.is_terminator():
                    blocks.append(current_block)
                    # Start a new block.
                    current_block = []
            if isinstance(instr, Label):
                # The current instruction is a label, end the current block
                # or start a new one with the label.
                if current_block:
                    blocks.append(current_block)
                # Preserve the label as a start for the new block.
                current_block = [instr]
        if current_block:
            blocks.append(current_block)
        # We now have a list of basic blocks processed from the function's
        # instruction, the next step is to associate a label (name) to them
        # then compute their predecessors and successors.
        for i, block in enumerate(blocks):
            # If the first instruction in the block is a label use it as
            # the default block name, otherwise assign a synthetic name.
            label = block[0].name if isinstance(block[0], Label) else f"block_{i}"
            basic_block = BasicBlock(label=label, instructions=block)
            self.block_map[label] = basic_block
            self.basic_blocks.append(basic_block)

        # Computing successors and predecssors.
        for i, block in enumerate(self.basic_blocks):
            # The successor of the current block is either the target of a
            # branch or a fallthrough.
            for instr in block.instructions:
                if isinstance(instr, ir.Jmp):
                    target_label = instr.target_label()
                    target_block = self.block_map[target_label]
                    # Append the target block to the list of successors.
                    block.successors.append(target_block)
                    # Append the current block to the list of predecessors.
                    target_block.predecessors.append(block)
                if isinstance(instr, ir.Br):
                    # Label and basic block for the then branch.
                    then_label = instr.then_label()
                    then_block = self.block_map[then_label]
                    # Label and basic block for the else branch.
                    else_label = instr.else_label()
                    else_block = self.block_map[else_label]
                    # The current block has two successors then and else
                    # branches.
                    block.successors.append(then_block)
                    block.successors.append(else_block)
                    # They each have the current block as a predecessors.
                    then_block.predecessors.append(block)
                    else_block.predecessors.append(block)
                    # Return is a fall through instruction i.e does not name
                    # a target label, instead it returns back to the call site.

        # Set the entry point (first block).
        if self.basic_blocks and len(self.basic_blocks) > 0:
            self.entry = self.basic_blocks[0]

        # Find exit points (blocks with no successors or contain return instructions).
        exit_points: List[BasicBlock] = []
        for block in self.basic_blocks:
            if not block.successors or any(
                isinstance(instr, ir.Ret) for instr in block.instructions
            ):
                exit_points.append(block)

        # If there's only one exit block, set it as the exit otherwise create an
        # exit block and connect all exit_points to it via a path.
        if len(exit_points) == 1:
            self.exit = exit_points[0]
        else:
            self.exit = BasicBlock(label="__reserved__exit", instructions=[])
            for block in exit_points:
                block.successors.append(self.exit)
                self.exit.predecessors.append(block)

            self.basic_blocks.append(self.exit)
            self.block_map["__reserved__exit"] = self.exit

    def __str__(self):
        dot_str = "digraph {} {{\n".format(self.function.name)
        for block in self.basic_blocks:
            dot_str += f'  {block.label} [shape=box, label="'
            for instr in block.instructions:
                dot_str += str(instr) + "\\n"
            dot_str += '"];\n'
            for successor in block.successors:
                dot_str += f"  {block.label} -> {successor.label};\n"
        dot_str += "}"
        return dot_str

    def dfs(self) -> List[BasicBlock]:
        """
        Perform depth-first search traversal of the nodes in the control flow
        graph and build a list of basic blocks in post-order.
        """
        visited: Set[BasicBlock] = set()
        ordered: List[BasicBlock] = []

        def _dfs(node: BasicBlock):
            """
            depth-first search traversal (recursive).
            """
            if node in visited:
                return
            visited.add(node)
            for succ in node.successors:
                _dfs(succ)
            ordered.append(node)

        _dfs(self.entry)
        return ordered

    def postorder(self) -> List[BasicBlock]:
        """Return the nodes of the CFG in post-order."""
        return self.dfs()

    def reverse_postorder(self) -> List[BasicBlock]:
        """Return the nodes of the CFG in reverse post-order."""
        return list(reversed(self.dfs()))

    def copy(self):
        return ControlFlowGraph(
            Function(
                self.function.name,
                self.function.return_type,
                self.function.params.copy(),
                self.function.instructions.copy(),
            )
        )

    def reassemble(self) -> List[Instruction]:
        """
        Flatten basic blocks back to a list of instructions.
        """
        instructions = []
        for block in self.basic_blocks:
            for phi in block.phi_nodes:
                instructions.append(phi)
            for instr in block.instructions:
                instructions.append(instr)
        return instructions


def reassemble(blocks: List[BasicBlock]) -> List[Instruction]:
    """
    Flatten basic blocks back to a list of instructions.
    """
    instructions = []
    for block in blocks:
        for instr in block.instructions:
            instructions.append(instr)
    return instructions

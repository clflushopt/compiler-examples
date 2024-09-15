"""
Implementation of into and out-of SSA transformation passes.
"""

from collections import defaultdict
from typing import Dict, List, Set

from bril.core.analyses import Analysis, DominanceRelationship
from bril.core.cfg import ControlFlowGraph
from bril.core.ir import BasicBlock, Function, Phi


class SSA(Analysis):
    """
    Implementation of to/out-of SSA transformations
    """

    def __init__(self, function: Function, cfg: ControlFlowGraph):
        super().__init__("ssa-transform")
        self.cfg: ControlFlowGraph = cfg
        self.blocks: List[BasicBlock] = cfg.basic_blocks
        self.dominance = DominanceRelationship(cfg=cfg)

        # Build the dominance relationship since we will need the dominance
        # frontier to calculate phi placement later.
        self.dominance.run()
        # Build the defs table.
        self.defs = self.block_defs()

    def run(self, function: Function):
        return super().run(function)

    def block_defs(self) -> Dict[str, set]:
        """
        Build a mapping from variable names to the basic block where they
        are defined.
        """
        mapping = defaultdict(set)
        for block in self.blocks:
            for instr in block.instructions:
                if instr.get_dest() is not None:
                    mapping[instr.get_dest()].add(block.label)
        return dict(mapping)

    def phi_placement(self):
        """
        Find the placement of phi-nodes in the blocks and build a mapping
        of block names to variable names that require phi-nodes.
        """
        phis = {b: set() for b in self.blocks}
        df = self.dominance.dominance_frontier
        for variable, block_defs in self.defs.items():
            block_defs_list = list(block_defs)
            for block_label in block_defs_list:
                bb = self.cfg.block_map[block_label]
                for block in df[bb]:
                    # Add a phi node.
                    if variable not in phis[block]:
                        phis[block].add(variable)
                        if block.label not in block_defs_list:
                            block_defs_list.append(block.label)
        return phis


class SSATransform(Analysis):
    def __init__(self):
        super().__init__("ssa")

    def run(self, function: Function):
        cfg = ControlFlowGraph(function)
        ssa = SSA(cfg)
        ssa.transform()
        return ssa.cfg


class SSA:
    def __init__(self, cfg: ControlFlowGraph):
        self.cfg = cfg
        self.current_def: Dict[str, Dict[str, str]] = defaultdict(dict)
        self.counter: Dict[str, int] = defaultdict(int)
        self.phis: Dict[str, Set[str]] = defaultdict(set)
        self.types: Dict[str, str] = {}
        self.dominance: DominanceRelationship = DominanceRelationship(cfg=cfg)

        # Compute the dominance frontier.
        self.dominance.run()

    def transform(self):
        self.get_types()
        self.insert_phi_functions()
        self.rename_variables()

    def get_types(self):
        for block in self.cfg.basic_blocks:
            for instr in block.instructions:
                if instr.get_dest():
                    self.types[instr.dest] = instr.type

    def insert_phi_functions(self):
        dom_frontier = self.dominance.dominance_frontier
        defs = self.get_def_blocks()

        for var, def_blocks in defs.items():
            worklist = list(def_blocks)
            for block_label in worklist:
                for frontier_block in dom_frontier[block_label]:
                    if var not in self.phis[frontier_block.label]:
                        self.phis[frontier_block.label].add(var)
                        if frontier_block.label not in def_blocks:
                            worklist.append(frontier_block.label)

    def get_def_blocks(self) -> Dict[str, Set[str]]:
        defs = defaultdict(set)
        for block in self.cfg.basic_blocks:
            for instr in block.instructions:
                if instr.get_dest():
                    defs[instr.dest].add(block.label)
        return defs

    def rename_variables(self):
        self.rename_block(self.cfg.entry)

    def rename_block(self, block: BasicBlock):
        old_defs = {
            var: self.current_def[block.label].get(var)
            for var in self.current_def[block.label]
        }

        # Rename phi node destinations
        for var in self.phis[block.label]:
            new_name = self.new_name(var)
            self.current_def[block.label][var] = new_name
            phi = Phi(
                new_name, self.types.get(var, "int"), [], []
            )  # Assuming 'int' as default type
            block.add_phi(phi)

        # Rename in regular instructions
        for instr in block.instructions:
            if instr.get_args():
                new_args = [
                    self.current_def[block.label].get(arg, arg)
                    for arg in instr.get_args()
                ]
                instr.set_args(new_args)
            if instr.get_dest():
                new_name = self.new_name(instr.dest)
                self.current_def[block.label][instr.dest] = new_name
                instr.dest = new_name

        # Rename phi node arguments in successors
        for succ in block.successors:
            for var in self.phis[succ.label]:
                phi = next(p for p in succ.phi_nodes if p.dest.split("_")[0] == var)
                phi.args.append(self.current_def[block.label].get(var, "__undefined"))
                phi.labels.append(block.label)

        # Recursive calls
        for succ in block.successors:
            if succ.label not in self.current_def:
                self.rename_block(succ)

        # Restore old definitions
        self.current_def[block.label] = old_defs

    def new_name(self, name: str) -> str:
        self.counter[name] += 1
        return f"{name}_{self.counter[name]}"

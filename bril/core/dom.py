"""
Implementation of dominance computation over a control flow graph.

In a control flow graph a node A dominates B iif all paths from the entry
to B pass through A. In a control flow graph, the entry node dominates all
nodes and every node dominates itself. The dominance relation being transitive
means that if A dom B and B dom C then A dom C.

In this file we compute three data structures over a control flow graph based
on dominance relations.

- Dominator Tree: Tree representation where each node's parent is its immediate
  dominator.

- Dominance Frontier: The set of nodes where a given basic block's dominance stops
  i.e the set of nodes that are not dominated by a block but have a predecessor that
  is dominated by it. This is required to implement the pass for SSA construction.

- Dominator Set: The set of all nodes that dominate a basic block.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set

from bril.core.cfg import ControlFlowGraph
from bril.core.ir import BasicBlock


@dataclass
class DominanceInformation:
    parent: BasicBlock = None
    semi: int = 0
    idom: BasicBlock = None
    ancestor: BasicBlock = None
    bucket: Set[BasicBlock] = field(default_factory=set)
    label: str = None


class DominanceRelationship:
    """
    Standalone class that computes the dominance relation ship of all nodes
    using the Lengauer-Tarjan algorihtm.

    The Lengauer-Tarjan algorithm computes dominance relationships by initially
    doing a DFS numbering phase where each node is assigned a unique monotonically
    increasing number representing their traversal order, this has the effect of
    creating a spanning tree of the graph and provides a total ordering of the
    nodes.

    The next step is to compute the semi-dominator, which is the node with the lowest
    assigned  DFS number that can reach n through a path. The semi-dominator ends
    up as a candidate for the immediate dominator or an ancestor of the immediate
    dominator.

    Since the semi-dominator computation phase selects candidates for dominators
    the next step is to do dominator selection where for each node we either assign
    it its semi-dominator as its immediate dominator or we find another node with
    the same semi-dominator and use it as the immediate dominator.

    The original paper[1], while heavy in lemmas and proofs is quite readeable
    the implementation here is not based on the paper but instead uses [2] and [3]
    as references.

    [1]: https://dl.acm.org/doi/10.1145/357062.357071
    [2]: https://cs.au.dk/~gerth/advising/thesis/henrik-knakkegaard-christensen.pdf
    [3]: https://www.cs.princeton.edu/techreports/2005/737.pdf
    """

    def __init__(self, cfg: ControlFlowGraph) -> None:
        self.cfg = cfg
        self.info: Dict[BasicBlock, DominanceInformation] = {}
        self.vertex: List[BasicBlock] = []
        self.idom: Dict[BasicBlock, BasicBlock] = {}
        # The dominance information we want to compute and their data structures.
        self.dominator_tree: Dict[BasicBlock, Set[BasicBlock]] = {}
        self.dominator_frontier: Dict[BasicBlock, Set[BasicBlock]] = {}
        self.dominance_frontier: Dict[str, Set[BasicBlock]] = {}
        self.dominator_sets: Dict[BasicBlock, Set[BasicBlock]] = {}

    def run(self):
        """
        Compute the dominators over the control flow graph and build all the
        information dictionnaries to answer dominance relationship questions
        during analysis.
        """
        self._compute_dominators()
        self.build_dominator_tree()
        self.compute_dominance_frontier()
        self.compute_dominator_sets()

    def results(self):
        print(f"Dominator tree : {self.dominator_tree}")
        print(f"Dominator frontier : {self.dominator_frontier}")
        print(f"Dominator sets {self.dominator_sets}")

    def dfs(self, block: BasicBlock, parent: BasicBlock = None):
        """
        Implementation of the DFS phase with node numbering.
        """
        self.info[block.label] = DominanceInformation(
            parent=parent,
            semi=len(self.vertex),
            label=block.label,
        )
        self.vertex.append(block)

        for succ in block.successors:
            if succ not in self.info:
                self.dfs(succ, block)

    def compress(self, block: BasicBlock):
        """
        Implementation of path compression as described in the original paper
        this is essentially union-find.
        """
        current_block_info = self.info[block.label]
        if (
            current_block_info.ancestor is not None
            and self.info[current_block_info.ancestor.label]
        ):
            self.compress(current_block_info.ancestor)
            if (
                self.info[self.info[current_block_info.ancestor.label].label].semi
                < self.info[current_block_info.label].semi
            ):
                current_block_info.label = self.info[
                    current_block_info.ancestor.label
                ].label
            current_block_info.ancestor = self.info[current_block_info.label].ancestor

    def eval(self, block: BasicBlock) -> str:
        """
        Evaluate if the current block can be path compressed.
        """
        print(f"Block {block.label} Info {self.info}")
        current_block_info = self.info[block.label]
        if not current_block_info.ancestor:
            return block.label
        self.compress(block=block)
        return current_block_info.label

    def link(self, block: BasicBlock, other: BasicBlock):
        """
        Link two basic blocks to enable path compression optimization.
        """
        self.info[other.label].ancestor = block

    def _compute_dominators(self):
        """
        Compute dominators and dominance information for the control flow graph
        using Lengauer-Tarjan algorithm.
        """
        # Phase 1: Traverse and number the nodes in the graph.
        self.dfs(self.cfg.entry)

        # Phase 2: Compute and bucketize candidates for dominators
        # the graph is traversed in reverse DFS order.
        for i in range(len(self.vertex) - 1, 0, -1):
            w = self.vertex[i]
            info_w = self.info[w.label]

            # Compute semi-dominators for `w`.
            for v in w.predecessors:
                # Can it be path compressed ?
                u = self.eval(v)
                if self.info[u].semi < info_w.semi:
                    info_w.semi = self.info[u].semi

            # Add w to its own semis candidate list (every node dominates itself).
            self.info[self.vertex[info_w.semi].label].bucket.add(w)
            self.link(info_w.parent, w)

            # Process w's parent candidates as potential immediate dominators.
            for v in list(self.info[info_w.parent.label].bucket):
                self.info[info_w.parent.label].bucket.remove(v)
                u = self.eval(v)
                self.idom[v] = (
                    u if self.info[u].semi < self.info[v.label].semi else info_w.parent
                )

        # Phase 3 compute the immediate dominators (i.e candidate selection).
        for i in range(1, len(self.vertex)):
            w = self.vertex[i]
            if self.idom[w] != self.vertex[self.info[w.label].semi]:
                self.idom[w] = self.idom[self.idom[w]]

    def build_dominator_tree(self):
        """
        Build the dominator tree, assuming we have computed the dominance relations.
        """
        self.dominator_tree = {block.label: set() for block in self.cfg.basic_blocks}
        for block, idom in self.idom.items():
            if block != self.cfg.entry:
                self.dominator_tree[idom.copy().label].add(block.copy())

    def compute_dominator_sets(self):
        """
        Compute the dominator sets for each node.
        """
        self.dominator_sets = {block: set() for block in self.cfg.basic_blocks}
        for block in self.cfg.basic_blocks:
            self._compute_dominator_set(block)

    def _compute_dominator_set(self, block: BasicBlock):
        """
        Compute the set of dominators for a given basic block.
        """
        if block not in self.dominator_sets:
            self.dominator_sets[block] = {block}
            if block in self.idom:
                idom_set = self._compute_dominator_set(self.idom[block])
                self.dominator_sets[block].update(idom_set)
        return self.dominator_sets[block]

    def compute_dominance_frontier(self):
        # Compute the dominance frontier for each node
        self.dominator_frontier = {block: set() for block in self.cfg.basic_blocks}
        self.dominance_frontier = {
            block.label: set() for block in self.cfg.basic_blocks
        }
        for block in self.cfg.basic_blocks:
            if len(block.predecessors) >= 2:
                for pred in block.predecessors:
                    runner = pred
                    while runner != self.idom[block]:
                        self.dominance_frontier[runner.label].add(block)
                        self.dominator_frontier[runner].add(block)
                        runner = self.idom[runner]

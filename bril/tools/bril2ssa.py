"""
Parse a Bril format from its textual representation to SSA form.
"""

import json
import sys

import bril
import bril.core.cfg
import bril.core.ir
import bril.core.parser
import bril.core.ssa

if __name__ == "__main__":
    input = sys.stdin.read()
    text = bril.core.parser.parse(input, False)
    ast = json.loads(text)
    program = bril.core.ir.Program(ast)
    for function in program.functions:
        cfg = bril.core.cfg.ControlFlowGraph(function)
        ssa_form = bril.core.ssa.SSATransform().run(function=function)
        print(f"SSA:\n{ssa_form}")

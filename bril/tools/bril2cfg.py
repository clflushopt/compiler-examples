"""
Parse a Bril format from its textual representation to control flow graphs.
"""

import json
import sys

import bril
import bril.core.ir
import bril.core.parser

if __name__ == "__main__":
    input = sys.stdin.read()
    text = bril.core.parser.parse(input, False)
    ast = json.loads(text)
    program = bril.core.ir.Program(ast)
    for function in program.functions:
        cfg = bril.core.ir.ControlFlowGraph(function)
        print(cfg)

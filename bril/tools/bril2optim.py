"""
Convert a Bril program from textual representation to an optimized version
of the internal intermediate language.
"""

import json
import sys
from typing import List

import bril
import bril.core.dce
import bril.core.ir
import bril.core.parser
from bril.core.transform import Transform

OPTIMIZATIONS: List[Transform] = [
    bril.core.dce.GlobalDeadCodeElimination(),
    bril.core.dce.RedundantStoreElimination(),
]

if __name__ == "__main__":
    input = sys.stdin.read()
    text = bril.core.parser.parse(input, False)
    ast = json.loads(text)
    program = bril.core.ir.Program(ast)
    for function in program.functions:
        for optimization in OPTIMIZATIONS:
            optimization.run(function)
        print(function)

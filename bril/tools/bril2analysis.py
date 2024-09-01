"""
Run analysis on bril programs.
"""

import json
import sys
from typing import List

import bril
import bril.core.df
import bril.core.ir
import bril.core.parser
from bril.core.analyses import Analysis, Liveness

ANALYSES: List[Analysis] = [
    Liveness(),
]

if __name__ == "__main__":
    input = sys.stdin.read()
    text = bril.core.parser.parse(input, False)
    ast = json.loads(text)
    program = bril.core.ir.Program(ast)
    for function in program.functions:
        for analysis in ANALYSES:
            analysis.run(function)
        print(function)

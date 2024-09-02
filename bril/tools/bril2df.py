"""
Run analysis on bril programs using the data-flow framework.
"""

import json
import sys

import bril
import bril.core.df
import bril.core.ir
import bril.core.parser
from bril.core.df import DataFlow

if __name__ == "__main__":
    input = sys.stdin.read()
    text = bril.core.parser.parse(input, False)
    ast = json.loads(text)
    program = bril.core.ir.Program(ast)
    for function in program.functions:
        analyzer = DataFlow(function=function)
        analyzer.run(function=function)
        print(function)

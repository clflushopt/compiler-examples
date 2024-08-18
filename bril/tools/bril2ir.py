"""
Convert a Bril program from textual representation to the internal intermediate
language.
"""

import json
import sys

import bril
import bril.core.parser
import bril.core.ir

if __name__ == "__main__":
    input = sys.stdin.read()
    text = bril.core.parser.parse(input, False)
    ast = json.loads(text)
    program = bril.core.ir.Program(ast)
    for function in program.functions:
        print(function)

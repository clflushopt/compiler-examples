"""
Parse a Bril program from textual representation to its AST in JSON format.
"""

import json
import sys

import bril
import bril.core.parser

if __name__ == "__main__":
    program = sys.stdin.read()
    ast = bril.core.parser.parse(program, False)
    print(ast)

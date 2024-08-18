"""
Convert a Bril program from the AST in JSON format to text representation.
"""

import json
import sys

import bril
import bril.core.parser

if __name__ == "__main__":
    input = sys.stdin.read()
    program = json.loads(input)
    print(program)

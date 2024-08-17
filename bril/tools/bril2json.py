import json
import sys

import bril
import bril.core.parser

if __name__ == '__main__':
    program = sys.stdin.read()
    ast = bril.core.parser.parse(program, False)
    print(ast)

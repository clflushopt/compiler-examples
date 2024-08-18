"""
Turnstile: snapshot testing for optimizing compilers.


Turnstile allows you to write tests for compiler passes that run end-to-end
on some input text code. It can be used in declarative form, provide a list
of files to run as input and compare their outputs or in more imperative form
where you can test individual compiler passes.
"""

import json

import bril
import bril.core.ir
import bril.core.parser


def parse(input: str) -> bril.core.ir.Program:
    """
    Parse a Bril program from text to intermediate language.
    """
    text = bril.core.parser.parse(input, False)
    ast = json.loads(text)
    program = bril.core.ir.Program(ast)
    return program


def run():
    files = sys
    print("Turnstile testing...")


if __name__ == "__main__":
    run()

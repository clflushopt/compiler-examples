"""
Turnstile: snapshot testing for optimizing compilers.


Turnstile allows you to write tests for compiler passes that run end-to-end
on some input text code. It can be used in declarative form, provide a list
of files to run as input and compare their outputs or in more imperative form
where you can test individual compiler passes.
"""

import argparse
import json
import os

import bril
import bril.core.dce
import bril.core.ir
import bril.core.parser
import bril.core.transform


def parse(input: str) -> bril.core.ir.Program:
    """
    Parse a Bril program from text to intermediate language.
    """
    text = bril.core.parser.parse(input, False)
    ast = json.loads(text)
    program = bril.core.ir.Program(ast)
    return program


def run():
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="turnstile",
        description="Turnstile is a snapshot testing tool for compiler examples.",
        epilog="Happy Hacking !",
    )

    parser.add_argument("-i", "--input")
    parser.add_argument("-e", "--expected")
    parser.add_argument("-o", "--optimizations")

    # Parse command line arguments and ensure some are available.
    args = parser.parse_args()
    assert args.input is not None
    assert args.expected is not None

    # Define input and expected output.
    root = os.getcwd()
    input = ""
    expected = ""
    optimizations = (
        [arg.strip() for arg in args.optimizations.split(",")]
        if args.optimizations is not None
        else []
    )

    with open(os.path.join(root, args.input)) as f:
        input = f.read()

    with open(os.path.join(root, args.expected)) as f:
        expected = f.read()

    input_program = parse(input)
    expected_program = parse(expected)

    print(optimizations)
    transforms = [
        (
            bril.core.dce.DeadCodeElimination()
            if "dce" in optimizations
            else bril.core.transform.Identity()
        )
    ]

    for transform in transforms:
        for function in input_program.functions:
            transform.run(function)

    assert input_program.functions == expected_program.functions

    print(f"actual: {input_program}")
    print(f"expected: {expected_program}")

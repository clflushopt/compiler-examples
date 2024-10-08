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
import bril.core.lvn
import bril.core.parser
import bril.core.transform
from bril.core.transform import DEAD_CODE_ELIMINATION, LOCAL_VALUE_NUMBERING


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

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Whether to dump the program to text format.",
    )
    parser.add_argument(
        "-i", "--input", help="Path to an input file in the current directory."
    )
    parser.add_argument(
        "-e",
        "--expected",
        help="Path to the expected output file in the current directory.",
    )
    parser.add_argument(
        "-o",
        "--optimizations",
        help="Comma separated list of optimizations to run (see transform.py).",
    )

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

    if args.verbose:
        print(f"Optimizations to run: {optimizations}")

    transforms = []

    for optim in optimizations:
        print(f"Optimization: {optim}")
        if optim == DEAD_CODE_ELIMINATION:
            transforms.append(bril.core.dce.DeadCodeElimination())
        elif optim == LOCAL_VALUE_NUMBERING:
            transforms.append(bril.core.lvn.LocalValueNumbering())
        else:
            transforms.append(bril.core.transform.Identity())

    for transform in transforms:
        for function in input_program.functions:
            transform.run(function)
            print(f"Post transform function {function}")

    if args.verbose:
        print(f"actual: {input_program}")
        print(f"expected: {expected_program}")

    for actual_function, expected_function in zip(
        input_program.functions, expected_program.functions
    ):
        for actual_instr, expected_instr in zip(
            actual_function.instructions, expected_function.instructions
        ):
            assert str(actual_instr) == str(
                expected_instr
            ), f"expected {expected_instr} found {actual_instr}"

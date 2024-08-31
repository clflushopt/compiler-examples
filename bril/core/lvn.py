"""
Implementation of local value numbering.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple

from bril.core.ir import BasicBlock, ControlFlowGraph, Function, Instruction, OPCode
from bril.core.transform import Transform


@dataclass
class ValueNumber:
    # Assigned during construction.
    index: int
    # Values are represented by encoded instructions.
    value: Tuple[Instruction, (OPCode, int, int)]
    # The variable name associated with this value number, this is the destination
    # of the value or constant operation. Since effect operations will not be
    # processed in this table.
    variable: str


class LocalValueNumbering(Transform):
    """
    Implementation of local value numbering with common sub-expression
    elimination, copy propagation and constant folding.

    The algorithm for local value numbering starts by building what we will
    refer to as the value table. The value table maps value numbers to values
    where the values are compact representation of the instructions that use
    value numbers as arguments.

    For example consider the following code block.

    ```
    a: int = const 7
    b: int = const 8
    sum1: int = add a b
    sum2: int = add a b
    prod: int = mul sum1 sum2
    ret prod
    ```
    The equivalent value table for this code is built by iterating over
    the instructions and building an environment mapping to map variable
    names to value numbers.

    To build the value table, assign each instruction a number, the value
    number. Then write the value and its canonical name on the right handside.

    Value table at the current step

    | # |   VALUE    | Name|
    | 1 | (const, 7) | 'a' |
    | 2 | (const, 8) | 'b' |

    Environment at the current step:

    - environment = {'a':1, 'b':2}

    For the instruction `sum1: int = add a b` we now need to explain the value
    resolution process of LVN. When we encounter an instruction that consumes
    non constant values (unlike `const`) the first step is to check if that
    variable has an existing value number.

    environment.resolve('a') <=> VN(a) = 1
    environment.resolve('b') <=> VN(b) = 1

    The environment is how we encode the mapping of variables names to value
    numbers. In this case we can find that `a` and `b` already exist in the
    table. So we encode `add a b` as `(ADD, 1, 2)`.

    Value table at the current step

    | # |   VALUE    | Name   |
    | 1 | (const, 7) | 'a'    |
    | 2 | (const, 8) | 'b'    |
    | 3 | (add, 1, 2)| 'sum1' |

    Environment at the current step:

    - environment = {'a':1, 'b':2, sum1: '3'}

    For the next step, we add more logic to our variable resolution, where
    we essentially check if the right hand side has already been encoded.

    In this case `sum2: add a b` will be encoded as `(add, 1, 2)` if we
    query the value to row number mapping we find that we already computed
    such an expression, so in this case we don't add a new row to the value
    table but instead add a new entry to `environment` that maps the variable
    to an existing value number.

    Value table at the current step

    | # |   VALUE    | Name   |
    | 1 | (const, 7) | 'a'    |
    | 2 | (const, 8) | 'b'    |
    | 3 | (add, 1, 2)| 'sum1' |

    Environment at the current step:

    - environment = {'a':1, 'b':2, sum1: '3', sum2: '3'}

    Finally for `prod: mul sum1 sum2` we add the last entry to the value table
    and environment since it doesn't map or resolve.

    Value table at the current step

    | # |   VALUE    | Name   |
    | 1 | (const, 7) | 'a'    |
    | 2 | (const, 8) | 'b'    |
    | 3 | (add, 1, 2)| 'sum1' |
    | 4 | (sum, 3, 3)| 'prod' |

    Environment at the current step:

    - environment = {'a':1, 'b':2, 'sum1': 3, 'sum2': 3, 'prod': 4}


    With the value table prepared, writing other optimizations becomes
    trivial. For example, any variable in the environment that maps
    to an existing value number its right hand side can be replaced
    by an `id` instruction.

    ```

    # Consider the iteration when we reach `sum2`
    for (index, entry) in ordered_list(environment):
        # This will yield [(a, 1), (b, 2), (sum1, 3)]
        previous_instructions = ordered_list(environment)[:index]
        if entry.var in previous_instructions:
            previous_entry = previous_instructions.get(entry.var)
            entry.instr = IR.Id(previous_entry.dest)
            # sum2: int = id sum1
            block.rewrite(entry.index, entry.instr)

    ```

    For constant folding, we fold recursively the value numbers until we reach
    a leaf expression.
    """

    def __init__(self):
        super().__init__("lvn")
        # Mapping of variables to rows in the value table.
        self.environment: Dict[str, ValueNumber] = {}

    def run(self, function):
        self._run(function=function)

    def _run(self, function: Function):
        worklist: List[BasicBlock] = ControlFlowGraph(function).basic_blocks
        for block in worklist:
            pass

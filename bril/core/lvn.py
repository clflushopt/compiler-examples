"""
Implementation of local value numbering.
"""

from typing import Callable, Dict, List, Tuple, Union

import bril.core.ir as ir
from bril.core.cfg import ControlFlowGraph, reassemble
from bril.core.ir import BasicBlock, Function, Instruction
from bril.core.transform import Transform


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
        self.environment: Dict[Tuple, int] = {}
        # Mapping of variable names to their constant values.
        self.constants: Dict[str, Union[int, bool]] = {}
        # Mapping of value numbers to values.
        self.values: Dict[int, str] = {}
        self.next_value_number: int = 1
        self.FOLDABLE: List[Instruction] = [
            ir.Add,
            ir.Sub,
            ir.Mul,
            ir.Div,
            ir.Eq,
            ir.Neq,
            ir.Gt,
            ir.Gte,
            ir.Lt,
            ir.Lte,
            ir.Lnot,
            ir.Lor,
            ir.Land,
        ]

    def reset(self):
        """
        Reset the internal state.
        """
        self.environment: Dict[Tuple, int] = {}
        self.constants: Dict[str, Union[int, bool]] = {}
        self.values: Dict[int, str] = {}
        self.next_value_number: int = 1

    def vn(self, op, args):
        """
        Compute the value numbering of an expression.
        """
        if op is ir.OPCode.CONST:
            key = (op, tuple(args))
            folded = args[0]
        else:
            args = self._canonicalize(op, args)
            resolved = tuple(self.constants.get(arg, arg) for arg in args)
            key = (op, resolved)
            folded = self._fold(op, resolved)
        if key not in self.environment:
            self.environment[key] = self.next_value_number
            self.next_value_number += 1
        return self.environment[key], folded

    def _fold(self, op: ir.OPCode, args: Tuple[Union[int, bool]]):
        FOLDABLE: Dict[ir.OPCode, Callable] = {
            ir.OPCode.ADD: lambda x, y: int(x) + int(y),
            ir.OPCode.SUB: lambda x, y: int(x) - int(y),
            ir.OPCode.MUL: lambda x, y: int(x) * int(y),
            ir.OPCode.DIV: lambda x, y: int(x) // int(y),
            ir.OPCode.EQ: lambda x, y: bool(x) == bool(y),
            ir.OPCode.GT: lambda x, y: bool(x) > bool(y),
            ir.OPCode.GTE: lambda x, y: bool(x) >= bool(y),
            ir.OPCode.LT: lambda x, y: bool(x) < bool(y),
            ir.OPCode.LTE: lambda x, y: bool(x) <= bool(y),
            ir.OPCode.LOR: lambda x, y: bool(x) or bool(y),
            ir.OPCode.LAND: lambda x, y: bool(x) and bool(y),
            ir.OPCode.LNOT: lambda x: not bool(x),
        }
        if op in FOLDABLE:
            if op is ir.OPCode.LNOT:
                return FOLDABLE[op](args[0])
            else:
                return FOLDABLE[op](args[0], args[1])
        return None

    def _canonicalize(self, op, args):
        """
        Canonicalize arguments when encoding them, this is how we handle
        commutative operations.
        """
        if op is ir.OPCode.ADD or op is ir.OPCode.MUL:
            return sorted(args)
        return args

    def run(self, function):
        self._run(function=function)

    def _run(self, function: Function):
        worklist: List[BasicBlock] = ControlFlowGraph(function).basic_blocks
        optimized_blocks: List[BasicBlock] = []
        for block in worklist:
            optimized: List[Instruction] = []
            for instr in block.instructions:
                # In the case of constant instructions we compute their value
                # number and load them into the constants table.
                if isinstance(instr, ir.Const):
                    args = instr.get_args()
                    vn = self.vn(ir.Const, args)
                    self.constants[instr.get_dest()] = instr.value
                    self.values[vn] = instr.get_dest()
                    optimized.append(instr)
                elif isinstance(instr, tuple(self.FOLDABLE)):
                    vn, folded = self.vn(instr.op, instr.get_args())
                    if folded is not None:
                        # We successfully folded this instruction so we replace
                        # with `const`.
                        optimized.append(ir.Const(instr.get_dest(), instr.type, folded))
                        self.constants[instr.get_dest()] = folded
                        self.values[vn] = instr.get_dest()
                    if vn in self.values:
                        optimized.append(
                            ir.Id(instr.get_dest(), instr.type, self.values[vn])
                        )
                    else:
                        optimized.append(instr)

                    self.values[vn] = instr.get_dest()
                else:
                    optimized.append(instr)
            optimized_blocks.append(BasicBlock(block.label, optimized))
            # Reset the local LVN state for the next basic block.
            self.reset()
        # Reassemble the optimized basic blocks back into the function.
        function.instructions = reassemble(optimized_blocks)

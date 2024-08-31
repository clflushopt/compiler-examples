"""
Bril IL as a three address code intermediate representation where instructions
are written in the form `dst <- op args`.
"""

from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional


class OPCode(Enum):
    """
    Each instruction is represented semantically by an OPCode.
    """
    NOP = 0
    ID = 1
    CONST = 2
    ADD = 3
    MUL = 4
    SUB = 5
    DIV = 6
    EQ = 7
    NEQ = 20
    LT = 8
    GT = 9
    LTE = 10
    GTE = 11
    LNOT = 12
    LAND = 13
    LOR = 14
    JMP = 15
    BR = 16
    CALL = 17
    RET = 18
    PRINT = 19

class Instruction(ABC):
    """
    An instruction in Bril is composed of an OPCode and different arguments. 
    """
    @abstractmethod
    def __init__(self, op):
        self.op = op

    def is_terminator(self) -> bool:
        """
        Return `True` if the instruction is a terminator i.e terminates
        control flow for a basic block.

        Bril has only three terminators `jmp`, `br` and `ret`.
        """
        if isinstance(self, Jmp) or isinstance(self, Br) or isinstance(self,Ret):
            return True
        return False

    def is_label(self) -> bool:
        """
        Return `True` if the instruction is a label.
        """
        if isinstance(self, Label):
            return True
        return False

    def get_dest(self) -> Optional[str]:
        """
        Return the destination of the instruction if one exists.
        """
        if isinstance(self, ConstOperation):
            return self.dest
        if isinstance(self, ValueOperation):
            return self.dest
        if isinstance(self, EffectOperation):
            return None

    def get_args(self) -> List[str]:
        """
        Returns the list of instruction arguments.
        """ 
        if isinstance(self, ConstOperation):
            return [self.value]
        if isinstance(self, ValueOperation):
            return self.args
        if isinstance(self, EffectOperation):
            return self.args

class ConstOperation(Instruction):
    """
    Instructions which are considered constant i.e the produce a constant value
    without any side effects.
    """
    def __init__(self, dest, type, value):
        super().__init__("const")
        self.op = OPCode.CONST
        self.dest = dest
        self.type = type
        self.value = value


class ValueOperation(Instruction):
    """
    Instructions which produce values but no side effects such as arithmetic
    or comparison instructions.
    """
    def __init__(self, op, dest, type, args=None, funcs=None, labels=None):
        super().__init__(op)
        self.dest = dest
        self.type = type
        self.args = args if args else []
        self.funcs = funcs if funcs else []
        self.labels = labels if labels else []


class EffectOperation(Instruction):
    """
    Instructions which produce side effects i.e they change the program's
    control flow such as function calls or conditional jumps.
    """
    def __init__(self, op, args=None, funcs=None, labels=None):
        super().__init__(op)
        self.args = args if args else []
        self.funcs = funcs if funcs else []
        self.labels = labels if labels else []


@dataclass
class Const(ConstOperation):
    """
    `const` instruction produces a constant value such as an integer, character
    or memory address.
    """
    def __init__(self, dest, type, value):
        super().__init__(dest, type, value)

    def __str__(self):
        return f"{self.dest}: {self.type} = const {self.value}"

@dataclass
class Id(ValueOperation):
    """
    `id` instruction produces the value it takes as input.
    """
    dest: str
    type: str
    src: str

    def __init__(self, dest, type, src):
        super().__init__(OPCode.ID, dest, type, [src])
        self.dest = dest
        self.src = src
        self.type = type

    def __str__(self) -> str:
        return f"{self.dest}: {self.type} = id {self.src}"


@dataclass
class Add(ValueOperation):
    """
    `add` instruction produces the sum of two operands.
    """
    dest: str
    type: str
    lhs: str
    rhs: str

    def __init__(self, dest, type, lhs, rhs):
        super().__init__(OPCode.ADD, dest, type, [lhs, rhs])
        self.dest = dest
        self.type = type
        self.lhs = lhs
        self.rhs = rhs

    def __str__(self) -> str:
        return f"{self.dest}: {self.type} = add {self.lhs} {self.rhs}"


@dataclass
class Mul(ValueOperation):
    """
    `mul` instruction produces the product of two operands.
    """
    dest: str
    type: str
    lhs: str
    rhs: str

    def __init__(self, dest, type, lhs, rhs):
        super().__init__(OPCode.MUL, dest, type, [lhs, rhs])
        self.dest = dest
        self.type = type
        self.lhs = lhs
        self.rhs = rhs

    def __str__(self) -> str:
        return f"{self.dest}: {self.type} = mul {self.lhs} {self.rhs}"


@dataclass
class Sub(ValueOperation):
    """
    `sub` instruction produces the difference of two operands.
    """
    dest: str
    type: str
    lhs: str
    rhs: str

    def __init__(self, dest, type, lhs, rhs):
        super().__init__(OPCode.SUB, dest, type, [lhs, rhs])
        self.dest = dest
        self.type = type
        self.lhs = lhs
        self.rhs = rhs

    def __str__(self) -> str:
        return f"{self.dest}: {self.type} = sub {self.lhs} {self.rhs}"


@dataclass
class Div(ValueOperation):
    """
    `div` instruction produces the integer division of two operands.
    """
    dest: str
    type: str
    lhs: str
    rhs: str

    def __init__(self, dest, type, lhs, rhs):
        super().__init__(OPCode.DIV, dest, type, [lhs, rhs])
        self.dest = dest
        self.type = type
        self.lhs = lhs
        self.rhs = rhs

    def __str__(self) -> str:
        return f"{self.dest}: {self.type} = div {self.lhs} {self.rhs}"


@dataclass
class Eq(ValueOperation):
    """
    `eq` instruction implements the equality operator and produces a boolean
    value showing whether two operands are equal.
    """
    dest: str
    type: str
    lhs: str
    rhs: str

    def __init__(self, dest, type, lhs, rhs):
        super().__init__(OPCode.EQ, dest, type, [lhs, rhs])
        self.dest = dest
        self.type = type
        self.lhs = lhs
        self.rhs = rhs

    def __str__(self) -> str:
        return f"{self.dest}: {self.type} = eq {self.lhs} {self.rhs}"


@dataclass
class Neq(ValueOperation):
    """
    `neq` instruction implements the equality operator and produces a boolean
    value showing whether two operands are equal.
    """
    dest: str
    type: str
    lhs: str
    rhs: str

    def __init__(self, dest, type, lhs, rhs):
        super().__init__(OPCode.NEQ, dest, type, [lhs, rhs])
        self.dest = dest
        self.type = type
        self.lhs = lhs
        self.rhs = rhs

    def __str__(self) -> str:
        return f"{self.dest}: {self.type} = neq {self.lhs} {self.rhs}"

@dataclass
class Lt(ValueOperation):
    """
    `lt` instruction implements the lesser than operator and produces a boolean
    value showing whether the first operand is lesser than the second operand.
    """
    dest: str
    type: str
    lhs: str
    rhs: str

    def __init__(self, dest, type, lhs, rhs):
        super().__init__(OPCode.LT, dest, type, [lhs, rhs])
        self.dest = dest
        self.type = type
        self.lhs = lhs
        self.rhs = rhs

    def __str__(self) -> str:
        return f"{self.dest}: {self.type} = lt {self.lhs} {self.rhs}"


@dataclass
class Gt(ValueOperation):
    """
    `lt` instruction implements the greater than operator and produces a boolean
    value showing whether the first operand is greater than the second operand.
    """
    dest: str
    type: str
    lhs: str
    rhs: str

    def __init__(self, dest, type, lhs, rhs):
        super().__init__(OPCode.GT, dest, type, [lhs, rhs])
        self.dest = dest
        self.type = type
        self.lhs = lhs
        self.rhs = rhs

    def __str__(self) -> str:
        return f"{self.dest}: {self.type} = gt {self.lhs} {self.rhs}"


@dataclass
class Lte(ValueOperation):
    """
    `lte` instruction implements the lesser than or equal comparison operator
    and produces a boolean value showing whether the first operand is lesser
    than or equal the second operand.
    """
    dest: str
    type: str
    lhs: str
    rhs: str

    def __init__(self, dest, type, lhs, rhs):
        super().__init__(OPCode.LTE, dest, type, [lhs, rhs])
        self.dest = dest
        self.type = type
        self.lhs = lhs
        self.rhs = rhs

    def __str__(self) -> str:
        return f"{self.dest}: {self.type} = lte {self.lhs} {self.rhs}"


@dataclass
class Gte(ValueOperation):
    """
    `gte` instruction implements the greater than or equal comparison operator
    and produces a boolean value showing whether the first operand is greater 
    than or equal the second operand.
    """
    dest: str
    type: str
    lhs: str
    rhs: str

    def __init__(self, dest, type, lhs, rhs):
        super().__init__(OPCode.GTE, dest, type, [lhs, rhs])
        self.dest = dest
        self.type = type
        self.lhs = lhs
        self.rhs = rhs

    def __str__(self) -> str:
        return f"{self.dest}: {self.type} = gte {self.lhs} {self.rhs}"


@dataclass
class Lnot(ValueOperation):
    """
    `lnot` instruction implements the logical not operator (negation).   
    """
    dest: str
    type: str
    arg: str

    def __init__(self, dest,type, arg):
        super().__init__(OPCode.LNOT, dest, type, [arg])
        self.dest = dest
        self.type = type
        self.arg = arg

    def __str__(self) -> str:
        return f"{self.dest}: {self.type} = not {self.arg}"


@dataclass
class Land(ValueOperation):
    """
    `land` implements logical and operator.
    """
    dest: str
    type: str
    lhs: str
    rhs: str

    def __init__(self, dest, type, lhs, rhs):
        super().__init__(OPCode.LAND, dest, type, [lhs, rhs])
        self.dest = dest
        self.type = type
        self.lhs = lhs
        self.rhs = rhs

    def __str__(self) -> str:
        return f"{self.dest}: {self.type} = and {self.lhs} {self.rhs}"


@dataclass
class Lor(ValueOperation):
    """
    `lor` implements logical or operator.
    """
    dest: str
    type: str
    lhs: str
    rhs: str

    def __init__(self, dest, type, lhs, rhs):
        super().__init__(OPCode.LOR, dest, type, [lhs, rhs])
        self.dest = dest
        self.type = type
        self.lhs = lhs
        self.rhs = rhs

    def __str__(self) -> str:
        return f"{self.dest}: {self.type} = or {self.lhs} {self.rhs}"


@dataclass
class Call(ValueOperation):
    """
    `call` instruction encodes function calls.
    """
    dest: str
    type: str
    func: str
    args: List[str]

    def __init__(self, dest, type, func, args):
        super().__init__(OPCode.CALL, dest, type, args, funcs=[func])
        self.dest = dest
        self.type = type
        self.func = func
        self.args = [arg for arg in args] if args else []

    def __str__(self) -> str:
        if self.dest is not None and self.type is not None:
            return f"{self.dest}: {self.type} = call {self.func} {" ".join(self.args)}"
        else :
            return f"call {self.func} {" ".join(self.args)}"


@dataclass
class Nop(EffectOperation):
    """
    `nop` instruction for no-operation.
    """
    def __init__(self):
        super().__init__(OPCode.NOP)

    def __str__(self) -> str:
        return f"nop"


@dataclass
class Print(EffectOperation):
    """
    `print` instruction used mainly for tracing or interpretation.
    """
    arg: str

    def __init__(self, arg):
        super().__init__(OPCode.PRINT, args=[arg])
        self.arg = arg

    def __str__(self) -> str:
        return f"print {self.arg}"


@dataclass
class Jmp(EffectOperation):
    """
    `jmp` instruction represents direct jumps.
    """
    target: str

    def __init__(self, label):
        super().__init__(OPCode.JMP, labels=[label])
        self.target = label

    def target_label(self):
        """
        Return the target (label) of the jump.
        """
        return self.target
    def __str__(self) -> str:
        return f"jmp .{self.target}"


@dataclass
class Br(EffectOperation):
    """
    `br` instruction represents conditional branches.
    """
    def __init__(self, cond, true_label, false_label):
        super().__init__(OPCode.BR, args=cond, labels=[true_label, false_label])

    def then_label(self):
        """
        Return the target label of the then case.
        """
        return self.labels[0]

    def else_label(self):
        """
        Return the target label of the else case.
        """
        return self.labels[1]

    def __str__(self) -> str:
        return f"br {self.args[0]} .{self.labels[0]} .{self.labels[1]}"


@dataclass
class Ret(EffectOperation):
    """
    `ret` instruction for returning from a call site.
    """
    value: str

    def __init__(self, value=None):
        super().__init__(OPCode.RET, args=[value] if value else [])
        self.value = value

    def __str__(self) -> str:
        if self.value:
            return f"ret {self.value}"
        else:
            return f"ret"


@dataclass
class Label(Instruction):
    name: str

    """
    `label` is a pseudo-instruction and is used to mark the target of direct
    and conditional jumps.
    """
    def __init__(self, name):
        super().__init__(Label)
        self.name = name
    
    def name(self) -> str:
        """
        Return the label name.
        """
        return self.name

    def __str__(self):
        return f".{self.name}:"


@dataclass
class Function:
    """
    Functions in the Bril intermediate language encode the function name and
    signature and the list of instructions representing the function.
    """
    def __init__(self, name, return_type, params, instructions):
        """
        Form a function from a list of instructions.
        """
        self.name:str = name
        self.return_type:str = return_type
        self.params:List[str] = params
        self.instructions:List[Instruction] = instructions

    def __init__(self, ast:dict):
        """
        Form a function from a JSON encoded AST.
        """
        self.instructions = []
        self.return_type = ast.get("type", "void")
        self.params = ast.get("args", [])
        self.name = ast.get("name", "main")
        for inst in ast["instrs"]:
            instruction: Instruction
            assert isinstance(inst, dict)
            if inst.get("label") is not None:
                instruction = Label(inst["label"])
            elif inst.get("op") is not None:
                match inst["op"]:
                    case "id":
                        instruction = Id(inst["dest"], inst["type"], inst["args"][0])
                    case "const":
                        instruction = Const(inst["dest"], inst["type"], inst["value"])
                    case "add":
                        instruction = Add(
                            inst["dest"], inst["type"], inst["args"][0], inst["args"][1]
                        )
                    case "mul":
                        instruction = Mul(
                            inst["dest"], inst["type"], inst["args"][0], inst["args"][1]
                        )
                    case "sub":
                        instruction = Sub(
                            inst["dest"], inst["type"], inst["args"][0], inst["args"][1]
                        )
                    case "div":
                        instruction = Div(
                            inst["dest"], inst["type"], inst["args"][0], inst["args"][1]
                        )
                    case "eq":
                        instruction = Eq(
                            inst["dest"], inst["type"], inst["args"][0], inst["args"][1]
                        )
                    case "lt":
                        instruction = Lt(
                            inst["dest"], inst["type"], inst["args"][0], inst["args"][1]
                        )
                    case "gt":
                        instruction = Gt(
                            inst["dest"], inst["type"], inst["args"][0], inst["args"][1]
                        )
                    case "lte":
                        instruction = Lte(
                            inst["dest"], inst["type"], inst["args"][0], inst["args"][1]
                        )
                    case "gte":
                        instruction = Gte(
                            inst["dest"], inst["type"], inst["args"][0], inst["args"][1]
                        )
                    case "not":
                        instruction = Lnot(inst["dest"], inst["type"], inst["args"][0])
                    case "and":
                        instruction = Land(
                            inst["dest"], inst["type"], inst["args"][0], inst["args"][1]
                        )
                    case "or":
                        instruction = Lor(
                            inst["dest"], inst["type"], inst["args"][0], inst["args"][1]
                        )
                    case "jmp":
                        instruction = Jmp(inst["labels"][0])
                    case "call":
                        instruction = Call(
                            inst.get("dest"), inst.get("type"), inst["funcs"][0], inst.get("args")
                        )
                    case "br":
                        instruction = Br(
                            inst["args"], inst["labels"][0], inst["labels"][1]
                        )
                    case "print":
                        instruction = Print(inst["args"][0])
                    case "ret":
                        instruction = Ret(None if inst.get("args") is None else inst["args"][0])
                    case "nop":
                        instruction = Nop()
                    case _:
                        raise NotImplementedError("unimplemented instruction for {}".format(inst))
            # Append the instruction the actual code section.
            self.instructions.append(instruction)

    def __str__(self):
        param_str = (
            ""
            if len(self.params) == 0
            else "(" + ", ".join(f"{param["name"]}: {param["type"]}" for param in self.params) + ")"
        )
        return (
            f"{self.name}: {self.return_type}{"" if len(param_str) == 0 else param_str} {{\n"
            + "\n".join("   " + instr.__str__() for instr in self.instructions)
            + "\n"
            + "}"
        )


class Program:
    """
    Programs in the Bril intermediate language are just sequence of functions
    without any top level declarations.
    """
    def __init__(self, ast):
        """
        Parse an AST from JSON format to a list of instructions.
        """
        self.functions: List[Function] = []
        for function in ast["functions"]:
            func = Function(function)
            self.functions.append(func)

    def __str__(self) -> str:
        functions_str = '\n'
        for function in self.functions:
            functions_str += f"{function}\n"
        return functions_str

            

class BasicBlock:
    """
    Basic block is a straight-line sequence of Bril instructions without
    branches except to the entry and at the exit.
    """
    def __init__(self, label, instructions):
        self.label:str = label
        self.instructions:List[Instruction] = instructions
        self.successors:List[BasicBlock] = []
        self.predecessors:List[BasicBlock] = []

    def __repr__(self) -> str:
        s = "\n"
        for instr in self.instructions:
            s += f"{instr}\n"
        return s

class ControlFlowGraph:
    """
    Control flow graph for a function is a graph where nodes are basic blocks
    and edges are control flow instructions.
    """
    def __init__(self, function):
        self.function: Function = function
        self.basic_blocks:List[BasicBlock] = []
        self.block_map: OrderedDict[str, BasicBlock] = OrderedDict()
        # List of blocks we are going to build.
        blocks = [] 
        # Current block being processed.
        current_block = []

        # The algorithm for forming basic blocks iterates over the function's
        # instruction and for each instruction, if it's not a terminator or a
        # label it gets appended to `current_block`.
        # 
        # If the instruction is a terminator it gets appended to `current_block`
        # and the `current_block` is appended to `blocks`.
        # 
        # In the case of labels the `current_block` is considered complete and
        # is appended to `blocks` with the label as its name.
        for instr in self.function.instructions:
            if not isinstance(instr, Label):
                # The current instruction is not label so it is appended to the
                # current block.
                current_block.append(instr)
                # If the current instruction is a terminator then the current
                # block is considered complete.
                if instr.is_terminator():
                    blocks.append(current_block)
                    # Start a new block.
                    current_block = []
            if isinstance(instr, Label):
                # The current instruction is a label, end the current block
                # or start a new one with the label.
                if current_block:
                    blocks.append(current_block)
                # Preserve the label as a start for the new block.
                current_block = [instr]
        if current_block:
            blocks.append(current_block)
        # We now have a list of basic blocks processed from the function's
        # instruction, the next step is to associate a label (name) to them
        # then compute their predecessors and successors.
        for i, block in enumerate(blocks):
            # If the first instruction in the block is a label use it as
            # the default block name, otherwise assign a synthetic name.
            label = block[0].name if isinstance(block[0], Label) else f"block_{i}"
            basic_block = BasicBlock(label=label, instructions=block)
            self.block_map[label] = basic_block
            self.basic_blocks.append(basic_block)

        # Computing successors and predecssors.
        for i, block in enumerate(self.basic_blocks):
            # The successor of the current block is either the target of a
            # branch or a fallthrough.
            for instr in block.instructions:
                if isinstance(instr, Jmp):
                    target_label = instr.target_label()
                    target_block = self.block_map[target_label]
                    # Append the target block to the list of successors.
                    block.successors.append(target_block)
                    # Append the current block to the list of predecessors.
                    target_block.predecessors.append(block)
                if isinstance(instr, Br):
                    # Label and basic block for the then branch.
                    then_label = instr.then_label()
                    then_block = self.block_map[then_label]
                    # Label and basic block for the else branch.
                    else_label = instr.else_label()
                    else_block = self.block_map[else_label]
                    # The current block has two successors then and else
                    # branches.
                    block.successors.append(then_block)
                    block.successors.append(else_block)
                    # They each have the current block as a predecessors.
                    then_block.predecessors.append(block)
                    else_block.predecessors.append(block)
                    # Return is a fall through instruction i.e does not name
                    # a target label, instead it returns back to the call site.

    def __str__(self):
        dot_str = "digraph {} {{\n".format(self.function.name)
        for block in self.basic_blocks:
            dot_str += f'  {block.label} [shape=box, label="'
            for instr in block.instructions:
                dot_str += str(instr) + "\\n"
            dot_str += '"];\n'
            for successor in block.successors:
                dot_str += f"  {block.label} -> {successor.label};\n"
        dot_str += "}"
        return dot_str

    def reassemble(blocks: List[BasicBlock]) -> List[Instruction]:
        """
        Flatten basic blocks back to a list of instructions.
        """
        instructions = []
        for block in blocks :
            for instr in block.instructions:
                instructions.append(instr)
        return instructions

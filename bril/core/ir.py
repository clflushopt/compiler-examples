"""
Bril IL as a three address code intermediate representation.
"""

from abc import abstractmethod, ABC
from dataclasses import dataclass
from typing import Any, Optional, List
from enum import Enum


class Opcode(Enum):
    NOP = 0
    ID = 1
    CONST = 2
    ADD = 3
    MUL = 4
    SUB = 5
    DIV = 6
    EQ = 7
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
    @abstractmethod
    def __init__(self, op):
        self.op = op

    def is_terminator(self) -> bool:
        if isinstance(self, Jmp) or isinstance(self, Br) or isinstance(self,Ret):
            return True
        return False

class ConstOperation(Instruction):
    def __init__(self, dest, type, value):
        super().__init__("const")
        self.op = Opcode.CONST
        self.dest = dest
        self.type = type
        self.value = value


class ValueOperation(Instruction):
    def __init__(self, op, dest, type, args=None, funcs=None, labels=None):
        super().__init__(op)
        self.dest = dest
        self.type = type
        self.args = args if args else []
        self.funcs = funcs if funcs else []
        self.labels = labels if labels else []


class EffectOperation(Instruction):
    def __init__(self, op, args=None, funcs=None, labels=None):
        super().__init__(op)
        self.args = args if args else []
        self.funcs = funcs if funcs else []
        self.labels = labels if labels else []


@dataclass
class Const(ConstOperation):
    def __init__(self, dest, type, value):
        super().__init__(dest, type, value)

    def __str__(self):
        return f"{self.dest}: {self.type} = const {self.value}"


@dataclass
class Id(ValueOperation):
    dest: str
    type: str
    src: str

    def __init__(self, dest, type, src):
        super().__init__(Opcode.ID, dest, type, [src])
        self.dest = dest
        self.src = src
        self.type = type

    def __str__(self) -> str:
        return f"{self.dest}: {self.type} = id {self.src}"


@dataclass
class Add(ValueOperation):
    dest: str
    type: str
    lhs: str
    rhs: str

    def __init__(self, dest, type, lhs, rhs):
        super().__init__(Opcode.ADD, dest, type, [lhs, rhs])
        self.dest = dest
        self.type = type
        self.lhs = lhs
        self.rhs = rhs

    def __str__(self) -> str:
        return f"{self.dest}: {self.type} = add {self.lhs} {self.rhs}"


@dataclass
class Mul(ValueOperation):
    dest: str
    type: str
    lhs: str
    rhs: str

    def __init__(self, dest, type, lhs, rhs):
        super().__init__(Opcode.MUL, dest, type, [lhs, rhs])
        self.dest = dest
        self.type = type
        self.lhs = lhs
        self.rhs = rhs

    def __str__(self) -> str:
        return f"{self.dest}: {self.type} = mul {self.lhs} {self.rhs}"


@dataclass
class Sub(ValueOperation):
    dest: str
    type: str
    lhs: str
    rhs: str

    def __init__(self, dest, type, lhs, rhs):
        super().__init__(Opcode.SUB, dest, type, [lhs, rhs])
        self.dest = dest
        self.type = type
        self.lhs = lhs
        self.rhs = rhs

    def __str__(self) -> str:
        return f"{self.dest}: {self.type} = sub {self.lhs} {self.rhs}"


@dataclass
class Div(ValueOperation):
    dest: str
    type: str
    lhs: str
    rhs: str

    def __init__(self, dest, type, lhs, rhs):
        super().__init__(Opcode.DIV, dest, type, [lhs, rhs])
        self.dest = dest
        self.type = type
        self.lhs = lhs
        self.rhs = rhs

    def __str__(self) -> str:
        return f"{self.dest}: {self.type} = div {self.lhs} {self.rhs}"


@dataclass
class Eq(ValueOperation):
    dest: str
    type: str
    lhs: str
    rhs: str

    def __init__(self, dest, type, lhs, rhs):
        super().__init__(Opcode.EQ, dest, type, [lhs, rhs])
        self.dest = dest
        self.type = type
        self.lhs = lhs
        self.rhs = rhs

    def __str__(self) -> str:
        return f"{self.dest}: {self.type} = eq {self.lhs} {self.rhs}"


@dataclass
class Lt(ValueOperation):
    dest: str
    type: str
    lhs: str
    rhs: str

    def __init__(self, dest, type, lhs, rhs):
        super().__init__(Opcode.LT, dest, type, [lhs, rhs])
        self.dest = dest
        self.type = type
        self.lhs = lhs
        self.rhs = rhs

    def __str__(self) -> str:
        return f"{self.dest}: {self.type} = lt {self.lhs} {self.rhs}"


@dataclass
class Gt(ValueOperation):
    dest: str
    type: str
    lhs: str
    rhs: str

    def __init__(self, dest, type, lhs, rhs):
        super().__init__(Opcode.GT, dest, type, [lhs, rhs])
        self.dest = dest
        self.type = type
        self.lhs = lhs
        self.rhs = rhs

    def __str__(self) -> str:
        return f"{self.dest}: {self.type} = gt {self.lhs} {self.rhs}"


@dataclass
class Lte(ValueOperation):
    dest: str
    type: str
    lhs: str
    rhs: str

    def __init__(self, dest, type, lhs, rhs):
        super().__init__(Opcode.LTE, dest, type, [lhs, rhs])
        self.dest = dest
        self.type = type
        self.lhs = lhs
        self.rhs = rhs

    def __str__(self) -> str:
        return f"{self.dest}: {self.type} = lte {self.lhs} {self.rhs}"


@dataclass
class Gte(ValueOperation):
    dest: str
    type: str
    lhs: str
    rhs: str

    def __init__(self, dest, type, lhs, rhs):
        super().__init__(Opcode.GTE, dest, type, [lhs, rhs])
        self.dest = dest
        self.type = type
        self.lhs = lhs
        self.rhs = rhs

    def __str__(self) -> str:
        return f"{self.dest}: {self.type} = gte {self.lhs} {self.rhs}"


@dataclass
class Lnot(ValueOperation):
    dest: str
    type: str
    arg: str

    def __init__(self, dest, arg):
        super().__init__(Opcode.LNOT, dest, type, [arg])
        self.dest = dest
        self.type = type
        self.arg = arg

    def __str__(self) -> str:
        return f"{self.dest}: {self.type} = not {self.arg}"


@dataclass
class Land(ValueOperation):
    dest: str
    type: str
    lhs: str
    rhs: str

    def __init__(self, dest, type, lhs, rhs):
        super().__init__(Opcode.LAND, dest, type, [lhs, rhs])
        self.dest = dest
        self.type = type
        self.lhs = lhs
        self.rhs = rhs

    def __str__(self) -> str:
        return f"{self.dest}: {self.type} = and {self.lhs} {self.rhs}"


@dataclass
class Lor(ValueOperation):
    dest: str
    type: str
    lhs: str
    rhs: str

    def __init__(self, dest, type, lhs, rhs):
        super().__init__(Opcode.LOR, dest, type, [lhs, rhs])
        self.dest = dest
        self.type = type
        self.lhs = lhs
        self.rhs = rhs

    def __str__(self) -> str:
        return f"{self.dest}: {self.type} = or {self.lhs} {self.rhs}"


@dataclass
class Call(ValueOperation):
    dest: str
    type: str
    func: str
    args: List[str]

    def __init__(self, dest, type, func, args):
        super().__init__(Opcode.CALL, dest, type, args, funcs=[func])
        self.dest = dest
        self.type = type
        self.func = func
        self.args = [arg for arg in args]

    def __str__(self) -> str:
        return f"{self.dest}: {self.type} = call {self.func} {" ".join(self.args)}"


@dataclass
class Nop(EffectOperation):
    def __init__(self):
        super().__init__(Opcode.NOP)

    def __str__(self) -> str:
        return f"nop"


@dataclass
class Print(EffectOperation):
    arg: str

    def __init__(self, arg):
        super().__init__(Opcode.PRINT, args=[arg])
        self.arg = arg

    def __str__(self) -> str:
        return f"print {self.arg}"


@dataclass
class Jmp(EffectOperation):
    target: str

    def __init__(self, label):
        super().__init__(Opcode.JMP, labels=[label])
        self.target = label

    def __str__(self) -> str:
        return f"jmp .{self.target}"


@dataclass
class Br(EffectOperation):

    def __init__(self, cond, true_label, false_label):
        super().__init__(Opcode.BR, args=[cond], labels=[true_label, false_label])


@dataclass
class Ret(EffectOperation):
    value: str

    def __init__(self, value=None):
        super().__init__(Opcode.RET, args=[value] if value else [])
        self.value = value

    def __str__(self) -> str:
        return f"ret {self.value}"


@dataclass
class Label(Instruction):

    def __init__(self, name):
        super().__init__(None, None)
        self.name = name

    def __str__(self):
        return f"label {self.name}"


@dataclass
class Function:
    name: str
    return_type: str
    params: List[str]
    instructions: List[Instruction]

    def __init__(self, name, return_type, params, instructions):
        self.name = name
        self.return_type = return_type
        self.params = params
        self.instructions = instructions

    def __init__(self, ast):
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
                    case "lnot":
                        instruction = Lnot(inst["dest"], inst["type"], inst["args"][0])
                    case "land":
                        instruction = Land(
                            inst["dest"], inst["type"], inst["args"][0], inst["args"][1]
                        )
                    case "lor":
                        instruction = Lor(
                            inst["dest"], inst["type"], inst["args"][0], inst["args"][1]
                        )
                    case "jmp":
                        instruction = Jmp(inst["labels"][0])
                    case "call":
                        instruction = Call(
                            inst["dest"], inst["type"], inst["funcs"][0], inst["args"]
                        )
                    case "br":
                        instruction = Br(
                            inst["args"], inst["labels"][0], inst["labels"][1]
                        )
                    case "print":
                        instruction = Print(inst["args"][0])
                    case "ret":
                        instruction = Ret(inst["args"][0])
                    case _:
                        raise Exception("unimplemented instruction for {}".format(inst))
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
            + "\n".join("   " + str(instr) for instr in self.instructions)
            + "\n"
            + "}"
        )


class Program:
    functions: List[Function] = []

    def __init__(self, ast) -> None:
        """
        Parse an AST from JSON format to a list of instructions.
        """
        for function in ast["functions"]:
            func = Function(function)
            self.functions.append(func)


class BasicBlock:
    def __init__(self, label, instructions):
        self.label = label
        self.instructions:List[Instruction] = instructions
        self.successors:List[BasicBlock] = []
        self.predecessors:List[BasicBlock] = []


class ControlFlowGraph:
    def __init__(self, function):
        self.function: Function= function
        self.basic_blocks:List[BasicBlock] = []
        self.build()

    def build(self):
        # Split instructions into basic blocks
        blocks = []
        current_block = []
        for instr in self.function.instructions:
            if instr.is_terminator():
                if current_block:
                    blocks.append(current_block)
                    current_block = []
            current_block.append(instr)
        if current_block:
            blocks.append(current_block)

        # Create BasicBlock objects
        for i, block in enumerate(blocks):
            label = f"block_{i}"
            self.basic_blocks.append(BasicBlock(label, block))

        # Find successors and predecessors
        for i, block in enumerate(self.basic_blocks):
            if i < len(self.basic_blocks) - 1:
                block.successors.append(self.basic_blocks[i + 1])
                self.basic_blocks[i + 1].predecessors.append(block)
            for instr in block.instructions:
                if isinstance(instr, Jmp):
                    target_label = instr.args[0]
                    for target_block in self.basic_blocks:
                        if target_block.label == target_label:
                            block.successors.append(target_block)
                            target_block.predecessors.append(block)
                            break
                elif isinstance(instr, Br):
                    true_label = instr.args[1]
                    false_label = instr.args[2]
                    for target_block in self.basic_blocks:
                        if target_block.label == true_label:
                            block.successors.append(target_block)
                            target_block.predecessors.append(block)
                        elif target_block.label == false_label:
                            block.successors.append(target_block)
                            target_block.predecessors.append(block)

    def __str__(self):
        dot_str = "digraph cfg {\n"
        for block in self.basic_blocks:
            dot_str += f'  {block.label} [label="'
            for instr in block.instructions:
                dot_str += str(instr) + "\\n"
            dot_str += '"];\n'
            for successor in block.successors:
                dot_str += f"  {block.label} -> {successor.label};\n"
        dot_str += "}"
        return dot_str

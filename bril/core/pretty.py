"""
Implementation of text format prettifiers and printer for Bril programs.
"""

def type_to_str(type):
    if isinstance(type, dict):
        assert len(type) == 1
        key, value = next(iter(type.items()))
        return '{}<{}>'.format(key, type_to_str(value))
    else:
        return type


def value_to_str(type, value):
    if not isinstance(type, dict) and type.lower() == "char":
        control_chars_reverse = {y: x for x, y in control_chars.items()}
        if ord(value) in control_chars_reverse:
            value = control_chars_reverse[ord(value)]
        return "'{}'".format(value)
    else:
        return str(value).lower()


def instr_to_string(instr):
    if instr['op'] == 'const':
        tyann = ': {}'.format(type_to_str(instr['type'])) \
            if 'type' in instr else ''
        return '{}{} = const {}'.format(
            instr['dest'],
            tyann,
            value_to_str(instr['type'], instr['value']),
        )
    else:
        rhs = instr['op']
        if instr.get('funcs'):
            rhs += ' {}'.format(' '.join(
                '@{}'.format(f) for f in instr['funcs']
            ))
        if instr.get('args'):
            rhs += ' {}'.format(' '.join(instr['args']))
        if instr.get('labels'):
            rhs += ' {}'.format(' '.join(
                '.{}'.format(f) for f in instr['labels']
            ))
        if 'dest' in instr:
            tyann = ': {}'.format(type_to_str(instr['type'])) \
                if 'type' in instr else ''
            return '{}{} = {}'.format(
                instr['dest'],
                tyann,
                rhs,
            )
        else:
            return rhs


def print_instr(instr):
    print('  {};'.format(instr_to_string(instr)))


def print_label(label):
    print('.{}:'.format(label['label']))


def args_to_string(args):
    if args:
        return '({})'.format(', '.join(
            '{}: {}'.format(arg['name'], type_to_str(arg['type']))
            for arg in args
        ))
    else:
        return ''


def print_func(func):
    typ = func.get('type', 'void')
    print('@{}{}{} {{'.format(
        func['name'],
        args_to_string(func.get('args', [])),
        ': {}'.format(type_to_str(typ)) if typ != 'void' else '',
    ))
    for instr_or_label in func['instrs']:
        if 'label' in instr_or_label:
            print_label(instr_or_label)
        else:
            print_instr(instr_or_label)
    print('}')


def print_prog(prog):
    for func in prog['functions']:
        print_func(func)
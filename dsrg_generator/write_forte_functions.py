import os
from dsrg_generator.helper.file_utils import multi_gsub
from dsrg_generator.phys_op_contraction import categorize_contractions


def save_terms_ambit_functions(input_terms, func_name, namespace, path_dir, add_t_dagger=True, destroy_h=False):
    """
    Write ambit functions in forte using ambit_template in the forte_templates folder.
    :param input_terms: a list of terms
    :param func_name: the name of major function
    :param namespace: the class name of the functions, used as C++ headers as well
    :param path_dir: the directory where all generated functions will be saved to
    :param add_t_dagger: add Hermitian conjugate of the results at the end of the function
    :param destroy_h: use H1, H2, ... as temp intermediates to add Hermitian conjugate
    """
    cat_cons = categorize_contractions(input_terms)

    tensor_ordering = {f"H{i}": i for i in range(1, 10)}
    tensor_ordering.update({f"T{i}": i + 100 for i in range(1, 10)})
    tensor_ordering.update({f"C{i}": i + 200 for i in range(10)})
    tensor_types = {i: 'double' if i == 'C0' else 'BlockedTensor' for i in tensor_ordering}

    func_calls = []
    func_declarations = []
    func_tensors = set()
    func_targets = set()

    for block in cat_cons.keys():
        t, c, call, declare = save_terms_blocks_ambit_function(cat_cons[block], block, tensor_ordering, func_name,
                                                               namespace, path_dir)
        func_tensors.update(t)
        func_targets.add(c)
        func_calls.append(call)
        func_declarations.append(declare)

    if any(f'H{i[1:]}' not in func_tensors for i in func_targets if i != 'C0'):
        destroy_h = False

    func_tensors = sorted(func_tensors, key=lambda x: tensor_ordering[x])
    func_targets = sorted(func_targets, key=lambda x: tensor_ordering[x])
    footprint = f'double factor, {", ".join(f"{tensor_types[i]}& {i}" for i in func_tensors + func_targets)}'

    indent = '\n    '
    func = f'void {namespace}::{func_name}({footprint}) {{' + indent

    prefix = []
    scale = []
    for i in func_targets:
        if i == 'C0':
            prefix.append('C0 = 0.0;')
            scale.append('C0 *= factor;')
        else:
            prefix.append(f'{i}.zero();')
            scale.append(f'{i}.scale(factor);')

    suffix = []
    if add_t_dagger:
        for i in func_targets:
            n_body = int(i[1:])
            upper = ','.join(f'g{i}' for i in range(n_body))
            lower = ','.join(f'g{i}' for i in range(n_body, 2 * n_body))

            if i == 'C0':
                suffix.append('C0 *= 2.0;')
                if not destroy_h:
                    suffix.append('BlockedTensor temp;')
            else:
                name = f'H{n_body}'
                if not destroy_h:
                    b = 'g' * (2 * n_body)
                    suffix.append(f'temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {{"{b}"}});')
                    name = 'temp'

                suffix.append(f'{name}["{upper},{lower}"] = {i}["{upper},{lower}"];')
                suffix.append(f'{i}["{upper},{lower}"] += {name}["{lower},{upper}"];')

    func += indent.join(prefix) + '\n' + indent
    func += indent.join(func_calls) + '\n'
    if suffix:
        func += indent
    func += indent.join(suffix)
    func += '\n}'

    filename = f'{path_dir}/{func_name}.cc'
    template = open(os.path.dirname(os.path.abspath(__file__)) + '/forte_templates/ambit_template').read()
    input_string = multi_gsub({"HEADERS": f'#include {namespace}.h'.lower(), "CPP_FUNCTIONS": func}, template)
    with open(filename, 'w') as f:
        f.write(input_string)

    filename = f'{path_dir}/{func_name}_functions.h'
    with open(filename, 'w') as f:
        f.write(f'void {func_name}({footprint});\n')
        f.write('\n'.join(func_declarations))


def save_terms_blocks_ambit_function(perm_terms, block, tensor_ordering, func_name, namespace, path_dir):
    """
    Write ambit functions in forte using ambit_template in the forte_templates folder for each space block.
    :param perm_terms: a map from permutation to a list of terms
    :param block: a string for the space block of given perm_terms
    :param tensor_ordering: a map for tensor ordering, e.g., {'H1': 1, 'H2': 2, 'C0': 10, ...}
    :param func_name: the name of major function for writing to disk
    :param namespace: the class name where func_name belongs
    :param path_dir: the directory where the generated function will be saved to
    :return: {contracted tensor names}, target tensor name, string for function call, string for declaration
    """
    target = f"C{len(block) // 2}"
    if block == '':
        block = '0'

    tensors = set()
    initiate_temp = True

    out = ""
    indent = "\n    "

    for perm, terms in perm_terms.items():
        for term in terms:
            for tensor in term.list_of_tensors:
                tensors.add(f"{tensor.name}{tensor.n_body}")

        i_last = len(terms) - 1
        for i, term in enumerate(terms):
            ambit = ""
            if perm and i == 0:
                if initiate_temp:
                    initiate_temp = False
                    ambit = f'temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {{"{block}"}});\n'
                else:
                    ambit = 'temp.zero();\n'

            ambit += term.ambit(ignore_permutations=(i != i_last), init_temp=False, declared_temp=True)
            ambit = indent.join(ambit.split('\n'))
            out += indent + ambit.strip()
        out += '\n'

    tensors = sorted(tensors, key=lambda x: tensor_ordering[x])
    func_footprint = ", ".join(f"BlockedTensor& {i}" for i in tensors)
    func_footprint += ', double& C0' if target == 'C0' else f', BlockedTensor& {target}'

    func = f"void {namespace}::{func_name}_{block}({func_footprint}) {{"
    if not initiate_temp:
        func += indent + "BlockedTensor temp;\n"

    out = func + out + '}'

    template = open(os.path.dirname(os.path.abspath(__file__)) + '/forte_templates/ambit_template').read()
    filename = f'{path_dir}/{func_name}_{block}.cc'
    input_string = multi_gsub({"HEADERS": f'#include {namespace}.h'.lower(), "CPP_FUNCTIONS": out}, template)
    with open(filename, 'w') as f:
        f.write(input_string)

    func_call = f"{func_name}_{block}({', '.join(tensors + [target])});"
    func_declare = f"void {func_name}_{block}({func_footprint});"

    return tensors, target, func_call, func_declare


def save_direct_t3(perm_terms, func_name, namespace, path_dir):
    """
    Write ambit functions in forte for T3 amplitudes using direct_t3_template in the forte_templates folder.
    :param perm_terms: a map from permutation to a list of terms
    :param func_name: the name of major function for writing to disk
    :param namespace: the class name where func_name belongs
    :param path_dir: the directory where the generated function will be saved to
    """
    tensors = set()

    cpp_func = ''
    indent = '\n    '

    for perm, terms in perm_terms.items():
        for term in terms:
            for tensor in term.list_of_tensors:
                tensors.add(f"{tensor.name}{tensor.n_body}")

        i_last = len(terms) - 1
        for i, term in enumerate(terms):
            ambit = ""
            if perm and i == 0:
                ambit = 'temp.zero();\n'

            ambit += term.ambit(ignore_permutations=(i != i_last), init_temp=False, declared_temp=True)
            ambit = indent.join(ambit.split('\n'))
            cpp_func += indent + ambit.strip()
        cpp_func += '\n'

    tensors = sorted(tensors)
    func_vars = ", ".join(f"BlockedTensor& {i}" for i in tensors)

    template = open(os.path.dirname(os.path.abspath(__file__)) + '/forte_templates/direct_t3_template').read()
    filename = f'{path_dir}/{func_name}_direct_t3.cc'
    input_string = multi_gsub({"HEADERS": f'#include {namespace}.h'.lower(), "NAMESPACE": namespace,
                               "FUNC_NAME": func_name, "FUNC_VARIABLES": func_vars, "CPP_EXPRESSIONS": cpp_func},
                              template)
    with open(filename, 'w') as f:
        f.write(input_string)

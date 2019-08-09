from dsrg_generator.helper.file_utils import multi_gsub
from dsrg_generator.phys_op_contraction import categorize_contractions


def save_terms_ambit_functions(input_terms, func_name, path_dir, template, namespace="MRDSRG_SO"):
    out_terms = categorize_contractions(input_terms)

    tensor_ordering = {f"H{i}": i for i in range(10)}
    tensor_ordering.update({f"T{i - 100}": i for i in range(100, 110)})
    tensor_ordering.update({f"C{i - 1000}": i for i in range(1000, 1010)})
    types = {i: 'BlockedTensor&' if '0' not in i else 'double&' for i in tensor_ordering}

    func_calls = []
    footprints = []
    func_tensors = set()

    for block in out_terms.keys():
        block_name = '0' if block == '' else block

        func_str, func_call, tensors, func_footprint = terms_ambit_block(out_terms[block], block_name, tensor_ordering, func_name, namespace)
        func_calls.append(func_call)
        func_tensors.update(tensors)
        footprints.append(func_footprint)

        filename = f'{path_dir}/{func_name}_{block_name}.cc'
        input_string = multi_gsub({"HEADERS": f'#include {namespace}.h'.lower(), "CPP_FUNCTIONS": func_str},
                                  template)
        with open(filename, 'w') as f:
            f.write(input_string)
        # print(func_str)

    func_tensors = sorted(func_tensors, key=lambda x: tensor_ordering[x])
    func_tensors_str = ", ".join(f"{types[i]} {i}" for i in func_tensors)
    footprints.append(f'void {func_name}(double factor, {func_tensors_str});')
    func = f"void {namespace}::{func_name}(double factor, {func_tensors_str}) {{\n    "

    c = [i for i in func_tensors if 'C' in i and '0' not in i]

    prefix = [] if 'C0' not in func_tensors else ['C0 = 0.0;']
    for i in c:
        prefix.append(f'{i}.zero();')

    func += "\n    ".join(prefix) + '\n\n    '

    func += "\n    ".join(func_calls) + '\n\n    '

    suffix = [] if 'C0' not in func_tensors else ['C0 *= 2.0;']
    if c:
        suffix.append('BlockedTensor temp;')
    for i in c:
        n_body = int(i[1:])
        upper = ','.join(f'g{i}' for i in range(n_body))
        lower = ','.join(f'g{i}' for i in range(n_body, 2 * n_body))
        b = 'g' * (2 * n_body)
        suffix.append(f'temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {{"{b}"}});')
        suffix.append(f'temp[{upper},{lower}] = {i}[{upper},{lower}];')
        suffix.append(f'{i}[{upper},{lower}] += temp[{lower},{upper}];')
    func += "\n    ".join(suffix) + '\n}'

    filename = f'{path_dir}/{func_name}.cc'
    input_string = multi_gsub({"HEADERS": f'#include {namespace}.h'.lower(), "CPP_FUNCTIONS": func}, template)
    with open(filename, 'w') as f:
        f.write(input_string)
    # print(func)

    filename = f'{path_dir}/{func_name}_append.h'
    with open(filename, 'w') as f:
        f.write('\n'.join(footprints))
    # print('\n'.join(footprints))


def terms_ambit_block(perm_terms, block, tensor_ordering, func_name, namespace):
    target = f"C{len(block) // 2}"
    do_temp = False
    tensors = set()
    out = ""

    init_temp = True

    for perm, terms in perm_terms.items():
        if perm:
            do_temp = True

        for term in terms:
            for tensor in term.list_of_tensors:
                tensors.add(f"{tensor.name}{tensor.n_body}")

        i_last = len(terms) - 1
        for i, term in enumerate(terms):
            ambit = ""
            if perm and i == 0:
                if init_temp:
                    init_temp = False
                    ambit = f'temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {{"{block}"}});\n'
                else:
                    ambit = 'temp.zero();\n'

            ambit += term.ambit(ignore_permutations=(i != i_last), init_temp=False, declared_temp=True)
            ambit = "\n    ".join(ambit.split('\n'))
            out += '\n    ' + ambit
        if perm == '':
            out += '\n'

    tensors = sorted(tensors, key=lambda x: tensor_ordering[x])
    tensors_str = ", ".join(f"BlockedTensor& {i}" for i in tensors)
    tensors_str += ', double& C0' if target == 'C0' else f', BlockedTensor& {target}'

    func_call = f"{func_name}_{block}({tensors_str})"
    func = f"void {namespace}::{func_call} {{"

    if do_temp:
        func += "\n    BlockedTensor temp;\n"
    if out[-1] != '\n':
        out = out[:-4]

    out = func + out + '}\n'

    return out, f"{func_name}_{block}({', '.join(tensors + [target])});", tensors + [target], f"void {func_call};"

    # # figure out levels of Hamiltonian, ClusterAmplitudes, and Cumulants for each block
    # tensor_levels = {}
    # total_levels = [set(), set(), set()]
    # for block in out_terms.keys():
    #     H_levels, T_levels = set(), set()
    #     for perm, terms in out_terms[block].items():
    #         for term in terms:
    #             for tensor in term.list_of_tensors:
    #                 n_body = tensor.n_body
    #                 if isinstance(tensor, Hamiltonian):
    #                     H_levels.add(n_body)
    #                 elif isinstance(tensor, ClusterAmplitude):
    #                     T_levels.add(n_body)
    #                 else:
    #                     continue
    #     tensor_levels[block] = [H_levels, T_levels]
    #     total_levels[0].union(H_levels)
    #     total_levels[1].union(T_levels)
    #     total_levels[2].add(len(block) // 2)
    #
    # title_H = ",".join([f"BlockedTensor& H{i}" for i in sorted(total_levels[0])])
    # title_T = ",".join([f"BlockedTensor& T{i}" for i in sorted(total_levels[1])])
    # C0 = 0 in total_levels[2]
    # Cn = [f"C{i}" for i in sorted(total_levels[2]) if i != 0]
    # title_C = "" if not C0 else "double& C0, "
    # title_C += ",".join([f"BlockedTensor& {i}" for i in Cn])
    # resetC = "" if not C0 else "C0 = 0.0;\n"
    # resetC += "\n".join([f"{i}.zero();" for i Cn])
    # prefix = f"""void {class_name}::{func_name}(double factor, {title_H}, {title_T}, {title_C}) {{
    # {resetC}
    # BlockedTensor temp;""" + "\n" * 2
    #
    # scaleC = "" if not C0 else "C0 *= factor;\n"
    # scaleC += "\n".join([f"{i}.scale(factor);" for i in Cn])
    # addC = "" if not C0 else "C0 *= 2.0;\n"
    # addC +=

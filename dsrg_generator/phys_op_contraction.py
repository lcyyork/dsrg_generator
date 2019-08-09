import multiprocessing
from threading import Thread
from copy import deepcopy
from collections import defaultdict, Iterable, deque
from itertools import combinations, product
from math import factorial, sqrt
from timeit import default_timer as timer
from sympy.utilities.iterables import multiset_permutations
from sympy.physics.quantum import Operator, HermitianOperator, Commutator, Dagger
from sympy.core.power import Pow
from sympy.core.add import Add
from sympy.core.mul import Mul

from dsrg_generator.helper.integer_partition import integer_partition
from dsrg_generator.mo_space import space_relation, space_priority
from dsrg_generator.Indices import Indices
from dsrg_generator.helper.multiprocess_helper import calculate_star
from dsrg_generator.SQOperator import SecondQuantizedOperator
from dsrg_generator.Tensor import ClusterAmplitude, HamiltonianTensor, Cumulant
from dsrg_generator.Term import Term, cluster_operator, hamiltonian_operator
from dsrg_generator.sqop_contraction import compute_operator_contractions
# from sqop_contraction import generate_operator_contractions, generate_operator_contractions_new
from dsrg_generator.helper.Timer import Timer


def multiprocessing_canonicalize_contractions(tensors, sq_op, coeff):
    return Term(tensors, sq_op, coeff).canonicalize_sympy()


def contract_terms(terms, max_cu=3, max_n_open=6, min_n_open=0, scale_factor=1.0,
                   for_commutator=False, expand_hole=True, n_process=1):
    """
    Contract a list of Term objects.
    :param terms: a list of Term objects to be contracted
    :param max_cu: max level of cumulant
    :param max_n_open: max number of open indices for contractions kept for return
    :param min_n_open: min number of open indices for contractions kept for return
    :param scale_factor: a scaling factor for the results
    :param for_commutator: compute only non-zero terms for commutators if True
    :param expand_hole: expand HoleDensity to Kronecker - Cumulant if True
    :param n_process: number of processes launched for tensor canonicalization
    :return: a list of contracted and canonicalized Term objects
    """
    if len(terms) == 0:
        raise ValueError("size of terms cannot be zero.")

    n_process = min(n_process, multiprocessing.cpu_count())

    coeff = float(scale_factor)
    tensors = []
    sq_ops_to_be_contracted = []
    for term in terms:
        if not isinstance(term, Term):
            raise TypeError(f"{term} if not of Term type.")
        coeff *= term.coeff
        tensors += term.list_of_tensors
        if not term.sq_op.is_empty():
            sq_ops_to_be_contracted.append(term.sq_op)

    if len(sq_ops_to_be_contracted) < 2:
        sq_op = sq_ops_to_be_contracted[0] if len(sq_ops_to_be_contracted) == 1 else terms[0].sq_op
        return [Term(tensors, sq_op, coeff)]
    else:
        # contractions = [i for con in compute_operator_contractions(sq_ops_to_be_contracted, max_cu,
        #                                                            max_n_open, min_n_open, for_commutator,
        #                                                            expand_hole, n_process, batch_size=0)
        #                 for i in con]
        # chunk_size = int(sqrt(len(contractions)) / n_process) + 1
        # out = canonicalize_contractions_batch(n_process, contractions, tensors, coeff, False, chunk_size)
        # return combine_terms(out)

        count = 0
        out = []
        contractions = []
        batch_size = 10000 * max(1, n_process // 2)
        chunk_size = int(sqrt(batch_size) / n_process) + 1

        for batch in compute_operator_contractions(sq_ops_to_be_contracted, max_cu, max_n_open, min_n_open,
                                                   for_commutator, expand_hole, 1, batch_size=0):
            count += len(batch)
            contractions += batch
            print(count, len(contractions))

            if count < batch_size:
                continue

            out += canonicalize_contractions_batch(n_process, contractions, tensors, coeff, True, chunk_size)
            contractions = []
            count = 0

        chunk_size = int(sqrt(len(contractions)) / n_process) + 1
        out += canonicalize_contractions_batch(n_process, contractions, tensors, coeff, False, chunk_size)

        return combine_terms(out)


def canonicalize_contractions_batch(n_process, contractions, tensors, coeff, simplify, chunk_size):
    """
    Canonicalize a batch of contractions
    :param n_process: number of processes launched for tensor canonicalization
    :param contractions: a list of contractions [(sign, densities, sq_op)]
    :param tensors: a list of tensors
    :param coeff: a scale factor for all contractions
    :param simplify: combine similar terms if True
    :param chunk_size: the chunk size for multiprocessing
    :return: a list of simplified canonicalized terms
    """
    if len(contractions) == 0:
        return []

    print("in func", len(contractions))
    if n_process == 1:
        temp = [Term(tensors + densities, sq_op, sign * coeff).canonicalize_sympy()
                for sign, densities, sq_op in contractions]
    else:
        with multiprocessing.Pool(n_process, maxtasksperchild=1000) as pool:
            tasks = [(multiprocessing_canonicalize_contractions, (tensors + densities, sq_op, sign * coeff))
                     for sign, densities, sq_op in contractions]
            imap_unordered_it = pool.imap_unordered(calculate_star, tasks, chunksize=chunk_size)
            temp = [x for x in imap_unordered_it]
    return combine_terms(temp) if simplify else temp


def combine_terms(terms):
    """
    Simplify the list of terms by combining similar terms.
    :param terms: a list of canonicalized Term objects
    :return: a list of simplified Term objects
    """
    if not isinstance(terms, list):
        raise ValueError("terms should be a list of Term objects.")

    if len(terms) == 0:
        return []

    terms_to_coeff = defaultdict(list)
    terms_dict = {}

    for term in terms:
        name = term.hash_term()
        terms_dict[name] = term
        terms_to_coeff[name].append(term.coeff)

    out = []
    for name, term in terms_dict.items():
        term.coeff = sum(terms_to_coeff[name])
        if not term.is_void():
            out.append(term)
    return sorted(out)


def single_commutator(left, right, max_cu=3, max_n_open=6, min_n_open=0,
                      scale_factor=1.0, for_commutator=True, expand_hole=True, n_process=1):
    """
    Compute a single commutator of the form [left, right].
    :param left: left term
    :param right: right term
    :param max_cu: max level of cumulant
    :param max_n_open: max number of open indices for contractions of each single commutator
    :param min_n_open: min number of open indices for contractions of each single commutator
    :param scale_factor: a scaling factor for the results
    :param for_commutator: compute only non-zero terms for commutators if True
    :param expand_hole: expand HoleDensity to (Kronecker - Cumulant) if True
    :param n_process: number of processes launched for tensor canonicalization
    :return: a list of contracted canonicalized Term objects
    """
    if for_commutator and (left.sq_op.is_empty() or right.sq_op.is_empty()):
        return []

    terms = contract_terms([left, right], max_cu, max_n_open, min_n_open, scale_factor,
                           for_commutator, expand_hole, n_process) \
        + contract_terms([right, left], max_cu, max_n_open, min_n_open, -scale_factor,
                         for_commutator, expand_hole, n_process)
    return combine_terms(terms)


def cluster_operators(levels, start, unitary=True, single_reference=True):
    """
    Return a list of cluster operators.
    :param levels: a list of integers for cluster operator, e.g., [1,2,3] for T1 + T2 + T3
    :param start: starting number of indices
    :param unitary: consider A = T - T^+ if True
    :param single_reference: use single-reference labels if True
    :return: a list of Term objects
    """
    h = 'c' if single_reference else 'h'
    p = 'v' if single_reference else 'p'

    amps = [cluster_operator(k, hole_label=h, particle_label=p, start=start) for k in levels]
    if unitary:
        amps += [cluster_operator(k, excitation=False, scale_factor=-1.0,
                                  hole_label=h, particle_label=p, start=start)
                 for k in levels]
    return amps


def linear_commutator_amp(left_terms, cluster_levels, scale_factor, min_n_open, max_n_open, start,
                          n_process=1, unitary=True, max_cu=3, for_commutator=True, single_reference=True):
    """
    Compute the commutator between the input list of terms and the cluster operators [left_term, T].
    :param left_terms: a list of Terms appear at the first entry of the commutator
    :param cluster_levels: a list of integers for cluster operator, e.g., [1,2,3] for T1 + T2 + T3
    :param scale_factor: the scale factor for the commutator
    :param min_n_open: min number of open indices for contractions of each single commutator
    :param max_n_open: max number of open indices for contractions of each single commutator
    :param start: starting number of indices for cluster amplitudes
    :param n_process: the number of process for multiprocessing
    :param unitary: consider A = T - T^+ if True
    :param max_cu: the max cumulant level
    :param for_commutator: ignore disconnected terms if True
    :param single_reference: use single-reference labels if True
    :return: a list of contracted Term objects
    """
    return linear_commutator(left_terms, cluster_operators(cluster_levels, start, unitary, single_reference),
                             scale_factor, min_n_open, max_n_open, max_cu, n_process, for_commutator)


def linear_commutator(left_terms, right_terms, scale_factor, min_n_open, max_n_open,
                      max_cu=3, n_process=1, for_commutator=True):
    """
    Compute the commutator between left and right terms.
    :param left_terms: a list of Term objects at the first entry of the commutator
    :param right_terms: a list of Term objects at the second entry of the commutator
    :param scale_factor: the scale factor for the commutator
    :param min_n_open: min number of open indices for contractions of each single commutator
    :param max_n_open: max number of open indices for contractions of each single commutator
    :param max_cu: the max cumulant level
    :param n_process: the number of process for multiprocessing
    :param for_commutator: ignore disconnected terms if True
    :return: a list of contracted Term objects
    """
    out = []
    for left in left_terms:
        for right in right_terms:
            out += single_commutator(left, right, max_cu, max_n_open, min_n_open, scale_factor,
                                     for_commutator, n_process=n_process)
    return combine_terms(out)


def recursive_single_commutator(terms, max_cu_levels, n_opens, for_commutator=True, expand_hole=True, n_process=1):
    """
    Compute nested commutators using recursive single commutator formalism.
    :param terms: a list of terms, computed as [[...[[term_0, term_1], term_2], ...], term_k]
    :param max_cu_levels: a list of integers for max cumulant level of each level of commutator
    :param n_opens: a list of tuple [(min, max)] for numbers of open indices of each level of commutator
    :param for_commutator: compute only non-zero terms for commutators if True
    :param expand_hole: expand HoleDensity to (Kronecker - Cumulant) if True
    :param n_process: number of processes launched for tensor canonicalization
    :return: a map of nested level to a list of contracted canonicalized Term objects
    """
    n_terms = len(terms)
    if n_terms < 2:
        raise ValueError("Need to have at least two terms for a valid commutator.")

    if isinstance(max_cu_levels, int):
        max_cu_levels = [max_cu_levels] * (n_terms - 1)
    if len(max_cu_levels) != n_terms - 1:
        raise ValueError(f"Inconsistent size of max_cu_levels ({max_cu_levels}), required {n_terms - 1}")
    if any(not isinstance(i, int) for i in max_cu_levels):
        raise ValueError(f"Invalid max_cu_levels ({max_cu_levels}): not all integers")

    if isinstance(n_opens, tuple):
        n_opens = [n_opens] * (n_terms - 1)
    if len(n_opens) != n_terms - 1:
        raise ValueError(f"Inconsistent size of n_opens ({n_opens}), required {n_terms - 1}")
    for n in n_opens:
        if len(n) != 2:
            raise ValueError(f"Invalid element in n_opens: {n} cannot be used for min/max numbers of open indices.")
        if not (isinstance(n[0], int) and isinstance(n[1], int)):
            raise ValueError(f"Invalid element in n_opens: {n} contains non-integer elements")

    left_pool = [terms[0]]

    out = defaultdict(list)

    for i in range(1, n_terms):
        right = terms[i]

        max_cu = max_cu_levels[i - 1]
        min_n_open, max_n_open = n_opens[i - 1]

        for left in left_pool:
            out[i] += single_commutator(left, right, max_cu, max_n_open, min_n_open,
                                        1.0, for_commutator, expand_hole, n_process)

        out[i] = combine_terms(out[i])
        left_pool = out[i]

    return out


def bch_cc_rsc(nested_level, cluster_levels, max_cu_levels, n_opens, for_commutator=True,
               expand_hole=True, single_reference=False, unitary=False, n_process=1):
    """
    Compute the BCH nested commutator in coupled cluster theory using recursive commutator formalism.
    :param nested_level: the level of nested commutator
    :param cluster_levels: a list of integers for cluster operator, e.g., [1,2,3] for T1 + T2 + T3
    :param max_cu_levels: a list of integers for max cumulant level of each level of commutator
    :param n_opens: a list of tuple [(min, max)] for numbers of open indices of each level of commutator
    :param for_commutator: compute only non-zero terms for commutators if True
    :param expand_hole: expand HoleDensity to Kronecker minus Cumulant if True
    :param single_reference: use single-reference amplitudes if True
    :param unitary: use unitary formalism if True
    :param n_process: number of processes launched for tensor canonicalization
    :return: a map of nested level to a list of contracted canonicalized Term objects
    """
    if not isinstance(nested_level, int):
        raise ValueError("Invalid nested_level (must be an integer)")

    if not all(isinstance(t, int) for t in cluster_levels):
        raise ValueError("Invalid content in cluster_operator (must be all integers)")

    if isinstance(max_cu_levels, int):
        max_cu_levels = [max_cu_levels] * nested_level
    if len(max_cu_levels) != nested_level:
        raise ValueError(f"Inconsistent size of max_cu_levels ({max_cu_levels}), required {nested_level}")
    if any(not isinstance(i, int) for i in max_cu_levels):
        raise ValueError(f"Invalid max_cu_levels ({max_cu_levels}): not all integers")

    if isinstance(n_opens, tuple):
        n_opens = [n_opens] * nested_level
    if len(n_opens) != nested_level:
        raise ValueError(f"Inconsistent size of n_opens ({n_opens}), required {nested_level}")
    for n in n_opens:
        if len(n) != 2:
            raise ValueError(f"Invalid element in n_opens: {n} cannot be used for min/max numbers of open indices.")
        if not (isinstance(n[0], int) and isinstance(n[1], int)):
            raise ValueError(f"Invalid element in n_opens: {n} contains non-integer elements")

    max_amp = max(cluster_levels) * 2
    amps = [cluster_operators(cluster_levels, start=(i * max_amp), unitary=unitary, single_reference=single_reference)
            for i in range(nested_level)]

    out = defaultdict(list)

    left_pool = [hamiltonian_operator(1), hamiltonian_operator(2)]

    for i in range(1, nested_level + 1):
        factor = 1.0 / i

        max_cu = max_cu_levels[i - 1]
        min_n_open, max_n_open = n_opens[i - 1]

        for left in left_pool:
            for right in amps[i - 1]:
                out[i] += single_commutator(left, right, max_cu, max_n_open, min_n_open,
                                            factor, for_commutator, expand_hole, n_process)

        out[i] = combine_terms(out[i])
        left_pool = out[i]

    return out


def sympy_nested_commutator_recursive(level, a, b):
    """
    Compute nested commutator of type [[...[[A, B], B], ...], B]
    :param level: the level of nested commutator
    :param a: Operator A
    :param b: Operator B
    :return: commutator of type Add
    """
    if level <= 1:
        return Commutator(a, b)
    for i in range(level)[::-1]:
        return Commutator(sympy_nested_commutator_recursive(i, a, b), b)


def nested_commutator_cc(nested_level, cluster_levels, max_cu=3, max_n_open=6, min_n_open=0,
                         for_commutator=True, expand_hole=True, single_reference=False, unitary=False, n_process=1):
    """
    Compute the BCH nested commutator in coupled cluster theory.
    :param nested_level: the level of nested commutator
    :param cluster_levels: a list of integers for cluster operator, e.g., [1,2,3] for T1 + T2 + T3
    :param max_cu: max value of cumulant allowed for contraction
    :param max_n_open: the max number of open indices for contractions kept for return
    :param min_n_open: the min number of open indices for contractions kept for return
    :param for_commutator: compute only non-zero terms for commutators if True
    :param expand_hole: expand HoleDensity to Kronecker minus Cumulant if True
    :param single_reference: use single-reference amplitudes if True
    :param unitary: use unitary formalism if True
    :param n_process: number of processes launched for tensor canonicalization
    :return: a list of contracted canonicalized Term objects
    """
    if not isinstance(nested_level, int):
        raise ValueError("Invalid nested_level (must be an integer)")
    if not isinstance(cluster_levels, Iterable):
        raise ValueError("Invalid type for cluster_operator")
    if not all(isinstance(t, int) for t in cluster_levels):
        raise ValueError("Invalid content in cluster_operator (must be all integers)")

    scale_factor = 1.0/factorial(nested_level)
    out = []

    hole_label = 'c' if single_reference else 'h'
    particle_label = 'v' if single_reference else 'p'

    # symbolic evaluate nested commutator
    t = sum(Operator(f'T{i}') for i in cluster_levels)
    h = HermitianOperator('H1') + HermitianOperator('H2')
    a = t - Dagger(t) if unitary else t

    for term in sympy_nested_commutator_recursive(nested_level, h, a).doit().expand().args:
        coeff, tensors = term.as_coeff_mul()
        factor = scale_factor * int(coeff)

        tensor_names = []
        for tensor in tensors:
            if isinstance(tensor, Pow):
                if isinstance(tensor.base, Dagger):
                    tensor_names += ['X' + str(tensor.base.args[0])[1:]] * int(tensor.exp)
                else:
                    tensor_names += [str(tensor.base)] * int(tensor.exp)
            else:
                if isinstance(tensor, Dagger):
                    tensor_names.append('X' + str(tensor.args[0])[1:])
                else:
                    tensor_names.append(str(tensor))

        list_of_terms = []
        start = 0
        for name in tensor_names:
            real_name, n_body = name[0], int(name[1:])
            if real_name == 'T' or real_name == 'X':
                list_of_terms.append(cluster_operator(n_body, start=start, excitation=(real_name == 'T'),
                                                      hole_label=hole_label, particle_label=particle_label))
                start += n_body
            else:
                list_of_terms.append(hamiltonian_operator(n_body))

        out += contract_terms(list_of_terms, max_cu, max_n_open, min_n_open, factor,
                              for_commutator, expand_hole, n_process)

    return combine_terms(out)


def categorize_contractions(terms):
    """
    Categorize contraction results to blocks of operators and permutations.
    :param terms: a list of terms computed from operator contractions
    :return: a dictionary {indices spaces: {permutation: list of terms}}
    """
    space_perm_repr = defaultdict(list)  # (space, permutation) being the key
    for term in terms:
        space_str = "".join([i.space for i in term.sq_op.indices()])
        n_perm, perm_str, sq_op_str = term.sq_op.latex_permute_format(*term.perm_partition_open())
        space_perm_repr[(space_str, perm_str)].append(term)

    out = {k[0]: {k[1]: []} for k in space_perm_repr.keys()}
    for k, terms in space_perm_repr.items():
        out[k[0]][k[1]] = sorted(terms)
    return out


def print_terms_ambit(input_terms):
    """
    Print contracted results in ambit format.
    :param input_terms: a list of terms computed from operator contractions
    """
    block_repr = categorize_contractions(input_terms)

    for block in block_repr.keys():
        for perm, terms in block_repr[block].items():
            i_last = len(terms) - 1
            for i, term in enumerate(terms):
                print(term.ambit(ignore_permutations=(i != i_last), init_temp=(i == 0), declared_temp=True))
            if perm == '':
                print()


def print_terms_latex(input_terms):
    """
    Print contracted results in latex format.
    :param input_terms: a list of terms computed from operator contractions
    """
    block_repr = categorize_contractions(input_terms)

    for block in block_repr.keys():
        print("\\begin{align}")
        for perm, terms in block_repr[block].items():
            i_last = len(terms) - 1
            for i, term in enumerate(terms):
                print(term.latex(delimiter=True, backslash=(i != i_last)))
        print("\\end{align}\n")

# def print_terms_ambit_symmetric(input_terms):
#     """
#     Print contracted results in ambit format assuming results are symmetric (i.e. block ccvv = block vvcc).
#     :param input_terms: a list of terms computed from operator contractions
#     """
#     block_repr = sort_contraction_results(input_terms)
#     sym_blocks = set()
#     for block_p, terms in sorted(block_repr.items(), key=lambda pair: (len(pair[0][0]), len(pair[0][1]), pair[0][0])):
#         block, perm = block_p
#         if block in sym_blocks:
#             continue
#         half1, half2 = block[:len(block)//2], block[len(block)//2:]
#         sym_blocks.add(half2 + half1)
#         scale = 0.5 if half1 == half2 else 1.0
#
#         i_last = len(terms) - 1
#         for i, term in enumerate(terms):
#             term.coeff *= scale
#             print(term.ambit(ignore_permutations=(i != i_last), init_temp=(i == 0), declared_temp=True))
#             term.coeff /= scale
#         if perm == '':
#             print()


# def save_terms_ambit(input_terms, full_path, func_name):
#     """
#     Save the contracted results to C++ file.
#     :param input_terms: a list of terms computed from operator contractions
#     :param full_path: the full path of the file name
#     :param func_name: the name of function
#     """
#     prefix = f"""void MRDSRG_SO::{func_name}(double factor, BlockedTensor& H1, BlockedTensor& H2,
#                                    BlockedTensor& T1, BlockedTensor& T2, double& C0, BlockedTensor& C1,
#                                    BlockedTensor& C2) {{
#     C0 = 0.0;
#     C1.zero();
#     C2.zero();
#     BlockedTensor temp;""" + "\n" * 2
#
#     suffix = """
#     // scale by factor
#     C0 *= factor;
#     C1.scale(factor);
#     C2.scale(factor);
#
#     // add T dagger
#     C0 *= 2.0;
#     temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"gg"});
#     temp["pq"] = C1["pq"];
#     C1["pq"] += temp["qp"];
#     temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"gggg"});
#     temp["pqrs"] = C2["pqrs"];
#     C2["pqrs"] += temp["rspq"];""" + "\n}"
#
#     with open(full_path, 'w') as w:
#         w.write(prefix)
#
#         block_repr = sort_contraction_results(input_terms)
#         sym_blocks = set()
#         for block_p, terms in sorted(block_repr.items(),
#                                      key=lambda pair: (len(pair[0][0]), len(pair[0][1]), pair[0][0])):
#             block, perm = block_p
#             if block in sym_blocks:
#                 continue
#             half1, half2 = block[:len(block) // 2], block[len(block) // 2:]
#             sym_blocks.add(half2 + half1)
#             scale = 0.5 if half1 == half2 else 1.0
#
#             i_last = len(terms) - 1
#             for i, term in enumerate(terms):
#                 term.coeff *= scale
#                 w.write(f'    {term.ambit(ignore_permutations=(i != i_last), init_temp=(i == 0), declared_temp=True)}\n')
#                 term.coeff /= scale
#             if perm == '':
#                 w.write('\n')
#
#         w.write(suffix)


# def print_results(results, form='latex'):
#     if form == 'latex':
#         print("\\begin{align}")
#         for n_open, terms in results.items():
#             print(f"% Open Indices {n_open}")
#             for term in terms:
#                 print(term.latex(delimiter=True, backslash=True))
#         print("\\end{align}")
#     else:
#         for n_open, terms in results.items():
#             print(f"// Open Indices {n_open}")
#             for term in terms:
#                 print(term.ambit())

# T2 = cluster_operator(2)
# T2_1 = cluster_operator(2, 2)
# V = Hamiltonian_operator(2)
# VT2_comm = commutator([V, T2, T2_1])
# for i, terms in VT2_comm.items():
#     filename = "VT2T2_{}".format(i)
#     with open(filename, 'w') as w:
#         for term in terms:
#             w.writelines(term.latex(delimiter=True, backslash=True) + '\n')

# for i, terms in commutator([V, T2, T2_1]).items():
#     filename = "./comm/srVT2T2/{}".format(i)
#     with open(filename, 'w') as w:
#         for term in terms:
#             w.writelines(term.latex(delimiter=True, backslash=True) + '\n')

# for i, terms in commutator([V, T2]).items():
#     filename = "./comm/VT2/ambit/{}".format(i)
#     with open(filename, 'w') as w:
#         last_sqop = terms[0].sqop
#         w.writelines(terms[0].ambit(not terms[0].sqop == terms[1].sqop) + '\n')
#         for term in terms[1:]:
#             add_permutation = False if term.sqop == last_sqop else True
#             w.writelines(term.ambit(add_permutation) + '\n')

# W = Hamiltonian_operator(3)
# T3 = cluster_operator(3)
# T3d = cluster_operator(3, 3, True)
# for i, terms in commutator([W, T3]).items():
#     filename = "./comm/WT3/ambit/{}".format(i)
#     with open(filename, 'w') as w:
#         nterms = len(terms)
#         for j in range(nterms):
#             try:
#                 last_sqop = terms[j - 1].sqop
#             except IndexError:
#                 last_sqop = SQOperator([], [])
#             try:
#                 next_sqop = terms[j + 1].sqop
#             except IndexError:
#                 next_sqop = SQOperator([], [])
#             add_permutation = False if terms[j].sqop == next_sqop else True
#             init_temp = False if terms[j].sqop == last_sqop else True
#             w.writelines(terms[j].ambit(add_permutation, init_temp))

# out = defaultdict(list)
# for i in range(1, 4):
#     H = Hamiltonian_operator(i)
#     for j in range(1, 4):
#         T = cluster_operator(j, hole_index='c', particle_index='v')
#         comm = commutator([H, T])
#         for nopen, terms in comm.items():
#             out[nopen] += terms
#
# for nopen, terms in out.items():
#     terms = sorted(terms)
#     filename = "./comm/ldsrg_sr/C{}".format(nopen / 2)
#     with open(filename, 'w') as w:
#         nterms = len(terms)
#         for i_term in range(nterms):
#             try:
#                 last_sqop = terms[i_term - 1].sqop
#             except IndexError:
#                 last_sqop = SQOperator([], [])
#             try:
#                 next_sqop = terms[i_term + 1].sqop
#             except IndexError:
#                 next_sqop = SQOperator([], [])
#             add_permutation = False if terms[i_term].sqop == next_sqop else True
#             init_temp = False if terms[i_term].sqop == last_sqop else True
#             w.writelines(terms[i_term].ambit(add_permutation, init_temp))
#             # w.writelines(terms[i_term].latex(delimiter=True, backslash=True) + '\n')

# for i in range(1, 4):
#     H = Hamiltonian_operator(i)
#     for j in range(1, 4):
#         T = cluster_operator(j)
#         for nopen, terms in commutator([H, T]).items():
#             filename = "./comm/ldsrg/H{}_T{}_C{}".format(i, j, nopen / 2)
#             with open(filename, 'w') as w:
#                 nterms = len(terms)
#                 for i_term in range(nterms):
#                     try:
#                         last_sqop = terms[i_term - 1].sqop
#                     except IndexError:
#                         last_sqop = SQOperator([], [])
#                     try:
#                         next_sqop = terms[i_term + 1].sqop
#                     except IndexError:
#                         next_sqop = SQOperator([], [])
#                     add_permutation = False if terms[i_term].sqop == next_sqop else True
#                     init_temp = False if terms[i_term].sqop == last_sqop else True
#                     w.writelines(terms[i_term].ambit(add_permutation, init_temp))

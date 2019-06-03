import multiprocessing
from threading import Thread
from joblib import Parallel, delayed
from copy import deepcopy
from collections import defaultdict, Iterable
from itertools import combinations, product
from math import factorial
from timeit import default_timer as timer
from sympy.utilities.iterables import multiset_permutations
from sympy.physics.quantum import Operator, HermitianOperator, Commutator, Dagger
from sympy.core.power import Pow
from sympy.core.add import Add
from sympy.core.mul import Mul

from integer_partition import integer_partition
from mo_space import space_relation
from Indices import Indices
from IndicesPair import make_indices_pair
from SQOperator import make_sqop, SecondQuantizedOperator
from Tensor import make_tensor_preset, ClusterAmplitude
from Term import Term
from sqop_contraction import generate_operator_contractions, generate_operator_contractions_new
from Timer import Timer


def HamiltonianOperator(k, start=0, indices_type='spin-orbital'):
    coeff = factorial(k) ** 2
    r0, r1, r2 = start, start + k, start + 2 * k
    tensor = make_tensor_preset("Hamiltonian",
                                [f"g{i}" for i in range(r1, r2)],
                                [f"g{i}" for i in range(r0, r1)],
                                indices_type)
    sq_op = make_sqop([f"g{i}" for i in range(r0, r1)], [f"g{i}" for i in range(r1, r2)], indices_type)
    return Term([tensor], sq_op, 1.0 / coeff)


def ClusterOperator(k, start=0, excitation=True, name='T', scale_factor=1.0,
                    hole_label='h', particle_label='p', indices_type='spin-orbital'):
    coeff = factorial(k) ** 2
    r0, r1 = start, start + k
    hole = [f"{hole_label}{i}" for i in range(r0, r1)]
    particle = [f"{particle_label}{i}" for i in range(r0, r1)]
    first = particle if excitation else hole
    second = hole if excitation else particle
    indices_pair = make_indices_pair(second, first, indices_type)
    tensor = ClusterAmplitude(indices_pair, name=name)
    sq_op = make_sqop(first, second, indices_type)
    return Term([tensor], sq_op, scale_factor / coeff)


def multiprocessing_canonicalize_contractions(tensors, sq_op, coeff):
    return Term(tensors, sq_op, coeff).canonicalize_sympy()


def calculate(func, args):
    return func(*args)


def calculatestar(args):
    return calculate(*args)


def contract_terms(terms, max_cu=3, max_n_open=6, min_n_open=0, scale_factor=1.0,
                   expand_hole=True, n_process=1):
    """
    Contract a list of Term objects.
    :param terms: a list of Term objects to be contracted
    :param max_cu: max level of cumulant
    :param max_n_open: max number of open indices for contractions kept for return
    :param min_n_open: min number of open indices for contractions kept for return
    :param scale_factor: a scaling factor for the results
    :param expand_hole: expand HoleDensity to Kronecker - Cumulant if True
    :param n_process: number of processes launched for tensor canonicalization
    :return: a map of number of open indices to a list of contracted and canonicalized Term objects
    """
    out = defaultdict(list)
    print(terms)

    if len(terms) == 0:
        raise ValueError("size of terms cannot be zero.")

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
        out[sq_op.n_ops].append(Term(tensors, sq_op, coeff))
    else:
        start = timer()
        contracted = generate_operator_contractions_new(sq_ops_to_be_contracted, max_cu,
                                                    max_n_open, min_n_open, expand_hole, n_process)
        end = timer()
        print(f'contraction: {end - start:.6f}s ')
        for k, contractions in contracted.items():
            n_contractions = len(contractions)
            print(n_contractions)

            start = timer()

            terms_k = list()

            if n_process == 1:
                for sign, densities, sq_op in contractions:
                    terms_k.append(Term(tensors + densities, sq_op, sign * coeff).canonicalize_sympy())
            else:
                with multiprocessing.Pool(n_process) as pool:
                    tasks = []
                    for sign, densities, sq_op in contractions:
                        tasks.append((multiprocessing_canonicalize_contractions,
                                      (tensors + densities, sq_op, sign * coeff)))
                    imap_unordered_it = pool.imap_unordered(calculatestar, tasks)
                    terms_k = [x for x in imap_unordered_it]

            end = timer()
            print(f'canonicalize: {end - start:.6f}s, ')
            start = timer()
            out[k] = combine_terms(terms_k)
            end = timer()
            print(f'combine: {end - start:.6f}s')

    return out


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

    tensors_to_coeff = defaultdict(list)
    tensors_to_term = dict()

    for term in terms:
        name = " ".join(str(tensor) for tensor in term.list_of_tensors) + f" {term.sq_op}"
        tensors_to_term[name] = term
        tensors_to_coeff[name].append(term.coeff)

    out = []
    for name, term in tensors_to_term.items():
        term.coeff = sum(tensors_to_coeff[name])
        out.append(term)
    return sorted(out)


def nested_commutator_lct(terms, max_cu=3, max_n_open=6, min_n_open=0, scale_factor=1.0,
                          expand_hole=True, n_process=1):
    """
    Compute the nested commutator of terms, i.e. [[...[[term_0, term_1], term_2], ...], term_k].
    :param terms: a list of terms
    :param max_cu: max level of cumulant
    :param max_n_open: max number of open indices for contractions of each single commutator
    :param min_n_open: min number of open indices for contractions of each single commutator
    :param scale_factor: a scaling factor for the results
    :param expand_hole: expand HoleDensity to Kronecker - Cumulant if True
    :param n_process: number of processes launched for tensor canonicalization
    :return: a map of number of open indices to a list of contracted canonicalized Term objects
    """
    if len(terms) == 0:
        raise ValueError("size of terms cannot be zero.")

    if len(terms) == 1:
        term = terms[0]
        return {term.sq_op.n_ops: [term]}

    out = defaultdict(list)

    def commutator_recursive(chosen_terms, unchosen_terms, sign):
        if len(unchosen_terms) == 0:
            temp = contract_terms(chosen_terms, max_cu, max_n_open, min_n_open,
                                  sign * scale_factor, expand_hole, n_process)
            for n_open, _terms in temp.items():
                out[n_open] = combine_terms(out[n_open] + _terms)
        else:
            commutator_recursive(chosen_terms + [unchosen_terms[0]], unchosen_terms[1:], sign)
            commutator_recursive([unchosen_terms[0]] + chosen_terms, unchosen_terms[1:], -sign)

    commutator_recursive(terms[:1], terms[1:], 1)

    return out


def sympy_nested_commutator_recursive(level, A, B):
    """
    Compute nested commutator of type [[...[[A, B], B], ...], B]
    :param level: the level of nested commutator
    :param A: Operator A
    :param B: Operator B
    :return: commutator of type Add
    """
    if level <= 1:
        return Commutator(A, B)
    for i in range(level)[::-1]:
        return Commutator(sympy_nested_commutator_recursive(i, A, B), B)


def nested_commutator_cc(nested_level, cluster_levels, max_cu=3, max_n_open=6, min_n_open=0,
                         expand_hole=True, single_reference=False, unitary=False, n_process=1):
    """
    Compute the BCH nested commutator in coupled cluster theory.
    :param nested_level: the level of nested commutator
    :param cluster_levels: a list of integers for cluster operator, e.g., [1,2,3] for T1 + T2 + T3
    :param max_cu: max value of cumulant allowed for contraction
    :param max_n_open: the max number of open indices for contractions kept for return
    :param min_n_open: the min number of open indices for contractions kept for return
    :param expand_hole: expand HoleDensity to Kronecker minus Cumulant if True
    :param single_reference: use single-reference amplitudes if True
    :param unitary: use unitary formalism if True
    :param n_process: number of processes launched for tensor canonicalization
    :return: a map of number of open indices to a list of contracted canonicalized Term objects
    """
    if not isinstance(nested_level, int):
        raise ValueError("Invalid nested_level (must be an integer)")
    if not isinstance(cluster_levels, Iterable):
        raise ValueError("Invalid type for cluster_operator")
    if not all(isinstance(t, int) for t in cluster_levels):
        raise ValueError("Invalid content in cluster_operator (must be all integers)")

    scale_factor = 1.0/factorial(nested_level)
    out = defaultdict(list)

    hole_label = 'c' if single_reference else 'h'
    particle_label = 'v' if single_reference else 'p'

    # symbolic evaluate nested commutator
    T = sum(Operator(f'T{i}') for i in cluster_levels)
    H = HermitianOperator('H1') + HermitianOperator('H2')
    A = T - Dagger(T) if unitary else T

    for term in sympy_nested_commutator_recursive(nested_level, H, A).doit().expand().args:
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
                list_of_terms.append(ClusterOperator(n_body, start=start, excitation=(real_name == 'T'),
                                                     hole_label=hole_label, particle_label=particle_label))
                start += n_body
            else:
                list_of_terms.append(HamiltonianOperator(n_body))

        for n_open, terms in contract_terms(list_of_terms, max_cu, max_n_open, min_n_open,
                                            factor, expand_hole, n_process).items():
            out[n_open] += terms

    for n_open, terms in out.items():
        out[n_open] = combine_terms(terms)

    return out


def sort_contraction_results(results):
    """
    Sort contraction results to blocks of operators and permutations.
    :param results: the sorted output from operator contractions
    :return: a dictionary of terms with (space, permutation) being the key
    """
    out = defaultdict(list)
    for n_open, terms in results.items():
        for term in terms:
            space_str = "".join([i.space for i in term.sq_op.ann_ops] + [i.space for i in term.sq_op.cre_ops])
            n_perm, perm_str, sq_op_str = term.sq_op.latex_permute_format(*term.exist_permute_format())
            out[(space_str, perm_str)].append(term)
    return out


def print_terms_ambit(results):
    """
    Print contracted results in ambit format.
    :param results: the sorted output from operator contractions
    """
    block_repr = sort_contraction_results(results)
    for k, terms in sorted(block_repr.items(), key=lambda pair: (len(pair[0][0]), len(pair[0][1]), pair[0][0])):
        i_last = len(terms) - 1
        for i, term in enumerate(terms):
            print(term.ambit(ignore_permutations=(i != i_last), init_temp=(i == 0), declared_temp=True))


def print_terms_ambit_symmetric(results):
    """
    Print contracted results in ambit format assuming results are symmetric (i.e. block ccvv = block vvcc).
    :param results: the sorted output from operator contractions
    """
    block_repr = sort_contraction_results(results)
    sym_blocks = set()
    for block, terms in sorted(block_repr.items(), key=lambda pair: (len(pair[0][0]), len(pair[0][1]), pair[0][0])):
        if block[0] in sym_blocks:
            continue
        half1, half2 = block[0][:len(block)//2], block[0][len(block)//2:]
        sym_blocks.add(half2 + half1)

        i_last = len(terms) - 1
        for i, term in enumerate(terms):
            print(term.ambit(ignore_permutations=(i != i_last), init_temp=(i == 0), declared_temp=True))


def save_terms_ambit(results, full_path):
    """
    Save the contracted results to C++ file.
    :param results: the output from operator contractions
    :param full_path: the full path of the file name
    """
    return


def print_results(results, form='latex'):
    if form == 'latex':
        print("\\begin{align}")
        for n_open, terms in results.items():
            print(f"% Open Indices {n_open}")
            for term in terms:
                print(term.latex(delimiter=True, backslash=True))
        print("\\end{align}")
    else:
        for n_open, terms in results.items():
            print(f"// Open Indices {n_open}")
            for term in terms:
                print(term.ambit())

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

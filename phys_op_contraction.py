from copy import deepcopy
from collections import defaultdict
from itertools import combinations, product
from math import factorial
from sympy.utilities.iterables import multiset_permutations
from integer_partition import integer_partition
from mo_space import space_relation
from Indices import Indices
from IndicesPair import make_indices_pair
from SQOperator import make_sqop, SecondQuantizedOperator
from Tensor import make_tensor_preset, ClusterAmplitude
from Term import Term
from sqop_contraction import generate_operator_contractions
from Timer import Timer


def Hamiltonian_operator(k, start=0, indices_type='spin-orbital'):
    coeff = factorial(k) ** 2
    r0, r1, r2 = start, start + k, start + 2 * k
    tensor = make_tensor_preset("Hamiltonian",
                                [f"g{i}" for i in range(r1, r2)],
                                [f"g{i}" for i in range(r0, r1)],
                                indices_type)
    sq_op = make_sqop([f"g{i}" for i in range(r0, r1)], [f"g{i}" for i in range(r1, r2)], indices_type)
    return Term([tensor], sq_op, 1.0 / coeff)


def cluster_operator(k, start=0, excitation=True, name='T', scale_factor=1.0,
                     hole_label='h', particle_label='p', indices_type='spin-orbital'):
    coeff = factorial(k) ** 2
    r0, r1 = start, start + k
    hole = [f"{hole_label}{i}" for i in range(r0, r1)]
    particle = [f"{particle_label}{i}" for i in range(r0, r1)]
    first = particle if excitation else hole
    second = hole if excitation else particle
    indices_pair = make_indices_pair(second, first, indices_type)
    tensor = ClusterAmplitude(indices_pair, name=name, excitation=excitation)
    sq_op = make_sqop(first, second, indices_type)
    return Term([tensor], sq_op, scale_factor / coeff)


def contract_terms(terms, max_cu=3, max_n_open=6, min_n_open=0, scale_factor=1.0, expand_hole=True):
    """
    Contract a list of Term objects.
    :param terms: a list of Term objects to be contracted
    :param max_cu: max level of cumulant
    :param max_n_open: max number of open indices for contractions kept for return
    :param min_n_open: min number of open indices for contractions kept for return
    :param scale_factor: a scaling factor for the results
    :param expand_hole: expand HoleDensity to Kronecker - Cumulant if True
    :return: a map of number of open indices to a list of contracted and canonicalized Term objects
    """
    out = defaultdict(list)

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
        contracted = generate_operator_contractions(sq_ops_to_be_contracted, max_cu,
                                                    max_n_open, min_n_open, expand_hole)
        for k, contractions in contracted.items():
            print(len(contractions))
            terms_k = []
            for sign, densities, sq_op in contractions:
                term = Term(tensors + densities, sq_op, sign * coeff)
                terms_k.append(term.canonicalize())
            out[k] = combine_terms(sorted(terms_k))

    return out


def combine_terms(terms, presorted=True):
    """
    Simplify the list of terms by combining similar terms.
    :param terms: a list of canonicalized Term objects
    :param presorted: the list is treated as sorted list if True
    :return: a list of simplified Term objects
    """
    if not isinstance(terms, list):
        raise ValueError("terms should be a list of Term objects.")

    if not presorted:
        terms = sorted(terms)

    if len(terms) == 0:
        return []

    out = [terms[0]]
    for term in terms[1:]:
        if term.almost_equal(out[-1]):
            out[-1].coeff += term.coeff
        else:
            if abs(out[-1].coeff) < 1.0e-15:
                out.pop()
            out.append(term)

    if abs(out[-1].coeff) < 1.0e-15:
        out.pop()

    return sorted(out)


def commutator(terms, max_cu=3, max_n_open=6, min_n_open=0, scale_factor=1.0, expand_hole=True):
    """
    Compute the nested commutator of terms, i.e. [[...[[term_0, term_1], term_2], ...], term_k].
    :param terms: a list of terms.
    :param max_cu: max level of cumulant
    :param max_n_open: max number of open indices for contractions kept for return
    :param min_n_open: min number of open indices for contractions kept for return
    :param scale_factor: a scaling factor for the results
    :param expand_hole: expand HoleDensity to Kronecker - Cumulant if True
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
            temp = contract_terms(chosen_terms, max_cu, max_n_open, min_n_open, sign * scale_factor, expand_hole)
            for n_open, _terms in temp.items():
                out[n_open] = combine_terms(out[n_open] + _terms, presorted=False)
        else:
            commutator_recursive(chosen_terms + [unchosen_terms[0]], unchosen_terms[1:], sign)
            commutator_recursive([unchosen_terms[0]] + chosen_terms, unchosen_terms[1:], -sign)

    commutator_recursive(terms[:1], terms[1:], 1)

    return out


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

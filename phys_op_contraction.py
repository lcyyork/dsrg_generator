from copy import deepcopy
from collections import defaultdict
from itertools import combinations, product
from math import factorial
from sympy.utilities.iterables import multiset_permutations
from integer_partition import integer_partition
from Index import space_relation
from Indices import Indices
from SQOperator import make_sqop
from Tensor import make_tensor_preset
from Term import Term, read_latex_term
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


def cluster_operator(k, start=0, deexcitation=False, hole_label='h', particle_label='p', indices_type='spin-orbital'):
    coeff = factorial(k) ** 2
    r0, r1 = start, start + k
    hole = [f"{hole_label}{i}" for i in range(r0, r1)]
    particle = [f"{particle_label}{i}" for i in range(r0, r1)]
    first = hole if deexcitation else particle
    second = particle if deexcitation else hole
    tensor = make_tensor_preset("cluster_amplitude", second, first, indices_type)
    sq_op = make_sqop(first, second, indices_type)
    return Term([tensor], sq_op, 1.0 / coeff)


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


def contract_terms(terms, maxcu=3, maxmb=3, minmb=0, scale_factor=1.0, expand_hole=True):
    """
    Contract a list of Term objects.
    :param terms: a list of Term objects to be contracted
    :param maxcu: max level of cumulants allowed
    :param maxmb: max level of many-body operators kept
    :param minmb: min level of many-body operators kept
    :param scale_factor: a scaling factor for the results
    :param expand_hole: expand HoleDensity to Kronecker - Cumulant if True
    :return: a map of number of open indices to a list of contracted and canonicalized Term objects
    """
    out = defaultdict(list)

    coeff = 1.0 * scale_factor
    tensors = []
    sqops_to_be_contracted = []
    for term in terms:
        if not isinstance(term, Term):
            raise ValueError("{} if not of Term type".format(term))
        coeff *= term.coeff
        tensors += term.list_of_tensors
        if not term.sqop.is_empty_sqop():
            sqops_to_be_contracted.append(term.sqop)

    if len(sqops_to_be_contracted) < 2:
        sqop = sqops_to_be_contracted[0] if len(sqops_to_be_contracted) == 1 else SQOperator([], [])
        nopen = sqop.ncre + sqop.nann
        out[nopen].append(Term(tensors, sqop, coeff))
    else:
        contracted = generate_operator_contractions(sqops_to_be_contracted, maxcu, maxmb, minmb, expand_hole)
        for k, contractions in contracted.items():
            print(len(contractions))
            terms_k = []
            for sign, densities, sqop in contractions:
                term = Term(tensors + densities, sqop, sign * coeff)
                terms_k.append(term.canonicalize())
            out[k] = combine_terms(sorted(terms_k))

    return out


def commutator(terms, maxcu=3, maxmb=3, minmb=0, scale_factor=1.0, expand_hole=True):
    """
    Compute the nested commutator of terms, i.e. [[...[[term_0, term_1], term_2], ...], term_k].
    :param terms: a list of terms.
    :param maxcu: max level of cumulants allowed
    :param maxmb: max level of many-body operators kept in results
    :param minmb: min level of many-body operators kept in results
    :param scale_factor: a scaling factor for the results
    :param expand_hole: expand HoleDensity to Kronecker - Cumulant if True
    :return: a map of number of open indices to a list of contracted canonicalized Term objects
    """
    if len(terms) < 2:
        term = terms[0] if len(terms) == 1 else Term([], SQOperator([], []), 0.0)
        nopen = term.sqop.nops()
        return {nopen: [term]}

    out = defaultdict(list)

    def commutator_recursive(chosen_terms, unchosen_terms, sign):
        if len(unchosen_terms) == 0:
            temp = contract_terms(chosen_terms, maxcu, maxmb, minmb, sign * scale_factor, expand_hole)
            for nopen, terms in temp.items():
                out[nopen] = combine_terms(out[nopen] + terms, presorted=False)
        else:
            commutator_recursive(chosen_terms + [unchosen_terms[0]], unchosen_terms[1:], sign)
            commutator_recursive([unchosen_terms[0]] + chosen_terms, unchosen_terms[1:], -sign)

    commutator_recursive(terms[:1], terms[1:], 1)

    return out




# def Hamiltonian_operator(k, start=0):
#     coeff = factorial(k) ** 2
#     r0, r1, r2 = start, start + k, start + 2 * k
#     tensor = Hamiltonian(["g{}".format(i) for i in range(r1, r2)],
#                          ["g{}".format(i) for i in range(r0, r1)])
#     sqop = SQOperator(["g{}".format(i) for i in range(r0, r1)],
#                       ["g{}".format(i) for i in range(r1, r2)])
#     return Term([tensor], sqop, 1 / coeff)
#
#
# def cluster_operator(k, start=0, deexcitation=False, hole_index='h', particle_index='p'):
#     coeff = factorial(k) ** 2
#     r0, r1 = start, start + k
#     hole = ["{}{}".format(hole_index, i) for i in range(r0, r1)]
#     particle = ["{}{}".format(particle_index, i) for i in range(r0, r1)]
#     if deexcitation:
#         tensor = ClusterAmp(particle, hole)
#         sqop = SQOperator(hole, particle)
#     else:
#         tensor = ClusterAmp(hole, particle)
#         sqop = SQOperator(particle, hole)
#     return Term([tensor], sqop, 1 / coeff)

T2 = cluster_operator(2)
T2_1 = cluster_operator(2, 2)
V = Hamiltonian_operator(2)
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

out = defaultdict(list)
for i in range(1, 4):
    H = Hamiltonian_operator(i)
    for j in range(1, 4):
        T = cluster_operator(j, hole_index='c', particle_index='v')
        comm = commutator([H, T])
        for nopen, terms in comm.items():
            out[nopen] += terms

for nopen, terms in out.items():
    terms = sorted(terms)
    filename = "./comm/ldsrg_sr/C{}".format(nopen / 2)
    with open(filename, 'w') as w:
        nterms = len(terms)
        for i_term in range(nterms):
            try:
                last_sqop = terms[i_term - 1].sqop
            except IndexError:
                last_sqop = SQOperator([], [])
            try:
                next_sqop = terms[i_term + 1].sqop
            except IndexError:
                next_sqop = SQOperator([], [])
            add_permutation = False if terms[i_term].sqop == next_sqop else True
            init_temp = False if terms[i_term].sqop == last_sqop else True
            w.writelines(terms[i_term].ambit(add_permutation, init_temp))
            # w.writelines(terms[i_term].latex(delimiter=True, backslash=True) + '\n')

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

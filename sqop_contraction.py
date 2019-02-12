from copy import deepcopy
from collections import defaultdict
from itertools import combinations, product
from math import factorial
from sympy.utilities.iterables import multiset_permutations
from integer_partition import integer_partition
from mo_space import space_relation
from Indices import Indices, IndicesSpinOrbital
from IndicesPair import IndicesPair
from SQOperator import SecondQuantizedOperator
from Tensor import make_tensor_preset, HoleDensity, Kronecker, Cumulant
from Timer import Timer


def generate_elementary_contractions(ops_list, max_cu=3):
    """
    Generate all elementary contractions from a list of second-quantized operators.
    :param ops_list: a list of SecondQuantizedOperator objects
    :param max_cu: the max level of cumulants
    :return: a list of elementary contractions represented by HoleDensity and Cumulant
    """
    for op in ops_list:
        if not isinstance(op, SecondQuantizedOperator):
            raise TypeError(f"Invalid type in ops_list, given '{op.__class__.__name__}', "
                            f"required 'SecondQuantizedOperator'.")
        if op.type_of_indices != IndicesSpinOrbital:
            raise NotImplementedError(f"Contractions only supports spin-orbital indices now.")
    if not isinstance(max_cu, int):
        raise TypeError(f"Invalid type of max_cu, given '{max_cu.__class__.__name__}', required 'int'.")

    # determine if max_cu makes sense (cumulant indices cannot be core or virtual)
    cv = ["c", "v"]
    n_valid_cre = sum([op.n_cre - op.cre_ops.count_index_space(cv) for op in ops_list])
    n_valid_ann = sum([op.n_ann - op.ann_ops.count_index_space(cv) for op in ops_list])
    max_cu_allowed = max(1, min(n_valid_cre, n_valid_ann))
    if max_cu < 1 or max_cu > max_cu_allowed:
        print(f"Max cumulant level is set to {max_cu_allowed}.")
        max_cu = max_cu_allowed

    out_list = []

    # 1-body cumulant and hole density
    for i, left in enumerate(ops_list):
        for right in ops_list[i + 1:]:
            for u in left.cre_ops:
                if u.space == 'v':
                    continue
                for l in right.ann_ops:
                    if l.space == 'v':
                        continue
                    if len(space_relation[u.space] & space_relation[l.space]) != 0:
                        out_list.append(make_tensor_preset([u], [l], 'spin-orbital', 'cumulant'))
            for l in left.ann_ops:
                if l.space == 'c':
                    continue
                for u in right.cre_ops:
                    if u.space == 'c':
                        continue
                    if len(space_relation[u.space] & space_relation[l.space]) != 0:
                        out_list.append(make_tensor_preset([u], [l], 'spin-orbital', 'hole_density'))
    if max_cu < 2:
        return out_list

    # for cumulant, since n_cre = n_ann, consider cre/ann separately
    cre_ops_list = [IndicesSpinOrbital([i for i in op.cre_ops if i.space not in cv]) for op in ops_list]
    ann_ops_list = [IndicesSpinOrbital([i for i in op.ann_ops if i.space not in cv]) for op in ops_list]

    # generate all possible partitions for k cre/ann legs for k cumulant
    # only unique partitions here, e.g., [2,1,1] included, not [1,2,1] or [1,1,2]
    n_sq_ops = len(ops_list)
    unique_partitions = [part for k in range(1, max_cu + 1)
                         for part in integer_partition(k) if len(part) <= n_sq_ops]

    # generate all possible pure creation or annihilation cumulant contractions
    def generate_half_cumulant_contractions(pure_ops_list):
        """
        Generate cumulant contractions for pure creation and annihilation operators.
        :param pure_ops_list: a list of pure creation or annihilation indices for each input operator
        :return: {cumulant level: [[n_cumulant of chosen indices (op index, relative index)], ...]}
        """
        results = {i: [] for i in range(2, max_cu + 1)}

        # generate all possible sub-indices for each input second-quantized operator
        # [{n_leg: [relative indices of the current string of cre/ann operators]}, ...]
        sub_indices = [{n_leg: [ele_ops for ele_ops in combinations(range(ops.size), n_leg)]
                        for n_leg in range(1, min(max_cu, ops.size) + 1)} for ops in pure_ops_list]

        for unique_partition in unique_partitions:
            n_ops = len(unique_partition)
            cu_level = sum(unique_partition)

            for partition in multiset_permutations(unique_partition):

                # choose n_ops from ops_list
                for ops in combinations(range(n_sq_ops), n_ops):

                    # check if this partition is valid on this ops
                    if any([len(pure_ops_list[i]) < n_leg for i, n_leg in zip(ops, partition)]):
                        continue

                    # generate all possibilities
                    for sub_indices_product in product(*[sub_indices[i][n_leg] for i, n_leg in zip(ops, partition)]):
                        results[cu_level].append([(i, index) for i, indices in zip(ops, sub_indices_product)
                                                  for index in indices])

        return results

    ann_results = generate_half_cumulant_contractions(ann_ops_list)
    cre_results = generate_half_cumulant_contractions(cre_ops_list)

    def all_equal_elements(lst):
        return lst.count(lst[0]) == len(lst)

    # now combine the cre/ann results
    for cu_level in range(2, max_cu + 1):
        for cre in cre_results[cu_level]:
            same_sq_op_cre = all_equal_elements([cre[i][0] for i in range(cu_level)])
            cre_indices = [cre_ops_list[cre[i][0]][cre[i][1]] for i in range(cu_level)]
            for ann in ann_results[cu_level]:
                # skip when cre and ann belong to same operator
                same_sq_op_ann = all_equal_elements([ann[i][0] for i in range(cu_level)])
                if same_sq_op_cre and same_sq_op_ann and cre[0][0] == ann[0][0]:
                    continue
                else:
                    ann_indices = [ann_ops_list[ann[i][0]][ann[i][1]] for i in range(cu_level)]
                    out_list.append(make_tensor_preset(cre_indices, ann_indices, 'spin-orbital', 'cumulant'))

    return out_list


def generate_operator_contractions(ops_list, max_cu=3, max_n_open=6, min_n_open=0, expand_hole=True):
    """
    Generate operator contractions for a list of SQOperator.
    :param ops_list: a list of SecondQuantizedOperator to be contracted
    :param max_cu: max level of cumulant
    :param max_n_open: max number of open indices for contractions kept for return
    :param min_n_open: min number of open indices for contractions kept for return
    :param expand_hole: expand hole density to Kronecker delta and 1-cumulant if True
    :return: a map of number of open indices to contractions
    """
    # generate elementary contractions
    elementary_contractions = generate_elementary_contractions(ops_list, max_cu)
    n_ele_con = len(elementary_contractions)

    # generate incompatible contractions between elementary contractions
    # the list index of elementary_contractions is saved
    incompatible_elementary = {i: set() for i in range(n_ele_con)}
    for i, ele_i in enumerate(elementary_contractions):
        for j, ele_j in enumerate(elementary_contractions[i + 1:], i + 1):
            if ele_i.any_overlapped_indices(ele_j):
                incompatible_elementary[i].add(j)
                incompatible_elementary[j].add(i)

    # backtracking (similar to generating sub-lists)
    contractions = []
    composite_contractions_backtracking(set(range(n_ele_con)), set(), incompatible_elementary, contractions)

    # translate contractions to readable form
    results = defaultdict(list)

    base_order_indices = Indices([])
    upper_indices_set, lower_indices_set = set(), set()
    for sq_op in ops_list:
        base_order_indices += sq_op.string_form
        upper_indices_set |= sq_op.cre_ops.indices_set
        lower_indices_set |= sq_op.ann_ops.indices_set

    # contraction is a list of indices of elementary contractions
    for contraction in contractions:

        n_sq_ops_open = base_order_indices.size - sum([elementary_contractions[i].size for i in contraction])
        if min_n_open <= n_sq_ops_open <= max_n_open:

            list_of_densities = []
            current_order = []

            for ele_con in (elementary_contractions[i] for i in contraction):
                list_of_densities.append(ele_con)

                left, right = ele_con.upper_indices, ele_con.lower_indices
                if isinstance(ele_con, HoleDensity):
                    left, right = right, left
                current_order += left.indices + right.indices[::-1]

            # expand hole densities to delta - lambda_1
            sign_densities_pairs = expand_hole_densities(list_of_densities) if expand_hole else [(1, list_of_densities)]

            # sort the open indices
            open_upper_indices, open_lower_indices = IndicesSpinOrbital([]), IndicesSpinOrbital([])
            if n_sq_ops_open != 0:
                contracted_indices_set = set(current_order)
                open_upper_indices = IndicesSpinOrbital(sorted(upper_indices_set - contracted_indices_set))
                open_lower_indices = IndicesSpinOrbital(sorted(lower_indices_set - contracted_indices_set))
                current_order += open_upper_indices.indices + open_lower_indices.indices[::-1]
            sq_op = SecondQuantizedOperator(IndicesPair(open_upper_indices, open_lower_indices))

            # determine sign and push to results
            sign = (-1) ** (base_order_indices.count_permutations(Indices(current_order)))
            for _sign, list_of_densities in sign_densities_pairs:
                results[n_sq_ops_open].append((sign * _sign, list_of_densities, sq_op))

    return results


def composite_contractions_backtracking(available, chosen, incompatible, out):
    """
    Generate composite contractions from elementary contractions.
    :param available: unexplored set of elementary contractions
    :param chosen: chosen set of elementary contractions
    :param incompatible: a map to test incompatible elementary contractions
    :param out: final results from outside
    :return: viable composite contractions
    """
    if len(available) == 0:  # base case, nothing to choose
        if len(chosen) != 0:
            out.append(deepcopy(chosen))
    else:
        # two choices to explore: with or without the given element
        temp = available.pop()  # choose

        chosen.add(temp)  # choose to include this element
        composite_contractions_backtracking(available - incompatible[temp], chosen, incompatible, out)

        chosen.remove(temp)  # choose not to include this element
        composite_contractions_backtracking(available, chosen, incompatible, out)

        available.add(temp)  # un-choose


def expand_hole_densities(list_of_tensors):
    """
    Expand all the hole densities in the list_of_tensors.
    :param list_of_tensors: a list of Tensor objects
    :return: a list of (sign, expanded Tensor objects)
    """
    out = []

    good_tensors, hole_densities = [], []
    for tensor in list_of_tensors:
        hole_densities.append(tensor) if isinstance(tensor, HoleDensity) else good_tensors.append(tensor)

    # if hole density contain any virtual index, it can only be delta
    skip_cumulant = [hole.upper_indices[0].space == 'v' or hole.lower_indices[0].space == 'v'
                     for hole in hole_densities]

    n_hole = len(hole_densities)
    if n_hole == 0:
        out.append((1, list_of_tensors))
    else:
        for temp in product(range(2), repeat=n_hole):
            sign = (-1) ** sum(temp)

            if any([i and j for i, j in zip(temp, skip_cumulant)]):
                continue

            expanded = []
            for i, to_cu in enumerate(temp):
                upper_indices, lower_indices = hole_densities[i].upper_indices, hole_densities[i].lower_indices
                if temp[i] == 0:
                    expanded.append(Kronecker(IndicesPair(upper_indices, lower_indices)))
                else:
                    expanded.append(Cumulant(IndicesPair(upper_indices, lower_indices)))

            out.append((sign, good_tensors + expanded))

    return out


# a = SQOperator(["g0", "g1"], ["g2", "g3"])
# b = SQOperator(["p0", "p1"], ["h0", "h1"])
# c = SQOperator(["p2", "p3"], ["h2", "h3"])
# d = SQOperator(["g{}".format(i) for i in range(3)], ["g{}".format(i) for i in range(3, 6)])
# e = SQOperator(["p{}".format(i) for i in range(3)], ["h{}".format(i) for i in range(3)])
#
# # with Timer("elementary contraction for V * T2 * T2, maxcu = 4") as t:
# #     out, (i,j) = generate_elementary_contractions([a, b, c], maxcu=4)
# #     # print(len(out))
# #     # for i in out:
# #     #     print(i)
# # # with Timer("elementary contraction for V * T2 * T2, maxcu = 3") as t:
# # #     out, (i,j) = generate_elementary_contractions([a, b, c], maxcu=3)
# # # with Timer("elementary contraction for W * T3, maxcu = 10") as t:
# # #     out, (i,j) = generate_elementary_contractions([d, e], maxcu=5)
# # with Timer("elementary contraction for W * T3, maxcu = 3") as t:
# #     out, (i,j) = generate_elementary_contractions([d, e], maxcu=3)
# #     # print(len(out))
# #     # for i in out:
# #     #     print(i)
#
# a = SQOperator(["g0", "g1", "g2"], ["g3", "g4", "g5"])
# b = SQOperator(["v0", "v1", "v2"], ["c0", "c1", "c2"])
# # b = SQOperator(["v4"], ["c5"])
# # b = SQOperator(["p0", "p1"], ["h0", "h1"])
# # a = SQOperator(["g0", "g1", "g2"], ["g3", "g4", "g5"])
# # b = SQOperator(["p0", "p1", "p2"], ["h0", "h1", "h2"])
# # ab = generate_operator_contractions([a, b])
# # for k, vs in ab.items():
# #     for v in vs:
# #         print(k, v)
#
#
# def combine_terms(terms, presorted=True):
#     """
#     Simplify the list of terms by combining similar terms.
#     :param terms: a list of canonicalized Term objects
#     :param presorted: the list is treated as sorted list if True
#     :return: a list of simplified Term objects
#     """
#     if not isinstance(terms, list):
#         raise ValueError("terms should be a list of Term objects.")
#
#     if not presorted:
#         terms = sorted(terms)
#
#     out = [terms[0]]
#     for term in terms[1:]:
#         if term.almost_equal(out[-1]):
#             out[-1].coeff += term.coeff
#         else:
#             if abs(out[-1].coeff) < 1.0e-15:
#                 out.pop()
#             out.append(term)
#
#     if abs(out[-1].coeff) < 1.0e-15:
#         out.pop()
#
#     return sorted(out)
#
#
# def contract_terms(terms, maxcu=3, maxmb=3, minmb=0, scale_factor=1.0, expand_hole=True):
#     """
#     Contract a list of Term objects.
#     :param terms: a list of Term objects to be contracted
#     :param maxcu: max level of cumulants allowed
#     :param maxmb: max level of many-body operators kept
#     :param minmb: min level of many-body operators kept
#     :param scale_factor: a scaling factor for the results
#     :param expand_hole: expand HoleDensity to Kronecker - Cumulant if True
#     :return: a map of number of open indices to a list of contracted and canonicalized Term objects
#     """
#     out = defaultdict(list)
#
#     coeff = 1.0 * scale_factor
#     tensors = []
#     sqops_to_be_contracted = []
#     for term in terms:
#         if not isinstance(term, Term):
#             raise ValueError("{} if not of Term type".format(term))
#         coeff *= term.coeff
#         tensors += term.list_of_tensors
#         if not term.sqop.is_empty_sqop():
#             sqops_to_be_contracted.append(term.sqop)
#
#     if len(sqops_to_be_contracted) < 2:
#         sqop = sqops_to_be_contracted[0] if len(sqops_to_be_contracted) == 1 else SQOperator([], [])
#         nopen = sqop.ncre + sqop.nann
#         out[nopen].append(Term(tensors, sqop, coeff))
#     else:
#         contracted = generate_operator_contractions(sqops_to_be_contracted, maxcu, maxmb, minmb, expand_hole)
#         for k, contractions in contracted.items():
#             print(len(contractions))
#             terms_k = []
#             for sign, densities, sqop in contractions:
#                 term = Term(tensors + densities, sqop, sign * coeff)
#                 terms_k.append(term.canonicalize())
#             out[k] = combine_terms(sorted(terms_k))
#
#     return out
#
#
# def commutator(terms, maxcu=3, maxmb=3, minmb=0, scale_factor=1.0, expand_hole=True):
#     """
#     Compute the nested commutator of terms, i.e. [[...[[term_0, term_1], term_2], ...], term_k].
#     :param terms: a list of terms.
#     :param maxcu: max level of cumulants allowed
#     :param maxmb: max level of many-body operators kept in results
#     :param minmb: min level of many-body operators kept in results
#     :param scale_factor: a scaling factor for the results
#     :param expand_hole: expand HoleDensity to Kronecker - Cumulant if True
#     :return: a map of number of open indices to a list of contracted canonicalized Term objects
#     """
#     if len(terms) < 2:
#         term = terms[0] if len(terms) == 1 else Term([], SQOperator([], []), 0.0)
#         nopen = term.sqop.nops()
#         return {nopen: [term]}
#
#     out = defaultdict(list)
#
#     def commutator_recursive(chosen_terms, unchosen_terms, sign):
#         if len(unchosen_terms) == 0:
#             temp = contract_terms(chosen_terms, maxcu, maxmb, minmb, sign * scale_factor, expand_hole)
#             for nopen, terms in temp.items():
#                 out[nopen] = combine_terms(out[nopen] + terms, presorted=False)
#         else:
#             commutator_recursive(chosen_terms + [unchosen_terms[0]], unchosen_terms[1:], sign)
#             commutator_recursive([unchosen_terms[0]] + chosen_terms, unchosen_terms[1:], -sign)
#
#     commutator_recursive(terms[:1], terms[1:], 1)
#
#     return out
#
#
#
#
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
#
# T2 = cluster_operator(2)
# T2_1 = cluster_operator(2, 2)
# V = Hamiltonian_operator(2)
# # VT2_comm = commutator([V, T2, T2_1])
# # for i, terms in VT2_comm.items():
# #     filename = "VT2T2_{}".format(i)
# #     with open(filename, 'w') as w:
# #         for term in terms:
# #             w.writelines(term.latex(delimiter=True, backslash=True) + '\n')
#
# # for i, terms in commutator([V, T2, T2_1]).items():
# #     filename = "./comm/srVT2T2/{}".format(i)
# #     with open(filename, 'w') as w:
# #         for term in terms:
# #             w.writelines(term.latex(delimiter=True, backslash=True) + '\n')
#
# # for i, terms in commutator([V, T2]).items():
# #     filename = "./comm/VT2/ambit/{}".format(i)
# #     with open(filename, 'w') as w:
# #         last_sqop = terms[0].sqop
# #         w.writelines(terms[0].ambit(not terms[0].sqop == terms[1].sqop) + '\n')
# #         for term in terms[1:]:
# #             add_permutation = False if term.sqop == last_sqop else True
# #             w.writelines(term.ambit(add_permutation) + '\n')
#
# # W = Hamiltonian_operator(3)
# # T3 = cluster_operator(3)
# # T3d = cluster_operator(3, 3, True)
# # for i, terms in commutator([W, T3]).items():
# #     filename = "./comm/WT3/ambit/{}".format(i)
# #     with open(filename, 'w') as w:
# #         nterms = len(terms)
# #         for j in range(nterms):
# #             try:
# #                 last_sqop = terms[j - 1].sqop
# #             except IndexError:
# #                 last_sqop = SQOperator([], [])
# #             try:
# #                 next_sqop = terms[j + 1].sqop
# #             except IndexError:
# #                 next_sqop = SQOperator([], [])
# #             add_permutation = False if terms[j].sqop == next_sqop else True
# #             init_temp = False if terms[j].sqop == last_sqop else True
# #             w.writelines(terms[j].ambit(add_permutation, init_temp))
#
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
#
# # for i in range(1, 4):
# #     H = Hamiltonian_operator(i)
# #     for j in range(1, 4):
# #         T = cluster_operator(j)
# #         for nopen, terms in commutator([H, T]).items():
# #             filename = "./comm/ldsrg/H{}_T{}_C{}".format(i, j, nopen / 2)
# #             with open(filename, 'w') as w:
# #                 nterms = len(terms)
# #                 for i_term in range(nterms):
# #                     try:
# #                         last_sqop = terms[i_term - 1].sqop
# #                     except IndexError:
# #                         last_sqop = SQOperator([], [])
# #                     try:
# #                         next_sqop = terms[i_term + 1].sqop
# #                     except IndexError:
# #                         next_sqop = SQOperator([], [])
# #                     add_permutation = False if terms[i_term].sqop == next_sqop else True
# #                     init_temp = False if terms[i_term].sqop == last_sqop else True
# #                     w.writelines(terms[i_term].ambit(add_permutation, init_temp))

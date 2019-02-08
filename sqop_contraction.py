from copy import deepcopy
from collections import defaultdict
from itertools import combinations, product
from math import factorial
from sympy.utilities.iterables import multiset_permutations
from integer_partition import integer_partition
from mo_space import space_relation
from Indices import Indices, IndicesSpinOrbital
from SQOperator import SecondQuantizedOperator
from IndicesPair import IndicesPair
from Tensor import make_tensor_preset
from Term import Term, read_latex_term
from Timer import Timer


def generate_ele_cont(ops_list, max_cu=3):
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
        if op.type_of_indices != type(IndicesSpinOrbital):
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
    cre_ops_list = [Indices([i for i in op.cre_ops if i.space not in cv]) for op in ops_list]
    ann_ops_list = [Indices([i for i in op.ann_ops if i.space not in cv]) for op in ops_list]

    # generate all possible sub-indices for each cre/ann operators of each second-quantized operator
    # [{nleg: [relative indices of current cre/ann operators]}, ...]
    cre_sub_indices = [{n_leg: [ele_ops for ele_ops in combinations(range(ops.size), n_leg)]
                        for n_leg in range(1, min(max_cu, ops.size) + 1)} for ops in cre_ops_list]
    ann_sub_indices = [{n_leg: [ele_ops for ele_ops in combinations(range(ops.size), n_leg)]
                        for n_leg in range(1, min(max_cu, ops.size) + 1)} for ops in ann_ops_list]

    # generate all possible partitions for k cre/ann legs for k cumulant
    n_sq_ops = len(ops_list)
    partitions = []
    for k in range(1, max_cu + 1):
        partitions += [part for part in integer_partition(k) if len(part) <= n_sq_ops]

    def generate_half_cumulant_contractions(part, op_indices, is_cre):
        """
        Generate cumulant contractions for pure creation and annihilation operators
        :param part: a partition of k for k-cumulant
        :param op_indices: a list of indices of chosen operator in the input operator list
        :param is_cre: True for creation part, False for annihilation part
        :return: generate viable half cumulant contractions
        """
        pure_ops_lists = cre_lists if is_cre else ann_lists
        pure_ops_sub_indices = cre_sub_indices if is_cre else ann_sub_indices

        def check_viable():
            for i, n_leg in zip(op_indices, part):
                if len(pure_ops_lists[i]) < nleg:
                    return False
            return True

        if check_viable():
            temp = [pure_ops_sub_indices[i][nleg] for i, nleg in zip(op_indices, part)]
            for sub_indices_product in product(*temp):
                result = []
                for sub_indices, i in zip(sub_indices_product, sqop_ids):
                    for index in sub_indices:
                        result.append((i, index))
                yield result


def generate_elementary_contractions(sqops, maxcu=3):
    """
    Generate all elementary contractions from a list of second-quantized operators.
    :param sqops: a list of SQOperators
    :param maxcu: max level of cumulants
    :return: a list of elementary contractions (represented by Indices)
    """

    # determine if maxcu makes sense (indices cannot be core or virtual)
    cv = ["c", "v"]
    ncre_sum = sum([op.ncre - op.Uindices.count_index_space(cv) for op in sqops])
    nann_sum = sum([op.nann - op.Lindices.count_index_space(cv) for op in sqops])
    maxcu_allowed = max(1, min(ncre_sum, nann_sum))
    if maxcu < 1 or maxcu > maxcu_allowed:
        maxcu = maxcu_allowed
        print("Max level cumulant is set to {}".format(maxcu_allowed))

    out = []

    # 1-cumulant (1-particle density)
    for i, left in enumerate(sqops):
        for right in sqops[i + 1:]:
            for u in left.Uindices:
                if u.space == 'v':
                    continue
                for l in right.Lindices:
                    if l.space == 'v':
                        continue
                    if len(space_relation[u.space] & space_relation[l.space]) != 0:
                        out.append(Indices([u, l]))

    # 1-hole density
    hole_start = len(out)
    hole_count = 0
    for i, left in enumerate(sqops):
        for right in sqops[i + 1:]:
            for l in left.Lindices:
                if l.space == 'c':
                    continue
                for u in right.Uindices:
                    if u.space == 'c':
                        continue
                    if len(space_relation[l.space] & space_relation[u.space]) != 0:
                        out.append(Indices([u, l]))
                        hole_count += 1

    # return if no need for cumulant
    if maxcu < 2:
        return out, (hole_start, hole_start + hole_count)

    # for cumulant, since ncre = nann, consider cre/ann separately
    nsqops = len(sqops)
    cre_lists = [Indices([i for i in op.Uindices if i.space not in cv]) for op in sqops]
    ann_lists = [Indices([i for i in op.Lindices if i.space not in cv]) for op in sqops]

    def generate_sub_indices(ops_lists):
        """
        Generate all possible sub-indices for each cre/ann string of each sqop.
        :param ops_lists: a list of cre/ann string
        :return: [{nleg: [relative indices of current cre/ann string]}, ...] for each sqop
        """
        return [{nleg: [ele_ops for ele_ops in combinations(range(ops_lists[i].size), nleg)]
                 for nleg in range(1, min(maxcu, ops_lists[i].size) + 1)}
                for i in range(nsqops)]

    cre_sub_indices = generate_sub_indices(cre_lists)
    ann_sub_indices = generate_sub_indices(ann_lists)

    # generate all possible partitions for k cre/ann legs for k cumulant
    partitions = []
    for k in range(1, maxcu + 1):
        for part in integer_partition(k):
            if len(part) <= nsqops:
                partitions.append(part)

    def check_viable(part, op_ids, lists):
        """
        Check if the partition is viable on the chosen sqops
        :param part: a partition
        :param op_ids: a list of sqop indices in sqops
        :param lists: list of cre/ann operators
        :return: True if viable, False otherwise
        """
        for i, nleg in zip(op_ids, part):
            if len(lists[i]) < nleg:
                return False
        return True

    def generate_half_cumulant_contractions(part, sqop_ids, is_cre):
        """
        Generate cumulant contractions for pure creation and annihilation operators
        :param part: a partition of k for k-cumulant
        :param sqop_ids: a list of indices of chosen sqop in sqops
        :param is_cre: True for creation part, False for annihilation part
        :return: generate viable half cumulant contractions
        """
        ops_lists = cre_lists if is_cre else ann_lists
        ops_sub_indices = cre_sub_indices if is_cre else ann_sub_indices

        if check_viable(part, sqop_ids, ops_lists):
            temp = [ops_sub_indices[i][nleg] for i, nleg in zip(sqop_ids, part)]
            for sub_indices_product in product(*temp):
                result = []
                for sub_indices, i in zip(sub_indices_product, sqop_ids):
                    for index in sub_indices:
                        result.append((i, index))
                yield result

    # cre/ann contractions from 2 to maxcu
    cre_results = {i: [] for i in range(2, maxcu + 1)}
    ann_results = {i: [] for i in range(2, maxcu + 1)}

    # generate cre/ann contractions separately
    for part_unique in partitions:
        nops = len(part_unique)
        cu_level = sum(part_unique)
        for part in multiset_permutations(part_unique):  # multiset permutation of integer partition
            for ops in combinations(range(nsqops), nops):  # choose nops from sqops list
                # creation combinations
                for indices in generate_half_cumulant_contractions(part, ops, True):
                    cre_results[cu_level].append(indices)
                # annihilation combinations
                for indices in generate_half_cumulant_contractions(part, ops, False):
                    ann_results[cu_level].append(indices)

    # now combine the cre/ann results
    for cu_level in range(2, maxcu + 1):
        for cre in cre_results[cu_level]:
            i_creops = [cre[i][0] for i in range(cu_level)]
            same_sqop_cre = i_creops.count(i_creops[0]) == len(i_creops)
            cre_indices = [cre_lists[cre[i][0]][cre[i][1]] for i in range(cu_level)]
            for ann in ann_results[cu_level]:
                if same_sqop_cre:
                    i_annops = [ann[i][0] for i in range(cu_level)]
                    same_sqop = i_annops.count(i_annops[0]) == len(i_annops) and i_creops[0] == i_annops[0]
                    if not same_sqop:
                        indices = cre_indices + [ann_lists[ann[i][0]][ann[i][1]] for i in range(cu_level)]
                        out.append(Indices(indices))
                else:
                    indices = cre_indices + [ann_lists[ann[i][0]][ann[i][1]] for i in range(cu_level)]
                    out.append(Indices(indices))

    return out, (hole_start, hole_start + hole_count)


def generate_operator_contractions(sqops, maxcu=3, maxmb=3, minmb=0, expand_hole=True):
    """
    Generate operator contractions for a list of SQOperator.
    :param sqops: a list of SQOperator to be contracted
    :param maxcu: max level of cumulant
    :param maxmb: max body term kept for return
    :param minmb: min body term kept for return
    :param expand_hole: expand hole density to Kronecker delta and 1-cumulant if True
    :return: a map of number of open indices to contractions
    """
    # generate elementary contractions
    elementary_contractions, (hole_start, hole_end) = generate_elementary_contractions(sqops, maxcu)
    n_ele_con = len(elementary_contractions)

    # generate incompatible contractions between elementary contractions
    # the list index of elementary_contractions is saved
    incompatible_elementary = {i: set() for i in range(n_ele_con)}
    for i in range(n_ele_con):
        ele_i = elementary_contractions[i]
        for j in range(i + 1, n_ele_con):
            ele_j = elementary_contractions[j]
            if len(ele_i.set & ele_j.set) != 0:
                incompatible_elementary[i].add(j)
                incompatible_elementary[j].add(i)

    # backtracking (similar to generating sublists)
    contractions = []
    composite_contractions_backtracking(set(range(n_ele_con)), set(), incompatible_elementary, contractions)

    # translate contractions to readable form
    results = defaultdict(list)

    base_order = []
    uindices_set, lindices_set = set(), set()
    for sqop in sqops:
        base_order += sqop.Uindices.indices + sqop.Lindices.indices[::-1]
        uindices_set |= sqop.Uindices.set
        lindices_set |= sqop.Lindices.set
    nops = len(base_order)

    for con in contractions:  # con is a list of indices of elementary contractions
        nops_con = sum([elementary_contractions[i].size for i in con])
        nops_open = nops - nops_con
        if 2 * minmb <= nops_open <= 2 * maxmb:
            # figure out the list of densities
            list_of_tensors = []
            current_order = []
            for i_con in con:
                ele_con = elementary_contractions[i_con]
                mid = ele_con.size // 2
                if hole_start <= i_con < hole_end:
                    current_order += ele_con.indices[::-1]  # ann comes before cre for hole density
                    list_of_tensors.append(HoleDensity(ele_con.indices[:mid], ele_con.indices[mid:]))
                else:
                    current_order += ele_con.indices
                    # need to reverse the ordering for annihilation operators
                    list_of_tensors.append(Cumulant(ele_con.indices[:mid], ele_con.indices[-1:-mid-1:-1]))

            # expand hole densities to delta - lambda1
            if expand_hole:
                signed_list_tensors = expand_hole_densities(list_of_tensors)
            else:
                signed_list_tensors = [(1, list_of_tensors)]

            # sort the indices for uncontracted operators
            uindices, lindices = Indices([]), Indices([])
            if nops_open != 0:
                uindices = Indices(sorted(uindices_set - set(current_order)))
                lindices = Indices(sorted(lindices_set - set(current_order)))
                current_order += uindices.indices + lindices.indices[::-1]

            sign = (-1) ** (Indices(base_order).count_permutations(Indices(current_order)))
            for _sign, list_of_tensors in signed_list_tensors:
                results[nops_open].append((sign * _sign, list_of_tensors, SQOperator(uindices, lindices)))

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

        available.add(temp)  # unchoose


def expand_hole_densities(list_of_tensors):
    """
    Expand all the hole densities in the list_of_tensors.
    :param list_of_tensors: a list of Tensor objects
    :return: a list of (sign, expanded Tensor objects)
    """
    out = []

    good_tensors = []
    hole_densities = []
    for tensor in list_of_tensors:
        if isinstance(tensor, HoleDensity):
            hole_densities.append(tensor)
        else:
            good_tensors.append(tensor)

    nhole = len(hole_densities)
    if nhole == 0:
        out.append((1, list_of_tensors))
    else:
        for temp in product(range(2), repeat=nhole):
            sign = (-1) ** sum(temp)
            expanded = []
            for i in range(nhole):
                Uindices, Lindices = hole_densities[i].Uindices, hole_densities[i].Lindices
                if temp[i] == 0:
                    expanded.append(Kronecker(Uindices, Lindices))
                else:
                    if Uindices[0].space != 'v' and Lindices[0].space != 'v':
                        expanded.append(Cumulant(Uindices, Lindices))
            if len(expanded) != nhole:
                continue
            out.append((sign, good_tensors + expanded))

    return out


a = SQOperator(["g0", "g1"], ["g2", "g3"])
b = SQOperator(["p0", "p1"], ["h0", "h1"])
c = SQOperator(["p2", "p3"], ["h2", "h3"])
d = SQOperator(["g{}".format(i) for i in range(3)], ["g{}".format(i) for i in range(3, 6)])
e = SQOperator(["p{}".format(i) for i in range(3)], ["h{}".format(i) for i in range(3)])

# with Timer("elementary contraction for V * T2 * T2, maxcu = 4") as t:
#     out, (i,j) = generate_elementary_contractions([a, b, c], maxcu=4)
#     # print(len(out))
#     # for i in out:
#     #     print(i)
# # with Timer("elementary contraction for V * T2 * T2, maxcu = 3") as t:
# #     out, (i,j) = generate_elementary_contractions([a, b, c], maxcu=3)
# # with Timer("elementary contraction for W * T3, maxcu = 10") as t:
# #     out, (i,j) = generate_elementary_contractions([d, e], maxcu=5)
# with Timer("elementary contraction for W * T3, maxcu = 3") as t:
#     out, (i,j) = generate_elementary_contractions([d, e], maxcu=3)
#     # print(len(out))
#     # for i in out:
#     #     print(i)

a = SQOperator(["g0", "g1", "g2"], ["g3", "g4", "g5"])
b = SQOperator(["v0", "v1", "v2"], ["c0", "c1", "c2"])
# b = SQOperator(["v4"], ["c5"])
# b = SQOperator(["p0", "p1"], ["h0", "h1"])
# a = SQOperator(["g0", "g1", "g2"], ["g3", "g4", "g5"])
# b = SQOperator(["p0", "p1", "p2"], ["h0", "h1", "h2"])
# ab = generate_operator_contractions([a, b])
# for k, vs in ab.items():
#     for v in vs:
#         print(k, v)


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




def Hamiltonian_operator(k, start=0):
    coeff = factorial(k) ** 2
    r0, r1, r2 = start, start + k, start + 2 * k
    tensor = Hamiltonian(["g{}".format(i) for i in range(r1, r2)],
                         ["g{}".format(i) for i in range(r0, r1)])
    sqop = SQOperator(["g{}".format(i) for i in range(r0, r1)],
                      ["g{}".format(i) for i in range(r1, r2)])
    return Term([tensor], sqop, 1 / coeff)


def cluster_operator(k, start=0, deexcitation=False, hole_index='h', particle_index='p'):
    coeff = factorial(k) ** 2
    r0, r1 = start, start + k
    hole = ["{}{}".format(hole_index, i) for i in range(r0, r1)]
    particle = ["{}{}".format(particle_index, i) for i in range(r0, r1)]
    if deexcitation:
        tensor = ClusterAmp(particle, hole)
        sqop = SQOperator(hole, particle)
    else:
        tensor = ClusterAmp(hole, particle)
        sqop = SQOperator(particle, hole)
    return Term([tensor], sqop, 1 / coeff)

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

import multiprocessing
import sys
from joblib import Parallel, delayed
from copy import deepcopy
from collections import defaultdict
from itertools import combinations, product
from sympy.combinatorics import Permutation
from sympy.utilities.iterables import multiset_permutations
from integer_partition import integer_partition
from mo_space import space_relation
from Indices import Indices, IndicesSpinOrbital
from IndicesPair import IndicesPair
from SQOperator import SecondQuantizedOperator
from Tensor import make_tensor_preset, HoleDensity, Kronecker, Cumulant
from timeit import default_timer as timer


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
                        out_list.append(make_tensor_preset('cumulant', [u], [l], 'spin-orbital'))
            for l in left.ann_ops:
                if l.space == 'c':
                    continue
                for u in right.cre_ops:
                    if u.space == 'c':
                        continue
                    if len(space_relation[u.space] & space_relation[l.space]) != 0:
                        out_list.append(make_tensor_preset('hole_density', [u], [l], 'spin-orbital'))
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
                    out_list.append(make_tensor_preset('cumulant', cre_indices, ann_indices, 'spin-orbital'))

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
    start = timer()
    elementary_contractions = generate_elementary_contractions(ops_list, max_cu)
    end = timer()
    print(f"generate elementary contractions: {end - start:.6f}s, number of elementary contractions: {len(elementary_contractions)}")
    n_ele_con = len(elementary_contractions)

    # generate incompatible contractions between elementary contractions
    # the list index of elementary_contractions is saved
    start = timer()
    incompatible_elementary = {i: set() for i in range(n_ele_con)}
    for i, ele_i in enumerate(elementary_contractions):
        for j, ele_j in enumerate(elementary_contractions[i + 1:], i + 1):
            if ele_i.any_overlapped_indices(ele_j):
                incompatible_elementary[i].add(j)
                incompatible_elementary[j].add(i)
    end = timer()
    print(f"incompatible elementary contractions: {end - start:.6f}s")

    # backtracking (similar to generating sub-lists)
    start = timer()
    contractions = []
    composite_contractions_backtracking(set(range(n_ele_con)), set(), incompatible_elementary, contractions)
    end = timer()
    print(f"backtracking contractions: {end - start:.6f}s")

    # translate contractions to readable form
    start = timer()
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
    end = timer()
    print(f"translate contractions: {end - start:.6f}s")

    return results


def calculate(func, args):
    return func(*args)


def calculatestar(args):
    return calculate(*args)


def process_contractions(contraction, elementary_contractions, expand_hole,
                         base_order_map, upper_indices_set, lower_indices_set):
    start = timer()
    list_of_densities = []
    current_order = []
    upper_contracted, lower_contracted = set(), set()

    xxx = timer()
    for i in contraction:
        ele_con = elementary_contractions[i]

        list_of_densities.append(ele_con)

        # creation (left) or annihilation (right)
        left, right = ele_con.upper_indices, ele_con.lower_indices
        upper_contracted.update(left.indices)
        lower_contracted.update(right.indices)

        if isinstance(ele_con, HoleDensity):
            left, right = right, left

        current_order += left.indices + right.indices[::-1]
    print(f"loop: {timer() - xxx: .6f}")

    # expand hole densities to delta - lambda_1
    xxx = timer()
    sign_densities_pairs = expand_hole_densities(list_of_densities) if expand_hole else [(1, list_of_densities)]
    print(f"expand: {timer() - xxx: .6f}")

    # sort the open indices
    xxx = timer()
    n_open = len(base_order_map) - len(current_order)
    if n_open != 0:
        open_upper_indices = IndicesSpinOrbital(sorted(upper_indices_set - upper_contracted))
        open_lower_indices = IndicesSpinOrbital(sorted(lower_indices_set - lower_contracted))
        current_order += open_upper_indices.indices + open_lower_indices.indices[::-1]
    else:
        open_upper_indices, open_lower_indices = IndicesSpinOrbital([]), IndicesSpinOrbital([])
    sq_op = SecondQuantizedOperator(IndicesPair(open_upper_indices, open_lower_indices))
    print(f"sqop: {timer() - xxx: .6f}")

    # determine sign
    xxx = timer()
    sign = (-1) ** Permutation([base_order_map[i] for i in current_order]).inversions()
    print(f"perm: {timer() - xxx: .6f}")
    print(f"process contraction: {timer() - start:.6f}s")

    return n_open, [(sign * _sign, list_of_densities, sq_op) for _sign, list_of_densities in sign_densities_pairs]


def multiprocess_contractions(list_of_densities, n_open, expand_hole,
                              base_order_map, upper_indices_set, lower_indices_set):
    current_order = list()
    for dens in list_of_densities:
        left, right = dens.upper_indices, dens.lower_indices
        if isinstance(dens, HoleDensity):
            left, right = right, left
        current_order += left.indices + right.indices[::-1]

    if n_open != 0:
        contracted = set(current_order)
        open_upper_indices = IndicesSpinOrbital(sorted(upper_indices_set - contracted))
        open_lower_indices = IndicesSpinOrbital(sorted(lower_indices_set - contracted))
        current_order += open_upper_indices.indices + open_lower_indices.indices[::-1]
    else:
        open_upper_indices, open_lower_indices = IndicesSpinOrbital([]), IndicesSpinOrbital([])
    sq_op = SecondQuantizedOperator(IndicesPair(open_upper_indices, open_lower_indices))

    sign = (-1) ** Permutation([base_order_map[i] for i in current_order]).inversions()
    sign_densities_pairs = expand_hole_densities(list_of_densities) if expand_hole else [(1, list_of_densities)]
    return n_open, [(sign * _sign, list_of_densities, sq_op) for _sign, list_of_densities in sign_densities_pairs]


def generate_operator_contractions_new(ops_list, max_cu=3, max_n_open=6, min_n_open=0,
                                       expand_hole=True, n_process=1):
    """
    Generate operator contractions for a list of SQOperator.
    :param ops_list: a list of SecondQuantizedOperator to be contracted
    :param max_cu: max level of cumulant
    :param max_n_open: max number of open indices for contractions kept for return
    :param min_n_open: min number of open indices for contractions kept for return
    :param expand_hole: expand hole density to Kronecker delta and 1-cumulant if True
    :return: a map of number of open indices to contractions
    """
    # original ordering of the second-quantized operators
    base_order_indices = []
    upper_indices_set, lower_indices_set = set(), set()
    for sq_op in ops_list:
        base_order_indices += sq_op.cre_ops.indices + sq_op.ann_ops.indices[::-1]
        upper_indices_set.update(sq_op.cre_ops.indices)
        lower_indices_set.update(sq_op.ann_ops.indices)
    n_indices = len(base_order_indices)
    base_order_map = {v: i for i, v in enumerate(base_order_indices)}

    # generate elementary contractions
    start = timer()
    elementary_contractions = generate_elementary_contractions(ops_list, max_cu)
    end = timer()
    n_ele_con = len(elementary_contractions)
    if n_ele_con > 1000:
        sys.setrecursionlimit(n_ele_con)
    print(f"generate elementary contractions: {end - start:.6f}s, number of elementary contractions: {n_ele_con}")

    # generate incompatible contractions between elementary contractions
    # the list index of elementary_contractions is saved
    start = timer()
    incompatible_elementary = defaultdict(set)
    for i, ele_i in enumerate(elementary_contractions):
        for j, ele_j in enumerate(elementary_contractions[i + 1:], i + 1):
            if ele_i.any_overlapped_indices(ele_j):
                incompatible_elementary[i].add(j)
                incompatible_elementary[j].add(i)
    end = timer()
    print(f"incompatible elementary contractions: {end - start:.6f}s")

    # backtracking (similar to generating sub-lists)
    start = timer()
    contractions = []
    composite_contractions_backtracking(set(range(n_ele_con)), set(), incompatible_elementary, contractions)
    end = timer()
    print(f"backtracking contractions: {end - start:.6f}s")
    print(f"number of contractions: {len(contractions)}")

    # output
    results = defaultdict(list)

    start = timer()
    if n_process == 1:
        cu1 = (max_cu == 1)
        for con in contractions:
            if cu1:
                n_open = n_indices - 2 * len(con)
            else:
                n_open = n_indices - 2 * sum(elementary_contractions[i].n_upper for i in con)
            if not (min_n_open <= n_open <= max_n_open):
                continue

            list_of_densities = []
            current_order = []

            for i in con:
                ele_con = elementary_contractions[i]

                list_of_densities.append(ele_con)

                # creation (left) or annihilation (right)
                left, right = ele_con.upper_indices, ele_con.lower_indices
                if isinstance(ele_con, HoleDensity):
                    left, right = right, left
                current_order += left.indices + right.indices[::-1]

            # expand hole densities to delta - lambda_1
            sign_densities_pairs = expand_hole_densities(list_of_densities) if expand_hole else [(1, list_of_densities)]

            # sort the open indices
            if n_open != 0:
                contracted = set(current_order)
                open_upper_indices = IndicesSpinOrbital(sorted(upper_indices_set - contracted))
                open_lower_indices = IndicesSpinOrbital(sorted(lower_indices_set - contracted))
                current_order += open_upper_indices.indices + open_lower_indices.indices[::-1]
            else:
                open_upper_indices, open_lower_indices = IndicesSpinOrbital([]), IndicesSpinOrbital([])
            sq_op = SecondQuantizedOperator(IndicesPair(open_upper_indices, open_lower_indices))

            # determine sign
            sign = (-1) ** Permutation([base_order_map[i] for i in current_order]).inversions()

            results[n_open] += [(sign * _s, list_of_densities, sq_op) for _s, list_of_densities in sign_densities_pairs]
    else:
        # filter contractions
        translated = list()
        if max_cu == 1:
            for con in contractions:
                n_open = n_indices - 2 * len(con)
                if not (min_n_open <= n_open <= max_n_open):
                    continue
                translated.append((n_open, [elementary_contractions[i] for i in con]))
        else:
            for con in contractions:
                n_open = n_indices - 2 * sum(elementary_contractions[i].n_upper for i in con)
                if not (min_n_open <= n_open <= max_n_open):
                    continue
                translated.append((n_open, [elementary_contractions[i] for i in con]))

        # xxx = Parallel(n_jobs=n_process)(delayed(multiprocess_contractions)(list_of_densities, n_open, expand_hole,
        #                                                   base_order_map, upper_indices_set, lower_indices_set) for n_open, list_of_densities in translated)
        # for n_open, sign_densities_ops in xxx:
        #     results[n_open] += sign_densities_ops

        # with multiprocessing.Pool(n_process) as pool:
        #     tasks = list()
        #     for n_open, list_of_densities in translated:
        #         tasks.append((multiprocess_contractions, (list_of_densities, n_open, expand_hole,
        #                                                   base_order_map, upper_indices_set, lower_indices_set)))
        #     imap_unordered_it = pool.imap_unordered(calculatestar, tasks)
        #     for n_open, sign_densities_ops in imap_unordered_it:
        #         results[n_open] += sign_densities_ops

        # if n_process == 1:
        #     for contraction in contractions:
        #         # contraction is a list of indices of elementary contractions
        #         n_sq_ops_open, sign_densities_ops = process_contractions(contraction, elementary_contractions, expand_hole,
        #                                                                  base_order_map,
        #                                                                  upper_indices_set, lower_indices_set)
        #         results[n_sq_ops_open] += sign_densities_ops
        # else:
        # xxx = Parallel(n_jobs=n_process)(delayed(process_contractions)(con, elementary_contractions,
        #                                                                expand_hole, base_order_map,
        #                                                                upper_indices_set, lower_indices_set) for con in contractions)
        # for n_open, sign_densities_ops in xxx:
        #     results[n_open] += sign_densities_ops
        # manager = multiprocessing.Manager()
        # elementary_contractions = manager.list(elementary_contractions)
        # # base_order_map = manager.dict(*base_order_map)
        # with multiprocessing.Pool(n_process) as pool:
        #     tasks = []
        #     for con in contractions:
        #         tasks.append((process_contractions,
        #                       (con, elementary_contractions, expand_hole,
        #                        base_order_map, upper_indices_set, lower_indices_set)))
        #     imap_unordered_it = pool.imap_unordered(calculatestar, tasks)
        #     for n_open, sign_densities_ops in imap_unordered_it:
        #         results[n_open] += sign_densities_ops
    # for contraction in contractions:
    #
    #     n_sq_ops_open = n_indices - sum([elementary_contractions[i].size for i in contraction])
    #
    #     if min_n_open <= n_sq_ops_open <= max_n_open:
    #
    #         list_of_densities = []
    #         current_order = []
    #         upper_contracted, lower_contracted = set(), set()
    #
    #         for i in contraction:
    #             ele_con = elementary_contractions[i]
    #
    #             list_of_densities.append(ele_con)
    #
    #             # creation (left) or annihilation (right)
    #             left, right = ele_con.upper_indices, ele_con.lower_indices
    #             upper_contracted.update(left.indices)
    #             lower_contracted.update(right.indices)
    #
    #             if isinstance(ele_con, HoleDensity):
    #                 left, right = right, left
    #
    #             current_order += left.indices + right.indices[::-1]
    #
    #         # expand hole densities to delta - lambda_1
    #         sign_densities_pairs = expand_hole_densities(list_of_densities) if expand_hole else [(1, list_of_densities)]
    #
    #         # sort the open indices
    #         open_upper_indices, open_lower_indices = IndicesSpinOrbital([]), IndicesSpinOrbital([])
    #         if n_sq_ops_open != 0:
    #             open_upper_indices = IndicesSpinOrbital(sorted(upper_indices_set - upper_contracted))
    #             open_lower_indices = IndicesSpinOrbital(sorted(lower_indices_set - lower_contracted))
    #             current_order += open_upper_indices.indices + open_lower_indices.indices[::-1]
    #         sq_op = SecondQuantizedOperator(IndicesPair(open_upper_indices, open_lower_indices))
    #
    #         # determine sign and push to results
    #         sign = (-1) ** Permutation([base_order_map[i] for i in current_order]).inversions()
    #         for _sign, list_of_densities in sign_densities_pairs:
    #             results[n_sq_ops_open].append((sign * _sign, list_of_densities, sq_op))
    end = timer()
    print(f"translate contractions: {end - start:.6f}s")

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
    :return: generate tuples of (sign, expanded Tensor objects)
    """
    good_tensors, hole_densities = [], []
    for tensor in list_of_tensors:
        hole_densities.append(tensor) if isinstance(tensor, HoleDensity) else good_tensors.append(tensor)

    # if hole density contain any virtual index, it can only be delta
    skip_cumulant = [hole.upper_indices[0].space == 'v' or hole.lower_indices[0].space == 'v'
                     for hole in hole_densities]

    n_hole = len(hole_densities)
    if n_hole == 0:
        yield 1, list_of_tensors
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

            yield sign, good_tensors + expanded

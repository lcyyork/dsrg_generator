import multiprocessing
import sys
import numpy as np
from joblib import Parallel, delayed
from copy import deepcopy
from collections import defaultdict, Counter
from itertools import combinations, product, chain
from sympy.combinatorics import Permutation
from sympy.utilities.iterables import multiset_permutations
from integer_partition import integer_partition
from mo_space import space_relation
from Indices import Indices, IndicesSpinOrbital
from IndicesPair import IndicesPair
from SQOperator import SecondQuantizedOperator
from Tensor import make_tensor_preset, HoleDensity, Kronecker, Cumulant
from timeit import default_timer as timer


def compute_elementary_contractions_list(ops_list, max_cu=3):
    """
    Generate all elementary contractions from a list of second-quantized operators.
    :param ops_list: a list of SecondQuantizedOperator objects
    :param max_cu: the max level of cumulants
    :return: a list of elementary contractions represented by HoleDensity and Cumulant
    """
    return list(chain(*compute_elementary_contractions_categorized(ops_list, max_cu).values()))


def compute_elementary_contractions_categorized(ops_list, max_cu=3):
    """
    Generate all elementary contractions from a list of second-quantized operators.
    :param ops_list: a list of SecondQuantizedOperator objects
    :param max_cu: the max level of cumulants
    :return: a dictionary of {(connected op indices): a list of contractions represented by HoleDensity and Cumulant}
    """
    for op in ops_list:
        if not isinstance(op, SecondQuantizedOperator):
            raise TypeError(f"Invalid type in ops_list, given '{op.__class__.__name__}', "
                            f"required 'SecondQuantizedOperator'.")
        if op.type_of_indices != IndicesSpinOrbital:
            raise NotImplementedError(f"Contractions only supports spin-orbital indices now.")
    if not isinstance(max_cu, int):
        raise TypeError(f"Invalid type of max_cu, given '{max_cu.__class__.__name__}', required 'int'.")

    max_cu_allowed = check_max_cu(ops_list, max_cu)

    out = compute_elementary_contractions_pairwise(ops_list)
    if max_cu_allowed < 2:
        return out

    out.update(compute_elementary_contractions_cumulant(ops_list, max_cu_allowed))

    return out


def check_max_cu(ops_list, max_cu):
    """
    Check if the input max_cu makes sense (cumulant indices cannot be core or virtual).
    :param ops_list: a list of SecondQuantizedOperator objects
    :param max_cu: the max level of cumulants
    :return: the max valid level of cumulants
    """
    cv = ["c", "v"]
    n_valid_cre = sum([op.n_cre - op.cre_ops.count_index_space(cv) for op in ops_list])
    n_valid_ann = sum([op.n_ann - op.ann_ops.count_index_space(cv) for op in ops_list])
    max_cu_allowed = max(1, min(n_valid_cre, n_valid_ann))
    if max_cu < 1 or max_cu > max_cu_allowed:
        print(f"Max cumulant level is set to {max_cu_allowed}.")
        return max_cu_allowed
    return max_cu


def compute_elementary_contractions_pairwise(ops_list):
    """
    Generate all single pairwise contractions from a list of second-quantized operators.
    :param ops_list: a list of SecondQuantizedOperator objects
    :return: a dictionary of {(connected op indices): a list of contractions, i.e., HoleDensity or Cumulant}
    """
    out = defaultdict(list)

    for i, left in enumerate(ops_list):
        for j, right, in enumerate(ops_list[i + 1:], i + 1):

            # 1-cumulant: left cre + right ann
            for upper in left.cre_ops:
                if upper.space == 'v':
                    continue
                for lower in right.ann_ops:
                    if lower.space == 'v':
                        continue
                    if len(space_relation[upper.space] & space_relation[lower.space]) != 0:
                        out[(i, j)].append(make_tensor_preset('cumulant', [upper], [lower], 'spin-orbital'))

            # 1-hole-density: left ann + right cre
            for lower in left.ann_ops:
                if lower.space == 'c':
                    continue
                for upper in right.cre_ops:
                    if upper.space == 'c':
                        continue
                    if len(space_relation[upper.space] & space_relation[lower.space]) != 0:
                        out[(j, i)].append(make_tensor_preset('hole_density', [upper], [lower], 'spin-orbital'))

    return out


def compute_elementary_contractions_cumulant(ops_list, max_cu):
    """
    Generate all cumulant-type elementary contractions from a list of second-quantized operators.
    :param ops_list: a list of SecondQuantizedOperator objects
    :param max_cu: the max level of cumulants
    :return: a dictionary of {(connected op indices): a list of contractions, i.e., Cumulant}
    """
    out = defaultdict(list)

    # for cumulant, since n_cre = n_ann, consider cre/ann separately
    cv = ["c", "v"]
    cre_ops_list = [IndicesSpinOrbital([i for i in op.cre_ops if i.space not in cv]) for op in ops_list]
    ann_ops_list = [IndicesSpinOrbital([i for i in op.ann_ops if i.space not in cv]) for op in ops_list]

    # generate all possible pure creation or annihilation for cumulant contractions
    ann_results = compute_elementary_contractions_half_cumulant(ann_ops_list, max_cu)
    cre_results = compute_elementary_contractions_half_cumulant(cre_ops_list, max_cu)

    # now combine the cre/ann results
    for cu_level in range(2, max_cu + 1):
        for cre in cre_results[cu_level]:  # cre contains cu_level numbers of pairs of (i_macro, i_micro)

            i_sq_op_cre = [i_macro for i_macro, _ in cre]
            same_sq_op_cre = i_sq_op_cre.count(i_sq_op_cre[0]) == len(i_sq_op_cre)

            cre_indices = [cre_ops_list[i_macro][i_micro] for i_macro, i_micro in cre]

            for ann in ann_results[cu_level]:

                # skip when cre and ann belong to same operator
                i_sq_op_ann = [i_macro for i_macro, _ in ann]
                same_sq_op_ann = i_sq_op_ann.count(i_sq_op_ann[0]) == len(i_sq_op_ann)

                if same_sq_op_cre and same_sq_op_ann and i_sq_op_cre[0] == i_sq_op_ann[0]:
                    continue
                else:
                    ann_indices = [ann_ops_list[i_macro][i_micro] for i_macro, i_micro in ann]
                    out[tuple(i_sq_op_cre + i_sq_op_ann)].append(make_tensor_preset('cumulant', cre_indices,
                                                                                    ann_indices, 'spin-orbital'))

    return out


def compute_elementary_contractions_half_cumulant(pure_ops_list, max_cu):
    """
    Generate all possible combinations of pure creation and annihilation operators for cumulant contractions.
    :param pure_ops_list: a list of pure creation or annihilation indices for each input operator
    :param max_cu: the max level of cumulants
    :return: {cumulant level: [[n_cumulant of chosen indices (op index, relative index)], ...]}

    Consider the following list of pure cre/ann operators: pure_ops_list = [[a, b], [c, d], [e, f, g]]
    and we want to obtain 2, 3, 4 cumulants, i.e., max_cu = 4.
    Note that there are three macro operators: [a, b], [c, d], [e, f, g]
    and the first macro operator has two micro operators: a, b

    This function will do the following:
        (1) generate all unique integer partitions of 2, 3, and 4, i.e, partition legs for k cumulant
            2 = 1 + 1                       # note a)
            3 = 2 + 1 = 1 + 1 + 1           # note b)
            4 = 3 + 1 = 2 + 2 = 2 + 1 + 1   # note c)
            Note:
                a) single partition is included, e.g, [2] is valid
                a) only unique partitions, e.g., 1 + 2 is ignored
                b) len(pure_ops_list) = 3 => number of partitions must <= 3
        (2) generate all possible sub-indices for each macro operator in pure_ops_list
            and return [{n_leg: [relative indices of the current string of cre/ann operators]}, ...]
            for the example, we should get:
                [{1: [(0,), (1,)], 2: [(0, 1)]},
                 {1: [(0,), (1,)], 2: [(0, 1)]},
                 {1: [(0,), (1,), (2,)], 2: [(0, 1), (0, 2), (1, 2)], 3: [(0, 1, 2)]}]
        (3) loop over integer partitions and choose the corresponding sub-indices partitions
            for example,
                => 2 + 1, i.e, select 2 legs from one macro operator and 1 leg from one of the others
                => consider multiset permutations, 2 + 1 = 1 + 2. another example: 2 + 1 + 1 = 1 + 2 + 1 = 1 + 1 + 2
                => choose 2 macro operators from the given three
                => say we choose the first two macro operators, where the first contributes two legs:
                    that is, [(0, 1)] from the first macro operator, [(0,), (1,)] from the second macro operator
                => generate all possible selections by a nested loop
                    => cartesian product between [(0, 1)] and [(0,), (1,)], i.e., a nested loop
                    => combine with macro operator index and get [(0, 0), (0, 1), (1, 0)], [(0, 0), (0, 1), (1, 1)]
    """
    results = {i: [] for i in range(2, max_cu + 1)}

    # generate all possible unique partitions for k cre/ann legs for k cumulant
    macro_size = len(pure_ops_list)
    unique_partitions = [part for k in range(2, max_cu + 1)
                         for part in integer_partition(k) if len(part) <= macro_size]

    # generate all possible sub-indices for each macro operator
    sub_indices = [{n_leg: [ele_ops for ele_ops in combinations(range(ops.size), n_leg)]
                    for n_leg in range(1, min(max_cu, ops.size) + 1)} for ops in pure_ops_list]

    for unique_partition in unique_partitions:
        n_macro = len(unique_partition)
        cu_level = sum(unique_partition)

        for leg_part in multiset_permutations(unique_partition):

            # choose n_macro from ops_list
            for macro_ops in combinations(range(macro_size), n_macro):

                # check if this partition is valid on these chosen macro operators
                if any([len(pure_ops_list[i]) < n_leg for i, n_leg in zip(macro_ops, leg_part)]):
                    continue

                # generate all possibilities
                for micro_ops_pro in product(*[sub_indices[i][n_leg] for i, n_leg in zip(macro_ops, leg_part)]):
                    results[cu_level].append([(i_macro, i_micro) for i_macro, micro_ops in zip(macro_ops, micro_ops_pro)
                                              for i_micro in micro_ops])

    return results


def compute_operator_contractions(ops_list, max_cu=3, max_n_open=6, min_n_open=0,
                                  for_commutator=False, expand_hole=True, n_process=1):
    """
    Generate operator contractions for a list of SQOperator.
    :param ops_list: a list of SecondQuantizedOperator to be contracted
    :param max_cu: max level of cumulant
    :param max_n_open: max number of open indices for contractions kept for return
    :param min_n_open: min number of open indices for contractions kept for return
    :param for_commutator: remove all-cumulant-type and disconnected contractions
    :param expand_hole: expand hole density to Kronecker delta and 1-cumulant if True
    :param n_process: the number of processes launched by multiprocessing
    :return: a list of contractions in terms of (sign, list_of_densities, sq_op)
    """
    max_cu_allowed = check_max_cu(ops_list, max_cu)

    # original ordering of the second-quantized operators
    base_order_indices = []
    upper_indices_set, lower_indices_set = set(), set()
    for sq_op in ops_list:
        base_order_indices += sq_op.cre_ops.indices + sq_op.ann_ops.indices[::-1]
        upper_indices_set.update(sq_op.cre_ops.indices)
        lower_indices_set.update(sq_op.ann_ops.indices)
    n_indices = len(base_order_indices)
    base_order_map = {v: i for i, v in enumerate(base_order_indices)}

    max_n_con, min_n_con = n_indices - min_n_open, n_indices - max_n_open

    if for_commutator:
        elementary_contractions = compute_elementary_contractions_categorized(ops_list, max_cu_allowed)
        elementary_contractions_macro = sorted(elementary_contractions.keys(), key=lambda x: (len(x), x), reverse=True)
        print(elementary_contractions)

        # integer partitions for memoization
        used_integer_partitions = defaultdict(list)

        # TODO: figure out the incompatible contractions


        # backtrack on elementary contracted macro operators
        for macro in contracted_operator_backtrack_macro(elementary_contractions_macro, [],
                                                         max_n_con, 0,
                                                         [sq_op.n_cre for sq_op in ops_list],
                                                         [sq_op.n_ann for sq_op in ops_list],
                                                         set(), True):
            n_con_one_sweep = sum(len(i) for i in macro)
            min_n_con = max(n_indices - max_n_open, n_con_one_sweep)

            if min_n_con == max_n_con:
                # TODO
                print("choose one from each category", macro)
            else:
                count_con = {k: len(elementary_contractions[k]) - 1 for k in macro}  # -1 for must choose one
                count_n_body = defaultdict(int)
                for k, v in count_con.items():
                    count_n_body[len(k) // 2] += v

                max_cu_this_macro = max(len(i) for i in macro) // 2

                for n_fill in range((min_n_con - n_con_one_sweep) // 2, (max_n_con - n_con_one_sweep) // 2 + 1):
                    if n_fill == 0:
                        # TODO
                        print("choose one from each category, min", macro)
                    else:
                        if n_fill not in used_integer_partitions:
                            partitions = [Counter(part) for part in integer_partition(n_fill)
                                          if all(n <= max_cu_this_macro for n in part)]
                            used_integer_partitions[n_fill] = partitions
                        else:
                            partitions = used_integer_partitions[n_fill]

                        # TODO: some more filtering
                        for part in partitions:
                            print(macro, part)
                            for macro_addon in autocomplete_macro_contractions_backtrack(macro, part, {**count_con},
                                                                                            {**count_n_body}, []):
                                macro_complete = macro + macro_addon
                                print("choose multiple", macro_complete)




        #
        # # TODO: generate actual contractions
        # incompatible_contractions = defaultdict(list)
        # for ops_ids in composite_contractions_ops_id[:200]:
        #     print(ops_ids)
        #     n_ops = len(ops_ids)
        #     size = sum([len(i) for i in ops_ids])
        #
        #     print(max(n_indices - max_n_open - size, 0) // 2, (n_indices - min_n_open - size) // 2)


    else:
        # TODO: use original implementation, need to include the non-contracted term
        elementary_contractions = compute_elementary_contractions_list(ops_list, max_cu)

    # TODO: translate contractions and return results


def contracted_operator_backtrack_macro(available, chosen, n_con_max, n_con_so_far,
                                        cre_available, ann_available, ops_so_far, all_cu_so_far):
    """
    Generate all possible connected macro operator contractions
    :param available: a list of unexplored elementary contracted macro operators (tuple of operator indices)
    :param chosen: a list of chosen contracted macro operators
    :param n_con_max: the max number of contractions specified by the user
    :param n_con_so_far: the number of contractions so far
    :param cre_available: a list of numbers of micro creation operators for each macro operator
    :param ann_available: a list of numbers of micro annihilation operators for each macro operator
    :param ops_so_far: a set of macro operator indices so far
    :param all_cu_so_far: a boolean to represent if contractions chosen so far are all cumulants

    Consider the following list of elementary contractions from T1^\dag * H2 * T1 * T1 up to 2-cumulant:
    [(1, 1, 0, 1), (1, 1, 0, 2), (1, 1, 0, 3), (1, 1, 1, 2), (1, 1, 1, 3), (1, 1, 2, 3), (0, 1, 1, 1),
     (0, 1, 0, 1), (0, 1, 0, 2), (0, 1, 0, 3), (0, 1, 1, 2), (0, 1, 1, 3), (0, 1, 2, 3), (0, 2, 1, 1),
     (0, 2, 0, 1), (0, 2, 0, 2), (0, 2, 0, 3), (0, 2, 1, 2), (0, 2, 1, 3), (0, 2, 2, 3), (0, 3, 1, 1),
     (0, 3, 0, 1), (0, 3, 0, 2), (0, 3, 0, 3), (0, 3, 1, 2), (0, 3, 1, 3), (0, 3, 2, 3), (1, 2, 1, 1),
     (1, 2, 0, 1), (1, 2, 0, 2), (1, 2, 0, 3), (1, 2, 1, 2), (1, 2, 1, 3), (1, 2, 2, 3), (1, 3, 1, 1),
     (1, 3, 0, 1), (1, 3, 0, 2), (1, 3, 0, 3), (1, 3, 1, 2), (1, 3, 1, 3), (1, 3, 2, 3), (2, 3, 1, 1),
     (2, 3, 0, 1), (2, 3, 0, 2), (2, 3, 0, 3), (2, 3, 1, 2), (2, 3, 1, 3), (2, 3, 2, 3),
     (0, 1), (1, 0), (0, 2), (2, 0), (0, 3), (3, 0), (1, 2), (2, 1), (1, 3), (3, 1), (2, 3), (3, 2)]

    Each tuple represent a possible contraction,
    e.g., (1, 1, 0, 1): 2-cumulant where 2 cre and 1 ann come from H2, and one ann comes from T1^\dag
    e.g., (1, 0): 1-hole-density where cre comes from H2 and ann comes from T1^\dag

    This function will ignore the following contractions:
    1) disconnected, e.g., [(0, 1)] or [(0, 1), (1, 0), (2, 3)]
    2) all cumulant, e.g., [(0, 1, 0, 2), (1, 2, 1, 3)]
    3) exceed cre/ann, e.g., [(1, 1, 0, 1), (1, 2), (2, 3)] because H2 has only 2 creations but 3 are in the list

    NOTE: The ordering of the available list matters!
          The backtracking algorithm starts from the end of the list:
          once pairwise contractions are all explored, we can terminate the entire process.
    """
    n_ops = len(cre_available)

    # base case, nothing to choose
    if len(available) == 0:
        if len(ops_so_far) == n_ops:
            yield chosen
    else:
        # explore when 1) number of contractions are not enough,
        #              2) previous contractions are not all cumulants
        if n_con_so_far < n_con_max and (len(available[-1]) == 2 if n_con_so_far == 0 else not all_cu_so_far):

            # choose an element
            temp = available.pop()

            temp_ops = set(temp)
            n_body = len(temp) // 2
            cre_count, ann_count = Counter(temp[:n_body]), Counter(temp[n_body:])

            # include this element when: 1) this element must connect to the previous ones,
            #                            2) both cre and ann won't exceed
            if ((not ops_so_far.isdisjoint(temp_ops)) or n_con_so_far == 0) and \
                    all(cre_available[k] - v >= 0 for k, v in cre_count.items()) and \
                    all(ann_available[k] - v >= 0 for k, v in ann_count.items()):

                chosen.append(temp)  # include this element

                available_new = _prune_available_contracted_operator(available, cre_available, ann_available,
                                                                     cre_count, ann_count)

                yield from contracted_operator_backtrack_macro(available_new[0], chosen,
                                                               n_con_max, n_con_so_far + 2 * n_body,
                                                               available_new[1], available_new[2],
                                                               ops_so_far | temp_ops, all_cu_so_far and n_body > 1)

                chosen.pop()  # not include this element

            # not include this element
            yield from contracted_operator_backtrack_macro(available, chosen, n_con_max, n_con_so_far,
                                                           cre_available, ann_available, ops_so_far, all_cu_so_far)

            # un-choose this element
            available.append(temp)
        else:
            yield from contracted_operator_backtrack_macro(list(), chosen, n_con_max, n_con_so_far,
                                                           cre_available, ann_available, ops_so_far, all_cu_so_far)


def _prune_available_contracted_operator(available, cre_available, ann_available, cre_count, ann_count):
    """
    Remove some of the unexplored elementary contractions based on cre/ann counts.
    :param available: a list of unexplored elementary contracted macro operators (tuple of operator indices)
    :param cre_available: a list of numbers of micro creation operators for each macro operator
    :param ann_available: a list of numbers of micro annihilation operators for each macro operator
    :param cre_count: a Counter {micro operator index: count} for creation operators of the current contraction
    :param ann_count: a Counter for annihilation operators of the current contraction
    :return: truncated unexplored list, updated cre/ann lists
    """

    def update_op_available(op_available, op_count):
        op_available_new = op_available[:]
        zero_count_id = set()
        for k, v in op_count.items():
            op_available_new[k] -= v
            if op_available_new[k] == 0:
                zero_count_id.add(k)
        return op_available_new, zero_count_id

    cre_available_new, cre_zero_count_id = update_op_available(cre_available, cre_count)
    ann_available_new, ann_zero_count_id = update_op_available(ann_available, ann_count)

    if len(cre_zero_count_id) == 0 and len(ann_zero_count_id) == 0:
        available_pruned = available
    else:
        available_pruned = list()
        for con_ops in available:
            if any(i_cre in cre_zero_count_id for i_cre in con_ops[:len(con_ops) // 2]):
                continue
            if any(i_ann in ann_zero_count_id for i_ann in con_ops[len(con_ops) // 2:]):
                continue
            available_pruned.append(con_ops)

    return available_pruned, cre_available_new, ann_available_new


def compute_incompatible_elementary_contractions(ele_cons):
    """
    Figure out incompatible elementary contractions from a list of second-quantized operators.
    :param elementary_contractions: elementary contractions generated by function elementary_contractions
    :return: a dictionary of {(connected op indices): a list of contractions represented by HoleDensity and Cumulant}
    """
    return


def autocomplete_macro_contractions_backtrack(choices, target, contraction_count, n_body_count, chosen):
    """
    Autocomplete the incomplete contractions from contracted_operator_backtrack_macro.
    :param choices: a connected macro operator contraction
    :param target: how many more contractions need to form a complete one
    :param contraction_count: a Counter for the remaining macro contraction count, i.e., {contraction: count}
    :param n_body_count: a Counter for the remaining n_body count (store this to avoid recompute from contraction_count)
    :param chosen: the chosen macro operator contractions

    Consider a connected macro operator contraction [(0, 1), (1, 0), (1, 2, 3, 4)].
    The target complete contractions needs three 1-body and one 2-body terms: target = {1: 3, 2: 1}.
    This function will generate
        [(1, 2, 3, 4), (0, 1), (0, 1), (1, 0)]
        [(1, 2, 3, 4), (0, 1), (1, 0), (1, 0)]
    """
    if any(v > n_body_count[k] for k, v in target.items()):
        return

    if len(choices) == 0:
        yield chosen
    else:
        # explore
        element = choices[-1]
        n_body = len(element) // 2

        # include this macro contraction (i.e., tuple of macro operator indices) element
        chosen.append(element)
        target[n_body] -= 1
        contraction_count[element] -= 1
        n_body_count[n_body] -= 1

        # filter choices when no longer need n_body terms (i.e. target[n_body] == 0) or remainder is zero
        choices_filtered = choices
        if target[n_body] == 0:
            choices_filtered = [i for i in choices if len(i) // 2 != n_body]
        else:
            if contraction_count[element] == 0:
                choices_filtered = choices[:-1]

        yield from autocomplete_macro_contractions_backtrack(choices_filtered, target,
                                                             contraction_count, n_body_count, chosen)

        # not include this element
        chosen.pop()
        target[n_body] += 1
        contraction_count[element] += 1
        n_body_count[n_body] += 1

        # remove this element for further exploring
        temp = contraction_count[element]
        n_body_count[n_body] -= contraction_count[element]
        contraction_count[element] = 0

        yield from autocomplete_macro_contractions_backtrack(choices[:-1], target,
                                                             contraction_count, n_body_count, chosen)

        n_body_count[n_body] += temp
        contraction_count[element] = temp


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

    out_list = list()

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

    # generate all possible pure creation or annihilation for cumulant contractions
    ann_results = generate_half_cumulant_contractions(ann_ops_list, ops_list, max_cu)
    cre_results = generate_half_cumulant_contractions(cre_ops_list, ops_list, max_cu)

    # now combine the cre/ann results
    for cu_level in range(2, max_cu + 1):
        for cre in cre_results[cu_level]:

            i_sq_op_cre = [cre[i][0] for i in range(cu_level)]
            same_sq_op_cre = i_sq_op_cre.count(i_sq_op_cre[0]) == len(i_sq_op_cre)

            cre_indices = [cre_ops_list[cre[i][0]][cre[i][1]] for i in range(cu_level)]

            for ann in ann_results[cu_level]:
                # skip when cre and ann belong to same operator
                i_sq_op_ann = [ann[i][0] for i in range(cu_level)]
                same_sq_op_ann = i_sq_op_ann.count(i_sq_op_ann[0]) == len(i_sq_op_ann)

                if same_sq_op_cre and same_sq_op_ann and cre[0][0] == ann[0][0]:
                    continue
                else:
                    ann_indices = [ann_ops_list[ann[i][0]][ann[i][1]] for i in range(cu_level)]
                    out_list.append(make_tensor_preset('cumulant', cre_indices, ann_indices, 'spin-orbital'))
    return out_list


def generate_half_cumulant_contractions(pure_ops_list, ops_list, max_cu):
    """
    Generate all possible combinations of pure creation and annihilation operators for cumulant contractions.
    :param pure_ops_list: a list of pure creation or annihilation indices for each input operator
    :param ops_list: the original list of second-quantized operators
    :param max_cu: the max level of cumulants
    :return: {cumulant level: [[n_cumulant of chosen indices (op index, relative index)], ...]}
    """
    results = {i: [] for i in range(2, max_cu + 1)}

    # generate all possible partitions for k cre/ann legs for k cumulant
    # only unique partitions here, e.g., [2,1,1] included, not [1,2,1] or [1,1,2]
    n_sq_ops = len(ops_list)
    unique_partitions = [part for k in range(1, max_cu + 1)
                         for part in integer_partition(k) if len(part) <= n_sq_ops]

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


def processing_contractions(contractions, elementary_contractions, n_indices, expand_hole,
                            base_order_map, upper_indices_set, lower_indices_set):
    """
    Process a list of contractions expressed in terms of indices of elementary contractions.
    :param contractions: a list of contractions
    :param elementary_contractions: a list of density cumulants / hole densities
    :param n_indices: the total number of indices
    :param expand_hole: expand hole densities to Kronecker delta minus one density if True
    :param base_order_map: the Index map to ordering index
    :param upper_indices_set: the set of all creation operators
    :param lower_indices_set: the set of all annihilation operators
    :return: a list of contractions in terms of (sign, list_of_densities, sq_op)
    """
    out = list()

    for con in contractions:
        list_of_densities = []
        current_order = []

        n_open = 0
        for i in con:
            ele_con = elementary_contractions[i]
            list_of_densities.append(ele_con)
            n_open += ele_con.n_upper

            # creation (left) or annihilation (right)
            left, right = ele_con.upper_indices, ele_con.lower_indices
            if isinstance(ele_con, HoleDensity):
                left, right = right, left
            current_order += left.indices + right.indices[::-1]
        n_open = n_indices - 2 * n_open

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

        out += [(sign * _s, list_of_densities, sq_op) for _s, list_of_densities in sign_densities_pairs]
    return out


def generate_operator_contractions_new(ops_list, max_cu=3, max_n_open=6, min_n_open=0,
                                       expand_hole=True, n_process=1, batch_size=50000):
    """
    Generate operator contractions for a list of SQOperator.
    :param ops_list: a list of SecondQuantizedOperator to be contracted
    :param max_cu: max level of cumulant
    :param max_n_open: max number of open indices for contractions kept for return
    :param min_n_open: min number of open indices for contractions kept for return
    :param expand_hole: expand hole density to Kronecker delta and 1-cumulant if True
    :param n_process: the number of processes launched by multiprocessing
    :param batch_size: the batch size for multiprocessing
    :return: a list of contractions in terms of (sign, list_of_densities, sq_op)
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
    # start = timer()
    elementary_contractions = generate_elementary_contractions(ops_list, max_cu)
    # end = timer()
    n_ele_con = len(elementary_contractions)
    if n_ele_con > 1000:
        sys.setrecursionlimit(n_ele_con)
    # print(f"generate elementary contractions: {end - start:.6f}s, number of elementary contractions: {n_ele_con}")

    # generate incompatible contractions between elementary contractions
    # the list index of elementary_contractions is saved
    # start = timer()
    incompatible_elementary = defaultdict(set)
    for i, ele_i in enumerate(elementary_contractions):
        for j, ele_j in enumerate(elementary_contractions[i + 1:], i + 1):
            if ele_i.any_overlapped_indices(ele_j):
                incompatible_elementary[i].add(j)
                incompatible_elementary[j].add(i)
    # end = timer()
    # print(f"incompatible elementary contractions: {end - start:.6f}s")

    # backtracking (similar to generating sub-lists)
    # start = timer()
    contractions = []
    composite_contractions_backtracking(set(range(n_ele_con)), set(), incompatible_elementary, contractions,
                                        0, elementary_contractions, (n_indices - max_n_open, n_indices - min_n_open))
    # end = timer()
    n_contractions = len(contractions)
    # print(f"backtracking contractions: {end - start:.6f}s")
    # print(f"number of contractions: {n_contractions}")

    # output
    results = list()

    # start = timer()
    if n_process == 1 or n_contractions < batch_size:
        results = processing_contractions(contractions, elementary_contractions, n_indices, expand_hole,
                                          base_order_map, upper_indices_set, lower_indices_set)
    else:
        # manually separate jobs
        n_batches = n_contractions // batch_size + 1
        block_sizes = [n_contractions // n_batches] * n_batches
        for i in range(n_contractions % n_batches):
            block_sizes[i] += 1
        block_ranges = []
        shift = 0
        for size in block_sizes:
            block_ranges.append(range(shift, shift + size))
            shift += size

        n_process = min(n_process, multiprocessing.cpu_count())

        with multiprocessing.Pool(n_process) as pool:
            tasks = list()
            for i in range(n_batches):
                tasks.append((processing_contractions,
                              ([contractions[j] for j in block_ranges[i]],
                               elementary_contractions, n_indices, expand_hole,
                               base_order_map, upper_indices_set, lower_indices_set)))
            for con in pool.imap_unordered(calculatestar, tasks):
                results += con
    # end = timer()
    # print(f"translate contractions: {end - start:.6f}s")

    return results


def composite_contractions_backtracking(available, chosen, incompatible, out, n_con_so_far, translator, desired_n_con):
    """
    Generate composite contractions from elementary contractions.
    :param available: unexplored set of elementary contractions
    :param chosen: chosen set of elementary contractions
    :param incompatible: a map to test incompatible elementary contractions
    :param out: final results from outside
    :param n_con_so_far: the number of contracted indices so far
    :param translator: {elementary contraction index: cumulant/density}
    :param desired_n_con: a tuple of the desired numbers of contracted indices (min, max)
    :return: viable composite contractions
    """
    if len(available) == 0:  # base case, nothing to choose
        desired_min, desired_max = desired_n_con
        if len(chosen) != 0 and (desired_min <= n_con_so_far <= desired_max):
            out.append(deepcopy(chosen))
    else:
        # two choices to explore: with or without the given element
        temp = available.pop()  # choose
        temp_size = 2 * translator[temp].n_upper

        chosen.add(temp)  # choose to include this element
        n_con_so_far += temp_size
        composite_contractions_backtracking(available - incompatible[temp], chosen, incompatible, out,
                                            n_con_so_far, translator, desired_n_con)

        chosen.remove(temp)  # choose not to include this element
        n_con_so_far -= temp_size
        composite_contractions_backtracking(available, chosen, incompatible, out,
                                            n_con_so_far, translator, desired_n_con)

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

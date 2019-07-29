"""
This file implements contractions for second-quantized operators.

Examples
--------
Consider the list of "normal-ordered" operators ops_list = [sq0, sq1, sq2].
Each sqx (x = 0, 1, 2) operator is a SecondQuantizedOperator object that has creation and annihilation attributes.
To distinguish these two types of operators, we refer "sqx" as a macro operator
and creation/annihilation as micro or cre/ann operators.

To be more specific, we use
    sq0 = SecondQuantizedOperator([i, j], [a, b])
    sq1 = SecondQuantizedOperator([p, q], [r, s])
    sq2 = SecondQuantizedOperator([c], [k])
Note that these are not actual object initiations.
For sq0, the cre/ann string is given by "i^+ j^+ b a" (note the annihilation ordering).

From Wick's theorem based on Mukherjee and Kutzelnigg, we consider all types of contractions:
sq0 sq1 sq2 = {i^+ j^+ p^+ q^+ c^+ k s r b a}
              + C([i], [r]) {j^+ b a p^+ q^+ s c^+ k} + ...           [single contractions]
              + C([i], [r]) C([j], [s]) {b a p^+ q^+ c^+ k} + ... -
              + H([p], [a]) H([q], [b]) {i^+ j^+ s r c^+ k} + ...  |- [double contractions, pairwise]
              + C([i], [k]) H([p], [a]) {j^+ b q^+ s r c^+} + ... -
              + C([i, j], [r, s]) {b a p^+ q^+ c^+ k} + ... -
              + C([p, q], [a, b]) {i^+ j^+ s r c^+ k} + ...  |-       [double contractions, cumulant]
              + C([i, c], [r, s]) {j^+ b a p^+ q^+ k} + ... -
              + ...                                                   [triple, ..., contractions]
Here C and H refer to Cumulant and HoleDensity, respectively.

However, some of the contractions are not connected,
for example, the first and second lines of pairwise double contractions.
If we "only" want to compute commutators, e.g., [A, B] = AB - BA,
then we can ignore all disconnected contractions and all pure cumulant-type contractions.
To achieve this, we first categorize contractions according to the macro operators.
For example, C([i], [r]) is categorized as (0, 1), i.e., creation from sq0 and annihilation from sq1.

The algorithm here to compute contractions is based on backtrack elementary contractions.
An elementary contraction is either a Cumulant or HoleDensity.
Then we combine different elementary contractions to obtain composite contractions.
"""

import multiprocessing
import sys
from bisect import bisect_right
from collections import defaultdict, Counter
from itertools import combinations, product, chain, accumulate
from sympy.combinatorics import Permutation
from sympy.utilities.iterables import multiset_permutations

from src.helper.integer_partition import integer_partition
from src.helper.multiprocess_helper import calculate_star
from src.mo_space import space_relation
from src.Indices import IndicesSpinOrbital
from src.SQOperator import SecondQuantizedOperator
from src.Tensor import Tensor, HoleDensity, Cumulant
# from timeit import default_timer as timer


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
    :return: a dictionary of {(macro op indices): a list of contractions represented by HoleDensity and Cumulant}
    """
    for op in ops_list:
        if not isinstance(op, SecondQuantizedOperator):
            raise TypeError(f"Invalid type in ops_list, given '{op.__class__.__name__}', "
                            f"required 'SecondQuantizedOperator'.")
        if op.indices_type != IndicesSpinOrbital:
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
    :return: a dictionary of {(macro op indices): a list of contractions, i.e., HoleDensity or Cumulant}
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
                        out[(i, j)].append(Tensor.make_tensor('lambda', [upper], [lower], 'spin-orbital'))

            # 1-hole-density: left ann + right cre
            for lower in left.ann_ops:
                if lower.space == 'c':
                    continue
                for upper in right.cre_ops:
                    if upper.space == 'c':
                        continue
                    if len(space_relation[upper.space] & space_relation[lower.space]) != 0:
                        out[(j, i)].append(Tensor.make_tensor('eta', [upper], [lower], 'spin-orbital'))

    return out


def compute_elementary_contractions_cumulant(ops_list, max_cu):
    """
    Generate all cumulant-type elementary contractions from a list of second-quantized operators.
    :param ops_list: a list of SecondQuantizedOperator objects
    :param max_cu: the max level of cumulants
    :return: a dictionary of {(macro op indices): a list of contractions, i.e., Cumulant}
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
                    key = tuple(i_sq_op_cre + i_sq_op_ann)
                    out[key].append(Tensor.make_tensor('lambda', cre_indices, ann_indices, 'so'))

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


class ElementaryContractionCategorized:
    def __init__(self, ele_con_categorized, category_sequence):
        """
        Categorized elementary contractions and related methods.
        :param ele_con_categorized: {(macro op indices): a list of Cumulant/HoleDensity}
        :param category_sequence: a list of connected op indices
        """
        self._ele_con = ele_con_categorized
        self._categories = category_sequence

        # starting index (when contractions are flatten to a list) for each category
        self._i_start = [0] + list(accumulate([len(ele_con_categorized[c]) for c in category_sequence[:-1]]))
        self._i_start_map = {c: start for c, start in zip(category_sequence, self._i_start)}

    @property
    def ele_con(self):
        return self._ele_con

    @property
    def categories(self):
        return self._categories

    def size(self):
        if len(self.categories) == 0:
            return 0
        return self._i_start[-1] + len(self.ele_con[self.categories[-1]])

    def encode(self, category, shift):
        """
        Encode a contraction according to its category and relative index in that category.
        :param category: a tuple of connected operator indices
        :param shift: relative index in the category
        :return: code (the index when ele_con is flatten to a list)
        """
        return self._i_start_map[category] + shift

    def find_category(self, code):
        """
        Find the category of a code computed from encode function.
        :param code: code generated from encode function
        :return: the category where the code belongs to
        """
        i = bisect_right(self._i_start, code) - 1
        return self.categories[i]

    def category_range(self, category):
        """ Return the code range for the input category. """
        return range(self._i_start_map[category], self._i_start_map[category] + len(self.ele_con[category]))

    def decode(self, code):
        """
        Decode the input code.
        :param code: code generated from encode function
        :return: the corresponding contraction (Cumulant/HoleDensity)
        """
        i = bisect_right(self._i_start, code) - 1
        c = self.categories[i]
        shift = code - self._i_start[i]
        return self.ele_con[c][shift]

    def compatible_elementary_contractions(self, categories=None):
        """
        Compute compatible elementary contractions.
        :param categories: a list of contraction categories
        :return: a dictionary of {contraction index (encoded): a set of indices (encoded) of compatible contractions}
        """
        if categories is None:
            categories = self.categories

        out = defaultdict(set)

        for i_c, c1 in enumerate(categories):
            for i, ele_i in enumerate(self.ele_con[c1]):
                i_encode = self.encode(c1, i)

                for c2 in categories[i_c:]:
                    for j, ele_j in enumerate(self.ele_con[c2]):

                        if not ele_i.any_overlapped_indices(ele_j):
                            j_encode = self.encode(c2, j)
                            out[i_encode].add(j_encode)
                            out[j_encode].add(i_encode)
        return out

    def composite_contractions(self, target, compatible=None):
        """
        Compute the composite contractions between elementary contractions.
        :param target: a Counter of {(macro op indices): count}
        :param compatible: compatible elementary contractions {code1: [codes compatible to code1]}
        :return: an iterator of "coded" composite contractions
        """
        if compatible is None:
            compatible = self.compatible_elementary_contractions(target.keys())

        choices = set(chain(*[self.category_range(k) for k in target.keys()]))

        return self.composite_contractions_backtrack(choices, target, compatible, [])

    def composite_contractions_backtrack(self, choices, target, compatible, chosen):
        """
        Compute the composite contractions.
        :param choices: a set of coded contractions
        :param target: a Counter of {(macro op indices): count}
        :param compatible: compatible elementary contractions
        :param chosen: a list of coded contractions that satisfies the target condition
        """
        available = {i: 0 for i in target.keys()}
        for con in choices:
            available[self.find_category(con)] += 1
        if any(target[c] > available[c] for c in target.keys()):
            return

        if len(choices) == 0:
            yield chosen
        else:
            # explore
            con = choices.pop()
            category = self.find_category(con)

            # include
            chosen.append(con)
            target[category] -= 1

            choices_new = choices.intersection(compatible[con])
            if target[category] == 0:
                choices_new -= set(self.category_range(category))

            yield from self.composite_contractions_backtrack(choices_new, target, compatible, chosen)

            # not include
            chosen.pop()
            target[category] += 1

            yield from self.composite_contractions_backtrack(choices, target, compatible, chosen)

            # un-explore
            choices.add(con)


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
        if len(elementary_contractions) == 0:
            return []

        # important to put pairwise contractions at the end due to contracted_operator_backtrack_macro
        elementary_sequence = sorted(elementary_contractions.keys(), key=lambda x: (len(x), x), reverse=True)
        ele_con = ElementaryContractionCategorized(elementary_contractions, elementary_sequence)
        if ele_con.size() > 1000:
            sys.setrecursionlimit(ele_con.size())

        # compatible contractions
        compatible = ele_con.compatible_elementary_contractions()

        # compute composite contractions
        #   1) compute valid combinations of categories of elementary contractions (backtrack algorithm)
        #   2) select elementary contractions for a given combination of categories (backtrack algorithm)
        #   3) translate elementary contractions to a list of Cumulant/HoleDensity
        #   4) expand HoleDensity to Kronecker delta - 1-body Cumulant
        #   5) determine sign and open (un-contracted) cre/ann operators

        if n_process == 1:
            for comp_cat in contracted_operator_backtrack_macro(ele_con.categories, [], (min_n_con, max_n_con), 0,
                                                                [sq_op.n_cre for sq_op in ops_list],
                                                                [sq_op.n_ann for sq_op in ops_list], set(), True):
                yield process_composite_categorized(comp_cat, ele_con, compatible, upper_indices_set,
                                                    lower_indices_set, base_order_map, n_indices, expand_hole)
        else:
            # save composite categories for parallel computation
            comp_cats_list = [i[:] for i in
                              contracted_operator_backtrack_macro(ele_con.categories, [], (min_n_con, max_n_con), 0,
                                                                  [sq_op.n_cre for sq_op in ops_list],
                                                                  [sq_op.n_ann for sq_op in ops_list], set(), True)]

            n_process = min(n_process, multiprocessing.cpu_count())
            with multiprocessing.Pool(n_process, maxtasksperchild=1000) as pool:
                tasks = []
                for comp_cat in comp_cats_list:
                    tasks.append((process_composite_categorized, (comp_cat, ele_con, compatible, upper_indices_set,
                                                                  lower_indices_set, base_order_map, n_indices,
                                                                  expand_hole)))
                imap_unordered_it = pool.imap_unordered(calculate_star, tasks)
                for results in imap_unordered_it:
                    yield results

    else:
        elementary_contractions = compute_elementary_contractions_list(ops_list, max_cu)
        compatible = compute_compatible_elementary_contractions_list(elementary_contractions)

        n_ele_con = len(elementary_contractions)
        if n_ele_con > 1000:
            sys.setrecursionlimit(n_ele_con)

        if n_process == 1:
            for con in composite_contractions_backtrack(set(range(n_ele_con)), set(), compatible, 0,
                                                        elementary_contractions, (min_n_con, max_n_con)):
                yield process_composite_contractions(con, elementary_contractions, n_indices, expand_hole,
                                                     base_order_map, upper_indices_set, lower_indices_set)
        else:
            composite = [i[:] for i in composite_contractions_backtrack(set(range(len(elementary_contractions))),
                                                                        set(), compatible, 0, elementary_contractions,
                                                                        (min_n_con, max_n_con))]
            n_process = min(n_process, multiprocessing.cpu_count())
            with multiprocessing.Pool(n_process, maxtasksperchild=1000) as pool:
                tasks = []
                for con in composite:
                    tasks.append((process_composite_contractions, (con, elementary_contractions, n_indices, expand_hole,
                                                                   base_order_map, upper_indices_set, lower_indices_set)
                                  ))
                imap_unordered_it = pool.imap_unordered(calculate_star, tasks)
                for results in imap_unordered_it:
                    yield results


def contracted_operator_backtrack_macro(available, chosen, n_con, n_con_so_far,
                                        cre_available, ann_available, ops_so_far, all_cu_so_far):
    """
    Generate all possible connected macro operator contractions
    :param available: a list of unexplored elementary contracted macro operators (tuple of operator indices)
    :param chosen: a list of chosen contracted macro operators
    :param n_con: the (min, max) numbers of contractions specified by the user
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
        if len(ops_so_far) == n_ops and n_con[0] <= n_con_so_far <= n_con[1]:
            yield chosen
    else:
        # explore when 1) number of contractions are not enough,
        #              2) previous contractions are not all cumulants
        if n_con_so_far < n_con[1] and (len(available[-1]) == 2 if n_con_so_far == 0 else not all_cu_so_far):

            # choose an element
            temp = available[-1]

            temp_ops = set(temp)
            n_body = len(temp) // 2
            cre_count, ann_count = Counter(temp[:n_body]), Counter(temp[n_body:])

            # include this element when: 1) this element must connect to the previous ones,
            #                            2) both cre and ann won't exceed the limit
            if ((not ops_so_far.isdisjoint(temp_ops)) or n_con_so_far == 0) and \
                    all(cre_available[k] - v >= 0 for k, v in cre_count.items()) and \
                    all(ann_available[k] - v >= 0 for k, v in ann_count.items()):

                chosen.append(temp)  # include this element

                available_new = _prune_available_contracted_operator(available, cre_available, ann_available,
                                                                     cre_count, ann_count)

                yield from contracted_operator_backtrack_macro(available_new[0], chosen,
                                                               n_con, n_con_so_far + 2 * n_body,
                                                               available_new[1], available_new[2],
                                                               ops_so_far | temp_ops, all_cu_so_far and n_body > 1)

                chosen.pop()  # not include this element

            # not include this element
            yield from contracted_operator_backtrack_macro(available[:-1], chosen, n_con, n_con_so_far,
                                                           cre_available, ann_available, ops_so_far, all_cu_so_far)
            # un-choose this element: nothing to do here
        else:
            yield from contracted_operator_backtrack_macro([], chosen, n_con, n_con_so_far, cre_available,
                                                           ann_available, ops_so_far, all_cu_so_far)


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
        available_pruned = []
        for con_ops in available:
            if any(i_cre in cre_zero_count_id for i_cre in con_ops[:len(con_ops) // 2]):
                continue
            if any(i_ann in ann_zero_count_id for i_ann in con_ops[len(con_ops) // 2:]):
                continue
            available_pruned.append(con_ops)

    return available_pruned, cre_available_new, ann_available_new


def process_composite_categorized(com_cat, ele_con, compatible, upper_indices_set, lower_indices_set,
                                  base_order_map, n_indices, expand_hole):
    """
    Process one composite categorized contraction.
    :param com_cat: a list of connected operator indices
    :param ele_con: an ElementaryContractionCategorized object
    :param compatible: compatible elementary contractions
    :param upper_indices_set: the set of all creation operators
    :param lower_indices_set: the set of all annihilation operators
    :param base_order_map: the Index map to ordering index
    :param n_indices: the total number of cre and ann operators
    :param expand_hole: expand hole densities to Kronecker delta minus one density if True
    :return: a list of contractions in terms of (sign, list_of_densities, sq_op)
    """
    n_open = n_indices - sum(len(i) for i in com_cat)
    out = []

    for coded_cons in ele_con.composite_contractions(Counter(com_cat), compatible):
        contractions = [ele_con.decode(i) for i in coded_cons]

        # cre/ann ordering of the current composite contraction
        current_order = []
        for con in contractions:
            left = con.lower_indices if isinstance(con, HoleDensity) else con.upper_indices
            right = con.upper_indices if isinstance(con, HoleDensity) else con.lower_indices
            current_order += left.indices + right.indices[::-1]

        # sort open indices
        if n_open != 0:
            contracted = set(current_order)
            open_upper = IndicesSpinOrbital(sorted(upper_indices_set - contracted))
            open_lower = IndicesSpinOrbital(sorted(lower_indices_set - contracted))
            current_order += open_upper.indices + open_lower.indices[::-1]
        else:
            open_upper, open_lower = IndicesSpinOrbital([]), IndicesSpinOrbital([])
        sq_op = SecondQuantizedOperator(open_upper, open_lower)

        # determine sign of current ordering
        sign = (-1) ** Permutation([base_order_map[i] for i in current_order]).inversions()

        # expand hole density to delta - 1-cumulant
        sign_densities = expand_hole_densities(contractions) if expand_hole else [(1, contractions)]

        # append results
        out += [(sign * _s, cons, sq_op) for _s, cons in sign_densities]

    return out


def compute_compatible_elementary_contractions_list(elementary_contractions):
    """
    Compute incompatible elementary contractions.
    :param elementary_contractions: elementary contractions generated by compute_elementary_contractions_list
    :return: a dictionary of {contraction index: a set of indices of incompatible contractions}
    """
    compatible_elementary = defaultdict(set)
    for i, ele_i in enumerate(elementary_contractions):
        for j, ele_j in enumerate(elementary_contractions[i + 1:], i + 1):
            if not ele_i.any_overlapped_indices(ele_j):
                compatible_elementary[i].add(j)
                compatible_elementary[j].add(i)
    return compatible_elementary


def composite_contractions_backtrack(available, chosen, compatible, n_con_so_far, translator, desired_n_con):
    """
    Generate composite contractions from elementary contractions.
    :param available: a unexplored set of elementary contractions
    :param chosen: chosen set of elementary contractions
    :param compatible: a map to test incompatible elementary contractions
    :param n_con_so_far: the number of contracted indices so far
    :param translator: {elementary contraction index: cumulant/density}
    :param desired_n_con: a tuple of the desired numbers of contracted indices (min, max)
    :return: viable composite contractions
    """
    if len(available) == 0:  # base case, nothing to choose
        if len(chosen) != 0 and (desired_n_con[0] <= n_con_so_far <= desired_n_con[1]):
            yield chosen
    else:
        # two choices to explore: with or without the given element
        temp = available.pop()  # choose
        temp_size = 2 * translator[temp].n_upper

        chosen.add(temp)  # include this element
        n_con_so_far += temp_size
        yield from composite_contractions_backtrack(available & compatible[temp], chosen, compatible,
                                                    n_con_so_far, translator, desired_n_con)

        chosen.remove(temp)  # not to include this element
        n_con_so_far -= temp_size
        yield from composite_contractions_backtrack(available, chosen, compatible, n_con_so_far, translator,
                                                    desired_n_con)

        available.add(temp)  # un-choose


def process_composite_contractions(contraction, elementary_contractions, n_indices, expand_hole,
                                   base_order_map, upper_indices_set, lower_indices_set):
    """
    Process a single composite contraction expressed in terms of indices of elementary contractions.
    :param contraction: a composite contraction
    :param elementary_contractions: a list of density cumulants / hole densities
    :param n_indices: the total number of indices
    :param expand_hole: expand hole densities to Kronecker delta minus one density if True
    :param base_order_map: the Index map to ordering index
    :param upper_indices_set: the set of all creation operators
    :param lower_indices_set: the set of all annihilation operators
    :return: a list of contractions in terms of (sign, list_of_densities, sq_op)
    """
    list_of_densities = []
    current_order = []

    n_open = 0
    for i in contraction:
        ele_con = elementary_contractions[i]
        list_of_densities.append(ele_con)
        n_open += ele_con.n_upper

        # creation (left) or annihilation (right)
        left, right = ele_con.upper_indices, ele_con.lower_indices
        if isinstance(ele_con, HoleDensity):
            left, right = right, left
        current_order += left.indices + right.indices[::-1]
    n_open = n_indices - 2 * n_open

    # sort the open indices
    if n_open != 0:
        contracted = set(current_order)
        open_upper_indices = IndicesSpinOrbital(sorted(upper_indices_set - contracted))
        open_lower_indices = IndicesSpinOrbital(sorted(lower_indices_set - contracted))
        current_order += open_upper_indices.indices + open_lower_indices.indices[::-1]
    else:
        open_upper_indices, open_lower_indices = IndicesSpinOrbital([]), IndicesSpinOrbital([])
    sq_op = SecondQuantizedOperator(open_upper_indices, open_lower_indices)

    # expand hole densities to delta - lambda_1
    sign_densities_pairs = expand_hole_densities(list_of_densities) if expand_hole else [(1, list_of_densities)]

    # determine sign
    sign = (-1) ** Permutation([base_order_map[i] for i in current_order]).inversions()

    return [(sign * _s, list_of_densities, sq_op) for _s, list_of_densities in sign_densities_pairs]


def expand_hole_densities(list_of_tensors):
    """
    Expand all the hole densities in the list_of_tensors.
    :param list_of_tensors: a list of Tensor objects
    :return: generate tuples of (sign, expanded Tensor objects)
    """
    good_tensors, hole_densities_expanded = [], []
    for tensor in list_of_tensors:
        if isinstance(tensor, HoleDensity):
            hole_densities_expanded.append(tensor.expand())
        else:
            good_tensors.append(tensor)

    if len(hole_densities_expanded) == 0:
        yield 1, list_of_tensors
    else:
        for expanded in product(*hole_densities_expanded):
            sign = (-1) ** sum(isinstance(tensor, Cumulant) for tensor in expanded)
            yield sign, good_tensors + list(expanded)

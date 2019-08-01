from copy import deepcopy
from collections import defaultdict
from fractions import Fraction
from itertools import product, groupby, accumulate
from math import factorial
from sympy.combinatorics.tensor_can import canonicalize
from sympy.combinatorics import Permutation

from typing import List
from src.mo_space import space_priority, space_relation, space_priority_so, space_relation_so, mo_space
from src.Index import Index
from src.Indices import Indices, IndicesSpinOrbital, IndicesAntisymmetric
from src.Tensor import Tensor, Cumulant, HamiltonianTensor, ClusterAmplitude, Kronecker, HoleDensity
from src.SQOperator import SecondQuantizedOperator
from SpaceCounter import SpaceCounter


class Term:
    def __init__(self, list_of_tensors, sq_op, coeff=1.0, need_to_sort=True):
        """
        The Term class to store a list of tensors, a coefficient, and a SecondQuantizedOperator.
        :param list_of_tensors: a list of Tensor objects
        :param sq_op: a SecondQuantizedOperator object
        :param coeff: the coefficient of the term
        :param need_to_sort: sort the list of tensors if True
        """
        if not isinstance(coeff, float):
            try:
                coeff = float(coeff)
            except ValueError:
                raise ValueError(f"Invalid Term::coeff, given {coeff} ('{coeff.__class__.__name__}'),"
                                 f" required 'float'.")
        self._coeff = coeff

        if not isinstance(sq_op, SecondQuantizedOperator):
            raise TypeError(f"Invalid Term::sq_op, given '{sq_op.__class__.__name__}',"
                            f" required 'SecondQuantizedOperator'.")
        self._sq_op = sq_op

        # test if this term is connected, need to separate diagonal indices
        connection = defaultdict(int)
        for i in sq_op.indices:
            connection[i] += 1
        diagonal_indices = sq_op.diagonal_indices()

        for tensor in list_of_tensors:
            if not isinstance(tensor, Tensor):
                raise TypeError(f"Invalid element in Term::list_of_tensors.\n"
                                f"{tensor} ({tensor.__class__.__name__}) is not of 'Tensor' or its derived type.")

            if tensor.indices_type is not sq_op.indices_type:
                raise TypeError(f"Invalid element in Term::list_of_tensors.\n"
                                f"Inconsistent indices types: "
                                f"{tensor} ({tensor.indices_type}), required '{sq_op.indices_type}'")

            for i in tensor.indices:
                connection[i] += 1
            diagonal_indices.update(tensor.diagonal_indices())

        if any(v != 2 for k, v in connection.items() if k not in diagonal_indices):
            raise ValueError(f"Invalid Term because it is not connected.\n"
                             f"tensors: {list_of_tensors}\n"
                             f"operator: {sq_op}\n"
                             f"indices count: {connection}")

        if any(connection[k] % 2 == 1 for k in diagonal_indices):
            raise ValueError(f"Invalid Term because diagonal indices do not appear in pairs.\n"
                             f"tensors: {list_of_tensors}\n"
                             f"operator: {sq_op}\n"
                             f"indices count: {connection}\n"
                             f"diagonal indices: {diagonal_indices}")

        self._list_of_tensors = sorted(list_of_tensors) if need_to_sort else list_of_tensors
        self._indices_set = set(connection.keys())
        self._sorted = need_to_sort

        # determine the next available index for each space
        next_index_number = {i: 0 for i in space_priority}
        for i in self._indices_set:
            next_index_number[i.space] = max(next_index_number[i.space], i.number + 1)
        self._next_index_number = next_index_number

    @classmethod
    def from_term(cls, term, flip_sign=False):
        """ Copy from a term. """
        sign = -1 if flip_sign else 1
        return cls(term.list_of_tensors, term.sq_op, sign * term.coeff)

    @classmethod
    def make_empty(cls, indices_type='so'):
        """ Create an empty Term. """
        return cls([], SecondQuantizedOperator.make_empty(indices_type), 0.0)

    @property
    def coeff(self):
        return self._coeff

    @coeff.setter
    def coeff(self, value):
        try:
            v = float(value)
        except ValueError:
            raise ValueError(f"Invalid Term::coeff, given {value} ('{value.__class__.__name__}'),"
                             f" required 'float'.")
        self._coeff = v

    @property
    def sq_op(self):
        return self._sq_op

    @property
    def list_of_tensors(self):
        return self._list_of_tensors

    @property
    def n_tensors(self):
        return len(self._list_of_tensors)

    @property
    def indices_set(self):
        return self._indices_set

    @property
    def sorted(self):
        return self._sorted

    @sorted.setter
    def sorted(self, value):
        if not isinstance(value, (int, float)):
            raise TypeError(f"Invalid value type, given '{value.__class__.__name__}', required 'Boolean'.")
        self._sorted = value

    @property
    def next_index_number(self):
        return self._next_index_number

    @property
    def comparison_tuple(self):
        return self.sq_op, self.n_tensors, self.list_of_tensors, abs(self.coeff), self.coeff

    @staticmethod
    def _is_valid_operand(other):
        if not isinstance(other, Term):
            raise TypeError(f"Cannot compare between 'Term' and '{other.__class__.__name__}'.")

    def __eq__(self, other):
        self._is_valid_operand(other)
        return (self.coeff, self.sq_op, self.list_of_tensors) == (other.coeff, other.sq_op, other.list_of_tensors)

    def almost_equal(self, other):
        self._is_valid_operand(other)
        return (self.sq_op, self.list_of_tensors) == (other.sq_op, other.list_of_tensors)

    def __ne__(self, other):
        self._is_valid_operand(other)
        return (self.coeff, self.sq_op, self.list_of_tensors) != (other.coeff, other.sq_op, other.list_of_tensors)

    def __lt__(self, other):
        self._is_valid_operand(other)
        return self.comparison_tuple < other.comparison_tuple

    def __le__(self, other):
        self._is_valid_operand(other)
        return self.comparison_tuple <= other.comparison_tuple

    def __gt__(self, other):
        self._is_valid_operand(other)
        return self.comparison_tuple > other.comparison_tuple

    def __ge__(self, other):
        self._is_valid_operand(other)
        return self.comparison_tuple >= other.comparison_tuple

    def __repr__(self):
        return self.latex(permute_format=False)

    def is_excitation(self):
        """
        Test if this term is a possible excitation operator.
        :return: True if this term is a possible excitation operator, otherwise False
        """
        return self.sq_op.is_excitation()

    def void(self):
        """ Return an empty Term object. """
        return Term([], self.sq_op.void(), 0.0)

    def is_void(self):
        """ Return True if this Term is zero. """
        return self.coeff == 0

    @staticmethod
    def format_coeff(value, form=None):
        fraction = Fraction(value).limit_denominator(1000000)

        if form == 'ambit':
            if '/' not in str(fraction):
                out = f"{fraction.numerator}.0"
            else:
                out = f"({fraction.numerator}.0 / {fraction.denominator}.0)"
        else:
            out = f"{fraction}"

        return out

    def latex(self, dollar=False, permute_format=True, delimiter=False, backslash=False):
        """
        Translate to latex form.
        :param dollar: use math mode if True
        :param permute_format: use permute format (when sq_op contains different MO spaces) if True
        :param delimiter: use & delimiter for latex align environment if True
        :param backslash: add \\ at the end for latex align environment if True
        :return: latex form (a string) of the Term
        """
        tensors_str = " " + " ".join((tensor.latex() for tensor in self.list_of_tensors))

        n_perm, perm_str, sq_op_str = 1, "", self.sq_op.latex()
        if permute_format:
            p_cre, p_ann = self.perm_partition_open()
            n_perm, perm_str, sq_op_str = self.sq_op.latex_permute_format(p_cre, p_ann)

        coeff_str = self.format_coeff(self.coeff / n_perm)

        if delimiter:
            coeff_str += " &"
        if perm_str:
            perm_str = " " + perm_str
        if sq_op_str:
            sq_op_str = " " + sq_op_str
        if backslash:
            sq_op_str += " \\\\"

        out = coeff_str + perm_str + tensors_str + sq_op_str
        if dollar:
            out = "$" + out + "$"
        return out

    def ambit(self, name='C', ignore_permutations=False, init_temp=True, declared_temp=True):
        """
        Translate to ambit form, forced to add permutations if found.
        :param name: output tensor name
        :param ignore_permutations: ignore permutations printing
        :param init_temp: initialize a temp tensor
        :param declared_temp: declared a temp tensor previously
        :return: ambit form (a string) of the Term
        """
        if not isinstance(name, str):
            raise TypeError(f"Invalid ambit name, given '{name.__class__.__name__}', required 'str'.")

        sq_op = self.sq_op

        factor = factorial(sq_op.n_cre) * factorial(sq_op.n_ann)
        p_cre, p_ann = self.perm_partition_open()
        n_perm = sq_op.n_multiset_permutation(p_cre, p_ann)
        coeff_str = self.format_coeff(self.coeff * factor / n_perm, 'ambit')

        tensors_str = " * ".join((tensor.ambit() for tensor in self.list_of_tensors))
        real_name = f"{name}{sq_op.n_ann}" if sq_op.is_particle_conserving() else f"{name}_{sq_op.n_ann}_{sq_op.n_cre}"

        if not sq_op.exist_permute_format(p_cre, p_ann):
            lhs = f"{real_name}{sq_op.ambit(cre_first=False)}"
            rhs = f"{coeff_str} * {tensors_str}"
            return lhs + " += " + rhs + ";"
        else:
            if not any([ignore_permutations, init_temp, declared_temp]):
                return f'{real_name}{sq_op.ambit(cre_first=False)} += {coeff_str} * {tensors_str};\n'

            out = ''
            if init_temp:
                space_str = "".join([i.space for i in sq_op.ann_ops] + [i.space for i in sq_op.cre_ops])
                out = f'temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {{"{space_str}"}});\n'
                declared_temp = True

            t_name = 'temp' if declared_temp else real_name

            if ignore_permutations:
                return out + f'{t_name}{sq_op.ambit(cre_first=False)} += {coeff_str} * {tensors_str};'
            else:
                if declared_temp:
                    temp_indices = f'{sq_op.ambit(cre_first=False)}'
                    out += f'temp{temp_indices} += {coeff_str} * {tensors_str};\n'
                    for sign, lhs_indices in sq_op.ambit_permute_format(p_cre, p_ann, cre_first=False):
                        out += f"{real_name}{lhs_indices} {'+' if sign == 1 else '-'}= temp{temp_indices};\n"
                else:
                    for sign, lhs_indices in sq_op.ambit_permute_format(p_cre, p_ann, cre_first=False):
                        out += f"{real_name}{lhs_indices} {'+' if sign == 1 else '-'}= {coeff_str} * {tensors_str};"

            return out

    def perm_partition_open(self):
        """
        Test if there are any permutation format of this term.
        :return: permutation partitions for upper and lower open indices
        """
        cre_ops = self.sq_op.cre_ops
        ann_ops = self.sq_op.ann_ops
        tensor_indices = [set(tensor.indices) for tensor in self.list_of_tensors]

        return self._perm_part_atomic(cre_ops, tensor_indices), self._perm_part_atomic(ann_ops, tensor_indices)

    @staticmethod
    def _perm_part_atomic(ops, tensor_indices):
        """
        Test if a list of indices has permutation partitions.
        If so, there is a permutation format for the SecondQuantizedOperator.
        :param ops: an Indices object for upper/lower open indices
        :param tensor_indices: a list of sets of tensors indices
        :return: a permutation partition of the indices

        Examples
        --------
        Consider ops = [i, j, k, l], tensor_indices = [[p, q, r, s], [i, j, a, b], [k, c], [l, p, q, m, b, c]].
        Note that indices 'i' and 'j' belong to the second tensor, 'k' to the third, and 'l' to the last.
        The permutation partition is thus P(ij/k/l) = [[i, j], [k], [l]].

        Consider ops = [p, i, c], tensor_indices = [[p, i, a, b], [b, c, j, k]].
        Indices 'p' and 'i' belong to the first tensor and 'c' belongs to the second tensor.
        However, since 'p' and 'i' belong to different MO space, we need to separate these as well.
        The permutation partition is thus P(p/c/i) = [[p], [c], [i]].
        Note that here will generate repetitions if term is not canonicalized.
        """
        if ops.size == 0:
            return [[]]
        elif ops.size == 1:
            return [[ops[0]]]
        else:
            # test if indices are in different operators
            processed = []
            for indices_set in tensor_indices:
                open_indices = indices_set & ops.indices_set
                if len(open_indices) != 0:
                    processed.append(sorted(open_indices))

            out = []
            for indices in processed:
                cat = defaultdict(list)
                for i in indices:
                    cat[i.space].append(i)
                for space in sorted(cat.keys(), key=lambda x: space_priority[x]):
                    out.append(cat[space])

            return out

    def void_self(self):
        """
        Make current Term an empty Term.
        """
        self._coeff = 0.0
        self._sq_op = self.sq_op.void_sq_op()
        self._indices_set = set()
        self._list_of_tensors = []

    @staticmethod
    def _generate_next_index(space, next_index, add=True):
        out = Index(f"{space}{next_index[space]}")
        if add:
            next_index[space] += 1
        return out

    @staticmethod
    def _replace_tensors_indices(input_tensors, replacement):
        """
        Replace indices of list_of_indices according to replacement map.
        :param input_tensors: a list of tensors, should not be empty
        :param replacement: a replacement map for indices {old index: new index}
        :return: a list of sorted tensors with replaced indices
        """
        if len(input_tensors) == 0:
            raise ValueError("list_of_tensors cannot be empty!")

        list_of_tensors = []
        for tensor in input_tensors:
            upper_indices = tensor.indices_type([replacement[i] for i in tensor.upper_indices])
            lower_indices = tensor.indices_type([replacement[i] for i in tensor.lower_indices])
            list_of_tensors.append(tensor.from_indices(upper_indices, lower_indices))
        return list_of_tensors

    @staticmethod
    def _replace_sqop_indices(input_sqop, replacement):
        """
        Replace indices of second quantized operator according to replacement map.
        :param input_sqop: a SecondQuantizedOperator object
        :param replacement: a replacement map for indices
        :return: a SecondQuantizedOperator object with replaced indices
        """
        upper_indices = input_sqop.indices_type([replacement[i] for i in input_sqop.cre_ops])
        lower_indices = input_sqop.indices_type([replacement[i] for i in input_sqop.ann_ops])
        return SecondQuantizedOperator(upper_indices, lower_indices)

    def _remove_kronecker_delta(self):
        """
        Remove Kronecker delta of this term in place.
        :return: a tuple of (non-delta tensors, a replacement map {index: downgraded index})
        """
        list_of_tensors = []

        # replacement = {i: i for i in self.indices_set}
        replacement = {}
        next_active_a, next_active_b = self.next_index_number['a'], self.next_index_number['A']

        for tensor in self.list_of_tensors:
            if isinstance(tensor, Kronecker):
                if tensor.downgrade_indices() == '':
                    self.void_self()
                    return
                else:
                    high, low = sorted([tensor.upper_indices[0], tensor.lower_indices[0]])
                    if (high.space, low.space) == ('p', 'h'):
                        index_name = f"a{next_active_a}"
                        replacement[high], replacement[low] = Index(index_name), Index(index_name)
                        next_active_a += 1
                    elif (high.space, low.space) == ('P', 'H'):
                        index_name = f"A{next_active_b}"
                        replacement[high], replacement[low] = Index(index_name), Index(index_name)
                        next_active_b += 1
                    else:
                        replacement[high] = low
            else:
                list_of_tensors.append(tensor)

        # # relabel tensors using replacement map
        # self._list_of_tensors = sorted(self._replace_tensors_indices(list_of_tensors, replacement))
        # self._sorted = True
        # self._indices_set = set(replacement.values())
        # self._n_tensors = len(self._list_of_tensors)

        # sanity changes to next available index for active labels
        self._next_index_number['a'] = next_active_a
        self._next_index_number['A'] = next_active_b

        return list_of_tensors, replacement

    def _downgrade_cumulant_indices(self, simplify_core_cumulant=True):
        """
        Downgrade cumulant indices: 1cu -> hole only, 2cu -> active only.
        :param simplify_core_cumulant: change a cumulant labeled by core indices to a Kronecker delta
        :return: a replacement map {index: downgraded index}
        """
        # replacement = {i: i for i in self.indices_set}
        replacement = {}
        next_index = {**self.next_index_number}

        for i_tensor, tensor in enumerate(self.list_of_tensors):
            if isinstance(tensor, Cumulant):
                # higher-order cumulant can only have active indices
                if tensor.n_body != 1:
                    for i in tensor.upper_indices:
                        replacement[i] = self._generate_next_index('A' if i.is_beta() else 'a', next_index)
                    for i in tensor.lower_indices:
                        replacement[i] = self._generate_next_index('A' if i.is_beta() else 'a', next_index)
                else:
                    u_index, l_index = tensor.upper_indices[0], tensor.lower_indices[0]
                    space_label = tensor.downgrade_indices()

                    if space_label in ('c', 'C'):
                        if simplify_core_cumulant:
                            self._list_of_tensors[i_tensor] = Kronecker(tensor.upper_indices, tensor.lower_indices)
                        else:
                            replacement[u_index] = self._generate_next_index(space_label, next_index)
                            replacement[l_index] = self._generate_next_index(space_label, next_index)
                    elif space_label == '':
                        self.void_self()
                        return
                    else:
                        replacement[u_index] = self._generate_next_index(space_label, next_index)
                        replacement[l_index] = self._generate_next_index(space_label, next_index)

        # set next available index number to the modified one
        self._next_index_number.update(next_index)

        return replacement

    def _remove_active_only_amplitudes(self):
        """
        Void the term if contains any amplitudes labeled by active indices.
        """
        for tensor in self.list_of_tensors:
            if isinstance(tensor, ClusterAmplitude):
                if tensor.is_all_active():
                    self.void_self()
                    return

    def simplify(self, simplify_core_cumulant=True, remove_active_amplitudes=True):
        """
        Simplify the term in place by downgrading cumulant indices and removing Kronecker deltas.
        :param simplify_core_cumulant: change a cumulant labeled by core indices to a Kronecker delta
        :param remove_active_amplitudes: remove terms when its contains all-active amplitudes
        """
        replacement = {i: i for i in self.indices_set}

        # downgrade cumulant indices
        replacement_cumulant = self._downgrade_cumulant_indices(simplify_core_cumulant)
        replacement.update(replacement_cumulant)
        if abs(self.coeff) < 1.0e-12:
            return

        # remove Kronecker deltas
        list_of_tensors, replacement_delta = self._remove_kronecker_delta()
        replacement.update(replacement_delta)
        if abs(self.coeff) < 1.0e-12:
            return

        # remove all-active amplitudes
        final_tensors = []
        if remove_active_amplitudes:
            for tensor in list_of_tensors:
                if isinstance(tensor, ClusterAmplitude):
                    space = set([replacement[i].space for i in tensor.upper_indices]).union(
                        set([replacement[i].space for i in tensor.lower_indices]))
                    if len(space) == 1 and (next(iter(space)) in ('a', 'A')):
                        self.void_self()
                        return
                final_tensors.append(tensor)
        else:
            final_tensors = list_of_tensors

        # relabel tensors using replacement map
        self._list_of_tensors = sorted(self._replace_tensors_indices(final_tensors, replacement))
        self._sorted = True
        self._indices_set = set(replacement.values())

    def build_adjacency_matrix(self, ignore_cumulant=True):
        """
        Build the adjacency matrix (upper triangle) of the current Term.
        For example, Term = 1.0 * H * T1 * T2 * L1 * L2 * L3 * SqOp
                   H         T1          T2          L1          L2          L3          SqOp
        -------------------------------------------------------------------------------------
        H       None  SC(H, T1)   SC(H, T2)   SC(H, L1)   SC(H, L2)   SC(H, L3)   SC(H, SqOp)
        T1      None       None  SC(T1, T2)  SC(T1, L1)  SC(T1, L2)  SC(T1, L3)  SC(T1, SqOp)
        T2      None       None        None  SC(T2, L1)  SC(T2, L2)  SC(T2, L3)  SC(T2, SqOp)
        L1      None       None        None        None  SC(L1, L2)  SC(L1, L3)          None
        L2      None       None        None        None        None  SC(L2, L3)          None
        L3      None       None        None        None        None        None          None
        SqOp    None       None        None        None        None        None          None

        :param ignore_cumulant: ignore building elements between cumulants, hole densities, or Kronecker deltas
        :return: the adjacency matrix specified by SpaceCounter
        """
        adjacency_matrix: List[List[SpaceCounter]] = [[None for _ in range(self.n_tensors + 1)]
                                                      for __ in range(self.n_tensors + 1)]

        for i1, tensor1 in enumerate(self.list_of_tensors):
            is_cumulant1 = True if isinstance(tensor1, (Cumulant, HoleDensity, Kronecker)) else False
            for i2, tensor2 in enumerate(self.list_of_tensors[i1 + 1:] + [self.sq_op], i1 + 1):
                if is_cumulant1:
                    if ignore_cumulant and isinstance(tensor2, (Cumulant, HoleDensity, Kronecker)):
                        continue
                    if isinstance(tensor2, SecondQuantizedOperator):
                        continue
                adjacency_matrix[i1][i2] = SpaceCounter(tensor1, tensor2)

        return adjacency_matrix

    def similar(self, other):
        self._is_valid_operand(other)
        return self.build_adjacency_matrix() == other.build_adjacency_matrix()

    def almost_similar(self, other):
        self._is_valid_operand(other)
        return self.order_tensors() == other.order_tensors()

    def order_tensors(self, simplified=True):
        """
        Order tensors in place (after simplified) according to adjacency matrix.
        :param simplified: True if this Term is simplified already
        :return: the adjacency matrix of the ordered list of tensors
        """
        if not simplified:
            self.simplify()

        n_tensors = self.n_tensors
        ordered_tensors: List[Tensor] = [None for _ in range(n_tensors)]

        # build adjacency matrix of the current term
        adj_mat = self.build_adjacency_matrix()
        adj_mat_copy = deepcopy(adj_mat)

        # figure out possible equivalent tensors with same name, n_upper, n_lower
        equivalent_tensors = defaultdict(list)
        n_non_cu = n_tensors
        for i, tensor in enumerate(self.list_of_tensors):
            equivalent_tensors[f"{tensor.name}_{tensor.n_upper}_{tensor.n_lower}"].append(i)
            if n_non_cu == n_tensors and isinstance(tensor, Cumulant):
                n_non_cu = i

        # make sure we fix ordering of amplitudes first, then cumulants
        for i_tensors in sorted(equivalent_tensors.values(), key=lambda i: (self.list_of_tensors[i[0]].priority,
                                                                            self.list_of_tensors[i[0]].n_upper,
                                                                            self.list_of_tensors[i[0]].n_lower)):
            if len(i_tensors) == 1:
                ordered_tensors[i_tensors[0]] = self.list_of_tensors[i_tensors[0]]
            else:
                if 0 in i_tensors:
                    raise NotImplementedError("Not considered this situation. Must have only one Hamiltonian.")
                    # max_i = max(i_tensors)
                    # ordered = sorted(i_tensors, key=lambda i: [adj_mat[i][n_tensors]] + sorted(adj_mat[i][max_i:-1]))
                    # for i, j in zip(i_tensor, ordered):
                    #     ordered_tensors[i] = self.list_of_tensors[j]
                    #     adj_mat[i] = adj_mat_copy[j]
                    # adj_mat_copy = deepcopy(adj_mat)

                if isinstance(self.list_of_tensors[i_tensors[0]], ClusterAmplitude):
                    ordered = sorted(i_tensors, key=lambda i: [adj_mat[i][n_tensors],
                                                               adj_mat[0][i]] + sorted(adj_mat[i][n_non_cu:]))
                    for i, j in zip(i_tensors, ordered):
                        ordered_tensors[i] = self.list_of_tensors[j]
                        adj_mat[i] = adj_mat_copy[j]
                    adj_mat_copy = deepcopy(adj_mat)

                if isinstance(self.list_of_tensors[i_tensors[0]], Cumulant):
                    ordered = sorted(i_tensors, key=lambda j: [adj_mat[i][j] for i in range(n_non_cu)])
                    for i, j in zip(i_tensors, ordered):
                        ordered_tensors[i] = self.list_of_tensors[j]
                        for k in range(n_non_cu):
                            adj_mat[k][i] = adj_mat_copy[k][j]

        self._list_of_tensors = ordered_tensors
        self._sorted = False

        return adj_mat

    def _minimal_indices(self):
        """
        Create a replacement map using minimal index labels to relabel the current term.
        :return: a replacement map {old index label: new index label}
        """
        replacement = {}
        next_index_number = {i: 0 for i in self.next_index_number.keys()}
        n_indices = len(self.indices_set)
        for tensor in [self.sq_op] + self.list_of_tensors:
            for upper in tensor.upper_indices:
                if upper in replacement:
                    continue
                replacement[upper] = self._generate_next_index(upper.space, next_index_number)
            for lower in tensor.lower_indices:
                if lower in replacement:
                    continue
                replacement[lower] = self._generate_next_index(lower.space, next_index_number)
            if len(replacement.keys()) == n_indices:
                break

        return replacement

    def _relabel_indices(self, replacement):
        """
        Relabel indices of this term according to the replacement map and resort the indices ordering.
        :param replacement: the replacement map
        :return: a tuple of (sign, sorted tensors, sorted sq_op)
        """
        sign = 1

        sq_op = self._replace_sqop_indices(self.sq_op, replacement)
        sq_op, _sign = sq_op.canonicalize()
        sign *= _sign

        list_of_tensors = self._replace_tensors_indices(self.list_of_tensors, replacement)
        for i, tensor in enumerate(list_of_tensors):
            list_of_tensors[i], _sign = tensor.canonicalize()
            sign *= _sign

        return sign, list_of_tensors, sq_op

    def canonicalize_simple(self):
        """
        Relabel the term using minimal index labels and reorder indices.
        :return: the relabeled term
        """
        replacement = self._minimal_indices()

        sign, list_of_tensors, sq_op = self._relabel_indices(replacement)

        return Term(list_of_tensors, sq_op, self.coeff * sign)

    def canonicalize(self, simplify_core_cumulant=True, remove_active_amplitudes=True):
        """
        Bring the current term to canonical form, which is defined by a sequence of ordering:
        1. order tensor by connection to Hamiltonian
        2. relabel indices
        :param simplify_core_cumulant: change a cumulant labeled by core indices to a Kronecker delta
        :param remove_active_amplitudes: remove terms when its contains all-active amplitudes
        :return: the "canonical" form of this term
        """
        # remove Kronecker delta, remove active amplitudes, simplify cumulant indices
        self.simplify(simplify_core_cumulant, remove_active_amplitudes)
        if self.coeff == 0:
            return self

        # order tensors according to adjacency matrix
        self.order_tensors(simplified=True)

        # create replacement map according to connections
        replacement = {}
        next_index = {i: 0 for i in space_priority}
        n_non_cu = sum([not isinstance(i, Cumulant) for i in self.list_of_tensors])

        open_indices = self.sq_op.cre_ops.indices_set ^ self.sq_op.ann_ops.indices_set
        if not self.sq_op.is_empty():
            for tensor in self.list_of_tensors[:n_non_cu]:
                for i in [j for j in tensor.upper_indices if j in open_indices]:
                    replacement[i] = self._generate_next_index(i.space, next_index)
                for i in [j for j in tensor.lower_indices if j in open_indices]:
                    replacement[i] = self._generate_next_index(i.space, next_index)

        for i, tensor1 in enumerate(self.list_of_tensors[:n_non_cu]):
            for tensor2 in self.list_of_tensors[i + 1:]:
                for k in sorted(tensor1.upper_indices.indices_set & tensor2.lower_indices.indices_set):
                    replacement[k] = self._generate_next_index(k.space, next_index)
                for k in sorted(tensor1.lower_indices.indices_set & tensor2.upper_indices.indices_set):
                    replacement[k] = self._generate_next_index(k.space, next_index)

        sign, list_of_tensors, sq_op = self._relabel_indices(replacement)

        self._list_of_tensors = list_of_tensors
        self._sq_op = sq_op
        self.coeff *= sign

        # relabel indices
        return self.canonicalize_simple()

    def canonicalize_sympy(self, simplify_core_cumulant=True, remove_active_amplitudes=True):
        """
        Bring the current term to canonical form using SymPy.
        :param simplify_core_cumulant: change a cumulant labeled by core indices to a Kronecker delta
        :param remove_active_amplitudes: remove terms when its contains all-active amplitudes
        :return: the "canonical" form of this term
        """
        # remove Kronecker delta, remove active amplitudes, simplify cumulant indices
        self.simplify(simplify_core_cumulant, remove_active_amplitudes)
        if self.coeff == 0:
            return self

        # use SymPy to canonicalize tensor indices
        # both tensors and sq-operator are considered contracted (dummies)
        minimal_indices_map = self._minimal_indices()

        dummy_indices = sorted(minimal_indices_map.values())
        dummy_group = [list(values) for k, values in groupby(dummy_indices, key=lambda x: x.space)]
        dummy_count = [0] + list(accumulate(map(len, dummy_group)))
        dummies = [range(2 * i, 2 * j) for i, j in zip(dummy_count[:-1], dummy_count[1:])]

        # figure out permutation g, note we put sq_op as the first "tensor"
        indices_tracker = {v: 2 * i for i, v in enumerate(dummy_indices)}
        g = []
        for tensor in [self.sq_op] + self.list_of_tensors:
            for index in tensor.lower_indices + tensor.upper_indices:
                minimal_index = minimal_indices_map[index]
                g.append(indices_tracker[minimal_index])
                indices_tracker[minimal_index] += 1

        consider_sign = isinstance(self.list_of_tensors[0].upper_indices, IndicesAntisymmetric)
        if consider_sign:
            g += [len(g), len(g) + 1]

        # figure out equivalent tensors and tensor/sq_op base and strong generating set
        bsgs_list = [] if self.sq_op.n_ops == 0 else [self.sq_op.base_strong_generating_set() + (1, 0)]
        tensor_count = [sum(1 for i in group) for k, group in groupby(self.list_of_tensors,
                                                                      key=lambda x: (x.name, x.n_body))]
        shift = 0
        for count in tensor_count:
            base, gens = self.list_of_tensors[shift].base_strong_generating_set()
            bsgs_list.append((base, gens, count, 0))
            shift += count

        # canonicalize indices
        gc = canonicalize(Permutation(g), dummies, [0] * len(dummies), *bsgs_list)

        # translate gc to Tensor and SecondQuantizedOperator
        ann_ops = self.sq_op.indices_type([dummy_indices[i // 2] for i in gc[:self.sq_op.n_ann]])
        cre_ops = self.sq_op.indices_type([dummy_indices[i // 2] for i in gc[self.sq_op.n_ann:self.sq_op.size]])
        sq_op = SecondQuantizedOperator(cre_ops, ann_ops)

        shift = self.sq_op.size
        list_of_tensors = []
        for tensor in self.list_of_tensors:
            lower_indices = tensor.indices_type([dummy_indices[gc[i + shift] // 2] for i in range(tensor.n_lower)])
            shift += tensor.n_lower
            upper_indices = tensor.indices_type([dummy_indices[gc[i + shift] // 2] for i in range(tensor.n_upper)])
            shift += tensor.n_upper

            list_of_tensors.append(tensor.from_indices(upper_indices, lower_indices))

        sign = 1
        if consider_sign and g[-1] != gc[-1]:
            sign = -1

        return Term(list_of_tensors, sq_op, sign * self.coeff)

    def generate_spin_cases_naive(self):
        """
        Generate a list of spin-integrated Term objects.
        :return: a list of spin-integrated Terms
        """
        if self.sq_op.indices_type is not IndicesSpinOrbital:
            raise TypeError(f"Invalid indices, expected 'IndicesSpinOrbital', given '{self.sq_op.indices_type}'.")

        terms = []

        for pairs in product(self.sq_op.generate_spin_cases(),
                             *[i.generate_spin_cases() for i in self.list_of_tensors]):
            try:
                sq_op = pairs[0]
                list_of_tensors = list(pairs[1:])
                terms.append(Term(list_of_tensors, sq_op, self.coeff, need_to_sort=False).canonicalize_sympy())
            except ValueError:
                pass

        if len(terms) == 0:
            return terms
        else:
            terms = sorted(terms)

        out = [terms[0]]
        for term in terms[1:]:
            if term.almost_equal(out[-1]):
                out[-1].coeff += term.coeff
            else:
                out.append(term)

        return out

    # TODO: need to check spin cases, if works for diagonal indices
    def make_excitation(self, single_ref):
        """
        Make a Term to excitation operator if possible.
        :param single_ref: use single-reference indices if True
        :return: excitation operator if possible, otherwise an empty Term
        """
        if not self.is_excitation():
            return self.void()

        hole = 'c' if single_ref else 'h'
        part = 'v' if single_ref else 'p'

        replacement = {}
        next_index_number = {i: 0 for i in self.next_index_number.keys()}
        for i in self.sq_op.cre_ops:
            overlap = space_relation[part] & space_relation[i.space]
            if len(overlap) == 0:
                return self.void()
            s = part if overlap == space_relation[part] else overlap.pop()
            replacement[i] = self._generate_next_index(s, next_index_number)
        for i in self.sq_op.ann_ops:
            overlap = space_relation[hole] & space_relation[i.space]
            if len(overlap) == 0:
                return self.void()
            s = hole if overlap == space_relation[hole] else overlap.pop()
            replacement[i] = self._generate_next_index(s, next_index_number)

        for tensor in self.list_of_tensors:
            for i in tensor.indices:
                if i not in replacement:
                    replacement[i] = self._generate_next_index(i.space, next_index_number)

        sign, list_of_tensors, sq_op = self._relabel_indices(replacement)

        return Term(list_of_tensors, sq_op, self.coeff * sign, False).canonicalize_sympy()

    # TODO: check if works for diagonal indices
    def make_one_body_diagonal(self, single_ref):
        """
        Make a one-body Term to a diagonal operator.
        :param single_ref: use single-reference indices if True
        :return: yield diagonal one-body terms
        """
        if not self.sq_op.n_body == 1:
            raise ValueError(f"{self} is not an one-body operator.")

        cre_index = self.sq_op.cre_ops.indices[0]
        ann_index = self.sq_op.ann_ops.indices[0]
        overlap = space_relation[cre_index.space] & space_relation[ann_index.space]

        if single_ref:
            if 'a' in overlap:
                overlap.remove('a')
            if 'A' in overlap:
                overlap.remove('A')

        for s in overlap:
            replacement = {}
            next_index_number = {i: 0 for i in self.next_index_number.keys()}
            replacement[ann_index] = self._generate_next_index(s, next_index_number)
            replacement[cre_index] = self._generate_next_index(s, next_index_number)

            for tensor in self.list_of_tensors:
                for i in tensor.indices:
                    if i not in replacement:
                        replacement[i] = self._generate_next_index(i.space, next_index_number)

            sign, list_of_tensors, sq_op = self._relabel_indices(replacement)

            yield Term(list_of_tensors, sq_op, self.coeff * sign, False).canonicalize_sympy()

    # TODO: check if works for diagonal indices
    def make_ddca(self, max_core, max_virt, single_ref):
        """
        Apply distinguished diagonal component approximation to this term.
        :param max_core: the max number of core indices kept
        :param max_virt: the max number of virtual indices kept
        :param single_ref: ignore active indices
        :return: yield filtered terms
        """
        indices_0 = self.sq_op.indices

        for spaces in product(*[space_relation[i.space] for i in indices_0]):
            count = {s: 0 for s in space_relation_so.keys()}
            for s in spaces:
                count[s.lower()] += 1
            if count['c'] > max_core or count['v'] > max_virt:
                continue
            if count['a'] != 0 and single_ref:
                continue

            replacement = {}
            next_index_number = {i: 0 for i in self.next_index_number.keys()}
            for index_0, space in zip(indices_0, spaces):
                replacement[index_0] = self._generate_next_index(space, next_index_number)
            for i in self.indices_set:
                if i not in replacement:
                    replacement[i] = self._generate_next_index(i.space, next_index_number)

            sign, list_of_tensors, sq_op = self._relabel_indices(replacement)

            yield Term(list_of_tensors, sq_op, self.coeff * sign, False).canonicalize_sympy()

    def expand_composite_indices(self, single_ref):
        """
        Expand composite indices in the SecondQuantizedOperator of this term.
        :param single_ref: ignore active indices
        :return: yield expanded terms
        """
        return self.make_ddca(self.sq_op.n_ops, self.sq_op.n_ops, single_ref)

#
# def read_latex_term(line):
#     sp = line.split()
#     try:
#         coeff = float(sp[0])
#     except:
#         n, d = sp[0].split("/")
#         coeff = float(n) / float(d)
#
#     list_of_tensors = []
#     indices = [[], []]
#     for s in sp[2:]:
#         sqop = SQOperator([], [])
#         if '^' in s:
#             name = s.split('^')[0]
#             lower = 0
#         elif s == '}_{':
#             lower = 1
#         elif s == '}':
#             tensor = None
#             if name == "H":
#                 tensor = Hamiltonian(indices[0], indices[1])
#             elif name == "T":
#                 tensor = ClusterAmp(indices[0], indices[1])
#             elif name == "L":
#                 tensor = Cumulant(indices[0], indices[1])
#             if tensor is not None:
#                 list_of_tensors.append(tensor)
#
#             if name == "a":
#                 sqop = SQOperator(indices[0], indices[1])
#
#             indices = [[], []]
#         else:
#             indices[lower].append(Index(s[0] + s[3:-1]))
#     return Term(list_of_tensors, sqop, coeff, need_to_sort=False)
#
#
# def read_latex_terms(lines):
#     terms = []
#     for line in lines.split('\\'):
#         terms.append(read_latex_term(line))
#     return terms
#
#
# # term = read_latex_term("1/64 & H^{ p_{0} p_{2} }_{ h_{4} h_{6} } T^{ h_{5} a_{0} }_{ p_{0} p_{1} } T^{ h_{7} h_{3} }_{ p_{2} a_{0} } L^{ h_{4} }_{ h_{5} } L^{ h_{6} }_{ h_{7} } a^{ p_{1} }_{ h_{3} }")
# # # print(term)
# # print(term.canonicalize())
#
# # text = """1/2 & H^{ p_{1} p_{2} }_{ h_{1} h_{2} } T^{ h_{0} h_{3} }_{ p_{0} a_{0} } T^{ h_{4} a_{0} }_{ p_{1} p_{2} } L^{ h_{1} }_{ h_{4} } L^{ h_{2} }_{ h_{3} } a^{ p_{0} }_{ h_{0} } \\
# # -7/8 & H^{ p_{1} p_{2} }_{ h_{1} h_{2} } T^{ h_{0} h_{3} }_{ p_{1} a_{0} } T^{ h_{4} a_{0} }_{ p_{0} p_{2} } L^{ h_{1} }_{ h_{4} } L^{ h_{2} }_{ h_{3} } a^{ p_{0} }_{ h_{0} } \\
# # -1/8 & H^{ p_{1} p_{2} }_{ h_{1} h_{2} } T^{ h_{0} h_{4} }_{ p_{2} a_{0} } T^{ h_{3} a_{0} }_{ p_{0} p_{1} } L^{ h_{1} }_{ h_{4} } L^{ h_{2} }_{ h_{3} } a^{ p_{0} }_{ h_{0} } \\
# # -1/2 & H^{ p_{1} p_{2} }_{ h_{1} h_{2} } T^{ h_{0} a_{0} }_{ p_{0} p_{1} } T^{ h_{3} h_{4} }_{ p_{2} a_{0} } L^{ h_{1} }_{ h_{3} } L^{ h_{2} }_{ h_{4} } a^{ p_{0} }_{ h_{0} } \\
# # -1/4 & H^{ p_{1} p_{2} }_{ h_{1} h_{2} } T^{ h_{0} a_{0} }_{ p_{1} p_{2} } T^{ h_{3} h_{4} }_{ p_{0} a_{0} } L^{ h_{1} }_{ h_{3} } L^{ h_{2} }_{ h_{4} } a^{ p_{0} }_{ h_{0} } """
#
# # text = """1 & H^{ g_{2} p_{1} }_{ h_{3} g_{1} } T^{ a_{1} h_{1} }_{ p_{0} p_{1} } T^{ h_{3} h_{4} }_{ p_{3} a_{0} } L^{ a_{0} }_{ a_{1} } a^{ g_{1} p_{0} p_{3} }_{ h_{4} g_{2} h_{1} } \\
# #           1 & H^{ p_{3} g_{2} }_{ g_{0} h_{1} } T^{ h_{0} h_{1} }_{ p_{0} a_{0} } T^{ a_{1} h_{4} }_{ p_{3} p_{4} } L^{ a_{0} }_{ a_{1} } a^{ g_{0} p_{0} p_{4} }_{ h_{4} g_{2} h_{0} } """
# #
# # terms = read_latex_terms(text)
# # for term in terms:
# #     print(term)
# #     print(term.canonicalize())


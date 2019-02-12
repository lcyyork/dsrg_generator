from copy import deepcopy
from collections import defaultdict
from fractions import Fraction
from math import factorial
from typing import List
from Index import Index, space_priority, space_relation
from Indices import Indices
from Tensor import Tensor, Cumulant, Hamiltonian, ClusterAmp, Kronecker
from SQOperator import SQOperator
from SpaceCounter import SpaceCounter


class Term:
    def __init__(self, list_of_tensors, operator, coeff=1.0, need_to_sort=True):
        """
        The Term class to store the tensors, coefficient, and SQOperator.
        :param list_of_tensors: a list of Tensor objects
        :param operator: a SQOperator object (open indices)
        :param coeff: the coefficient of the term
        :param need_to_sort: sort the list of tensors if True
        """

        # test if the input is valid
        # - coeff must be a number
        if not isinstance(coeff, float):
            try:
                self.coeff = float(coeff)
            except ValueError:
                print("Coefficient must be a number")
        self.coeff = coeff

        # - operator must of Operator type
        if not isinstance(operator, SQOperator):
            raise ValueError("Operator must be Operator type")
        self.sqop = operator

        # - list of tensors must of Tensor type
        # - upper indices and lower indices should form identical sets
        Uindices = operator.Uindices.set.copy()
        Lindices = operator.Lindices.set.copy()
        for t in list_of_tensors:
            if not isinstance(t, Tensor):
                raise ValueError("The list contains element not of Tensor type, {}".format(t))
            else:
                Uindices = Uindices | t.Uindices.set
                Lindices = Lindices | t.Lindices.set
        if not Uindices == Lindices:
            print("Upper Indices: {}".format(Uindices))
            print("Lower Indices: {}".format(Lindices))
            raise ValueError("Not a valid Term (must be connected).")

        self.list_of_tensors = sorted(list_of_tensors) if need_to_sort else list_of_tensors
        self.indices = Uindices

    def ambit(self, add_permutation=True, init_temp=True):
        """
        Translate to ambit form.
        :param add_permutation: add permutation contributions (when sqop contains difference MO spaces) if True
        :return: ambit form (string) of the Term
        """
        nopen = self.sqop.ncre + self.sqop.nann
        factor = factorial(self.sqop.ncre) * factorial(self.sqop.nann)

        tensors_str = " * ".join((i.ambit() for i in self.list_of_tensors))

        def format_coeff(value):
            fraction = Fraction(value).limit_denominator(1000000)
            if '/' not in str(fraction):
                return "{}.0 * ".format(fraction.numerator)
            else:
                return "({}.0 / {}.0) * ".format(fraction.numerator, fraction.denominator)

        exist_permute_format = not self.sqop.is_empty_sqop()
        if not self.sqop.is_empty_sqop():
            exist_permute_format = self.sqop.exist_permute_format()

        if not exist_permute_format:
            lhs = "C0" if self.sqop.is_empty_sqop() else \
                'C{}["{},{}"]'.format(nopen, self.sqop.Lindices.ambit(), self.sqop.Uindices.ambit())
            rhs = format_coeff(self.coeff * factor) + tensors_str
            return lhs + " += " + rhs + ";\n"
        else:
            space_str = "".join([i.space for i in self.sqop.Lindices] + [i.space for i in self.sqop.Uindices])
            if init_temp:
                out = 'temp = ambit::BlockedTensor::build(CoreTensor, "temp", {{"{}"}});\n'.format(space_str)
            else:
                out = ''

            nperm = self.sqop.n_multiset_permutation()
            coeff_str = format_coeff(self.coeff * factor / nperm)

            temp_str = 'temp["{},{}"]'.format(self.sqop.Lindices.ambit(), self.sqop.Uindices.ambit())
            out += "{} += {};\n".format(temp_str, coeff_str + tensors_str)

            if add_permutation:
                lhs_list = list(self.sqop.ambit_permute_format(True))
                for sign, lhs_indices in lhs_list:
                    lhs = 'C{}["{}"]'.format(nopen, lhs_indices)
                    out += "{} {}= {};\n".format(lhs, '+' if sign == 1 else '-', temp_str)

            return out

    def latex(self, dollar=False, permute_format=True, delimiter=False, backslash=False):
        """
        Translate to latex form.
        :param dollar: use math mode if True
        :param permute_format: use permute format (when sqop contains different MO spaces) if True
        :param delimiter: use & delimiter for latex align environment if True
        :param backslash: add \\ at the end for latex align environment if True
        :return: latex form (string) of the Term
        """
        tensors_str = " " + " ".join((i.latex() for i in self.list_of_tensors))

        nperm, perm_str, sqop_str = self.sqop.latex_permute_format() if permute_format else (1, "", self.sqop.latex())
        fraction = Fraction(self.coeff / nperm).limit_denominator(1000000)
        coeff_str = "{}".format(fraction)

        if delimiter:
            coeff_str += " &"
        if perm_str:
            perm_str = " " + perm_str
        if sqop_str:
            sqop_str = " " + sqop_str
        if backslash:
            sqop_str += " \\\\"

        out = coeff_str + perm_str + tensors_str + sqop_str
        if dollar:
            out = "$" + out + "$"
        return out

    def __repr__(self):
        return self.latex(permute_format=False)

    def __eq__(self, other):
        if self.coeff != other.coeff:
            return False
        if self.sqop != other.sqop:
            return False
        if self.list_of_tensors != other.list_of_tensors:
            return False
        return True

    def almost_equal(self, other):
        if self.sqop != other.sqop:
            return False
        if self.list_of_tensors != other.list_of_tensors:
            return False
        return True

    def __ne__(self, other):
        return not self == other

    def __lt__(self, other):
        if self.sqop != other.sqop:
            return True if self.sqop < other.sqop else False
        if len(self.list_of_tensors) != len(other.list_of_tensors):
            return True if len(self.list_of_tensors) < len(other.list_of_tensors) else False
        for i, j in zip(self.list_of_tensors, other.list_of_tensors):
            if i != j:
                return True if i < j else False
        if abs(self.coeff) != abs(other.coeff):
            return True if abs(self.coeff) < abs(other.coeff) else False
        return False

    def __le__(self, other):
        return self == other or self < other

    def __gt__(self, other):
        return not self <= other

    def __ge__(self, other):
        return not self < other

    @staticmethod
    def tensor_type(tensor):
        if isinstance(tensor, Cumulant):
            return Cumulant
        if isinstance(tensor, Hamiltonian):
            return Hamiltonian
        if isinstance(tensor, ClusterAmp):
            return ClusterAmp
        raise TypeError("Cannot use this function for general tensor.")

    def void_self(self):
        """
        Make current Term an empty Term.
        """
        self.coeff = 0.0
        self.sqop = SQOperator([], [])
        self.indices = set()
        self.list_of_tensors = []

    def simplify(self, remove_actv_amps=True):
        """
        Remove (in place) Kronecker deltas and downgrade cumulant indices.
        :param remove_actv_amps: remove terms when its contains all-active amplitudes
        """
        # check the next available index labels for each MO space
        indices_next_available = {i: 0 for i in space_priority}
        for i in self.indices:
            indices_next_available[i.space] = max(indices_next_available[i.space], i.number + 1)

        def next_available_index(space, add=True):
            """ Return the next available index of the given MO space. """
            out = Index("{}{}".format(space, indices_next_available[space]))
            if add:
                indices_next_available[space] += 1
            return out

        # indices replacement map
        replacement = {i: i for i in self.indices}

        # relabel densities and Kronecker deltas
        for ti, tensor in enumerate(self.list_of_tensors):
            # analyze Kronecker delta indices
            if isinstance(tensor, Kronecker):
                Uindex, Lindex = tensor.Uindices[0], tensor.Lindices[0]
                if len(space_relation[Uindex.space] & space_relation[Lindex.space]) == 0:
                    self.void_self()
                    return

                high, low = sorted([Uindex, Lindex])
                if high.space == 'p' and low.space == 'h':
                    # replace both with the same active index
                    replacement[Uindex] = next_available_index('a', add=False)
                    replacement[Lindex] = next_available_index('a')
                else:
                    # replace high priority index to low priority index
                    replacement[high] = low

            # analyze Cumulant
            if isinstance(tensor, Cumulant):
                # higher order cumulant have only active indices
                if tensor.nbody != 1:
                    for i in tensor.Uindices:
                        replacement[i] = next_available_index('a')
                    for i in tensor.Lindices:
                        replacement[i] = next_available_index('a')
                else:
                    Uindex, Lindex = tensor.Uindices[0], tensor.Lindices[0]
                    overlap = space_relation[Uindex.space] & space_relation[Lindex.space]
                    if len(overlap) == 0:
                        self.void_self()
                        return
                    elif len(overlap) == 1:
                        if 'c' in overlap:
                            replacement[Uindex] = next_available_index('c', False)
                            replacement[Lindex] = next_available_index('c')
                            self.list_of_tensors[ti] = Kronecker([Uindex], [Lindex])
                        elif 'a' in overlap:
                            replacement[Uindex] = next_available_index('a')
                            replacement[Lindex] = next_available_index('a')
                        else:
                            self.void_self()
                            return
                    else:
                        if space_relation["h"] <= overlap:
                            replacement[Uindex] = next_available_index('h')
                            replacement[Lindex] = next_available_index('h')
                        else:
                            replacement[Uindex] = next_available_index('a')
                            replacement[Lindex] = next_available_index('a')

        # relabel tensors using replacement map
        list_of_tensors = []
        for tensor in self.list_of_tensors:
            if not isinstance(tensor, Kronecker):
                if remove_actv_amps and isinstance(tensor, ClusterAmp):
                    indices = set([(replacement[i]).space for i in tensor.Uindices.indices + tensor.Lindices.indices])
                    if 'a' in indices and len(indices) == 1:
                        self.void_self()
                        return
                Uindices = Indices([replacement[i] for i in tensor.Uindices])
                Lindices = Indices([replacement[i] for i in tensor.Lindices])
                try:
                    list_of_tensors.append(Term.tensor_type(tensor)(Uindices, Lindices))
                except TypeError:
                    list_of_tensors.append(Tensor(tensor.name, Uindices, Lindices, tensor.priority))

        self.list_of_tensors = sorted(list_of_tensors)
        self.indices = set(replacement.values())

    def similar(self, other):
        return self.build_adjacency_matrix == other.build_adjacency_matrix

    def almost_similar(self, other):
        tensors1, adj1, (_, _) = self.order_tensors()
        tensors2, adj2, (_, _) = other.order_tensors()
        return adj1 == adj2

    def build_adjacency_matrix(self):
        """
        Build the adjacency matrix of the current Term (should be simplified already).
        Since densities cannot be connected among themselves, we only need the non-cumulant tensors.
        :return: the adjacency matrix specified by SpaceCounter
        """
        # number of tensors that are not cumulants
        noncu = sum([not isinstance(i, Cumulant) for i in self.list_of_tensors])

        # build adjacency matrix
        ntensors = len(self.list_of_tensors)
        adjacency_matrix: List[List[SpaceCounter]] = [[SpaceCounter() for j in range(ntensors)] for i in range(noncu)]

        for i in range(noncu):
            tensor1 = self.list_of_tensors[i]
            for j in range(i + 1, ntensors):
                tensor2 = self.list_of_tensors[j]
                adjacency_matrix[i][j].add(tensor1.Uindices.set & tensor2.Lindices.set, is_upper=True)
                adjacency_matrix[i][j].add(tensor1.Lindices.set & tensor2.Uindices.set, is_upper=False)

        return adjacency_matrix, (noncu, ntensors)

    def order_tensors(self):
        """
        Order tensors (after simplified) according to adjacency matrix.
        :return: a ordered list of Tensor objects and the corresponding adjacency matrix
        """
        # build adjacency matrix of the current term
        adj_mat, (noncu, ntensors) = self.build_adjacency_matrix()
        adj_mat_copy = deepcopy(adj_mat)

        # figure out possible equivalent tensors (same name and nbody)
        tensors = defaultdict(list)
        for i, tensor in enumerate(self.list_of_tensors):
            key = "{}{}".format(tensor.name, tensor.nbody)
            tensors[key].append(i)

        equiv_tensors = []
        ordered_tensors: List[Tensor] = [0] * ntensors
        for v in tensors.values():
            if len(v) > 1:
                equiv_tensors.append(v)
            else:
                ordered_tensors[v[0]] = self.list_of_tensors[v[0]]

        # connection to sqop (open indices)
        open_space_counters = [SpaceCounter() for i in range(ntensors)]
        for i in range(ntensors):
            tensor = self.list_of_tensors[i]
            open_space_counters[i].add(tensor.Uindices.set & self.sqop.Lindices.set, is_upper=True)
            open_space_counters[i].add(tensor.Lindices.set & self.sqop.Uindices.set, is_upper=False)

        # sort the list of tensors according to adjacency matrix
        for tensors in equiv_tensors:
            # reorder amplitudes
            if isinstance(self.list_of_tensors[tensors[0]], ClusterAmp):
                def criteria_1(i):  # connection to open indices
                    return open_space_counters[i]

                def criteria_2(i):  # connection to Hamiltonian
                    return adj_mat[0][i]

                def criteria_3(i):  # connection to cumulants
                    return sorted(adj_mat[i][noncu:])

                sorted_tensors = sorted(tensors, key=lambda i: [criteria_1(i), criteria_2(i)] + criteria_3(i))
                for i, j in zip(tensors, sorted_tensors):
                    ordered_tensors[i] = self.list_of_tensors[j]
                    adj_mat[i] = adj_mat_copy[j]
                adj_mat_copy = deepcopy(adj_mat)

            # reorder cumulants based on connections to Hamiltonian and amplitudes
            if isinstance(self.list_of_tensors[tensors[0]], Cumulant):
                sorted_tensors = sorted(tensors, key=lambda j: [adj_mat[i][j] for i in range(noncu)])
                for i, j in zip(tensors, sorted_tensors):
                    ordered_tensors[i] = self.list_of_tensors[j]
                    for k in range(noncu):
                        adj_mat[k][i] = adj_mat_copy[k][j]

        return ordered_tensors, adj_mat, (noncu, ntensors)

    def canonicalize(self, remove_actv_amps=True):
        """
        Bring the current term to canonical form, which is defined by a sequence of ordering:
        1. order tensor by connection to Hamiltonian
        2. relabel indices
        :param remove_actv_amps: remove terms when its contains all-active amplitudes
        :return: the "canonical" form of this term
        """
        # remove any Kronecker delta and simplify cumulant indices
        self.simplify(remove_actv_amps)
        if self.coeff == 0:
            return Term([], SQOperator([], []), 0.0)

        # reorder tensor sequence according to adjacency matrix
        ordered_tensors, adj_mat, (noncu, ntensors) = self.order_tensors()

        # find index isomorphism
        replacement = {}
        indices_next_available = {i: 0 for i in space_priority}

        def next_available_index(space, add=True):
            """ Return the next available index of the given MO space. """
            out = Index("{}{}".format(space, indices_next_available[space]))
            if add:
                indices_next_available[space] += 1
            return out

        # first determine the replacement for open indices
        open_indices = self.sqop.Uindices.set ^ self.sqop.Lindices.set
        if len(open_indices) != 0:
            for tensor in ordered_tensors:
                for i in [j for j in tensor.Uindices if j in open_indices]:
                    replacement[i] = next_available_index(i.space)
                for i in [j for j in tensor.Lindices if j in open_indices]:
                    replacement[i] = next_available_index(i.space)

        for i in range(noncu):
            tensor1 = ordered_tensors[i]
            for j in range(i + 1, ntensors):
                tensor2 = ordered_tensors[j]
                for k in sorted(tensor1.Uindices.set & tensor2.Lindices.set):
                    replacement[k] = next_available_index(k.space)
                for k in sorted(tensor1.Lindices.set & tensor2.Uindices.set):
                    replacement[k] = next_available_index(k.space)

        # apply replacement map to ordered_tensors
        sign = 1

        def relabel(indices, sign):
            temp, _sign = Indices([replacement[i] for i in indices]).canonicalize()
            return temp, sign * _sign

        cre_indices, sign = relabel(self.sqop.Uindices, sign)
        ann_indices, sign = relabel(self.sqop.Lindices, sign)

        for i, tensor in enumerate(ordered_tensors):
            Uindices, sign = relabel(tensor.Uindices, sign)
            Lindices, sign = relabel(tensor.Lindices, sign)
            try:
                ordered_tensors[i] = Term.tensor_type(tensor)(Uindices, Lindices)
            except TypeError:
                ordered_tensors[i] = Tensor(tensor.name, Uindices, Lindices, tensor.priority)

        # relabel indices to canonical form (from left to right, upper to lower)
        indices_next_available = {i: 0 for i in space_priority}
        current_indices = cre_indices.set | ann_indices.set
        for tensor in ordered_tensors:
            current_indices |= tensor.Uindices.set | tensor.Lindices.set
        relabeled = {i: None for i in current_indices}

        def relabel(indices, sign):
            list_of_index = []
            for i in indices:
                if relabeled[i] is None:
                    relabeled[i] = next_available_index(i.space)
                list_of_index.append(relabeled[i])
            temp, _sign = Indices(list_of_index).canonicalize()
            return temp, sign * _sign

        cre_indices, sign = relabel(cre_indices, sign)
        ann_indices, sign = relabel(ann_indices, sign)
        sqop = SQOperator(cre_indices, ann_indices)

        list_of_tensors = []
        for tensor in ordered_tensors:
            Uindices, sign = relabel(tensor.Uindices, sign)
            Lindices, sign = relabel(tensor.Lindices, sign)
            try:
                list_of_tensors.append(Term.tensor_type(tensor)(Uindices, Lindices))
            except TypeError:
                list_of_tensors.append(Tensor(tensor.name, Uindices, Lindices, tensor.priority))

        return Term(list_of_tensors, sqop, sign * self.coeff)


# h1 = Hamiltonian(["p0", "p1", "a0"], ["a1", "a2", "a3"])
# t1 = ClusterAmp(["a4", "a5", "a6"], ["p0", "p1", "a7"])
# l1 = Cumulant(["a1", "a3"], ["a0", "a4"])
# l2 = Cumulant(["a2", "a7"], ["a5", "a6"])
# term1 = Term([h1, t1, l1, l2], SQOperator([], []), 0.25)
# print(term1)
# print(term1.canonicalize())

# h1 = Hamiltonian(["g2", "g3"], ["g0", "g1"])
# t1 = ClusterAmp(["h0", "h1"], ["p0", "p1"])
# t2 = ClusterAmp(["h3", "h4"], ["p3", "p4"])
# k1 = Kronecker(["p3"], ["h1"])
# l1 = Cumulant(["g0"], ["h0"])
# l2 = Cumulant(["g1"], ["h3"])
# l3 = Cumulant(["p0"], ["h4"])
# l4 = Cumulant(["p1", "p4"], ["g3", "g2"])
# term2 = Term([h1, t1, t2, k1, l1, l2, l3, l4], SQOperator([], []), -1./64)
# print(term2)
# print(term2.canonicalize())

def read_latex_term(line):
    sp = line.split()
    try:
        coeff = float(sp[0])
    except:
        n, d = sp[0].split("/")
        coeff = float(n) / float(d)

    list_of_tensors = []
    indices = [[], []]
    for s in sp[2:]:
        sqop = SQOperator([], [])
        if '^' in s:
            name = s.split('^')[0]
            lower = 0
        elif s == '}_{':
            lower = 1
        elif s == '}':
            tensor = None
            if name == "H":
                tensor = Hamiltonian(indices[0], indices[1])
            elif name == "T":
                tensor = ClusterAmp(indices[0], indices[1])
            elif name == "L":
                tensor = Cumulant(indices[0], indices[1])
            if tensor is not None:
                list_of_tensors.append(tensor)

            if name == "a":
                sqop = SQOperator(indices[0], indices[1])

            indices = [[], []]
        else:
            indices[lower].append(Index(s[0] + s[3:-1]))
    return Term(list_of_tensors, sqop, coeff, need_to_sort=False)


def read_latex_terms(lines):
    terms = []
    for line in lines.split('\\'):
        terms.append(read_latex_term(line))
    return terms


# term = read_latex_term("1/64 & H^{ p_{0} p_{2} }_{ h_{4} h_{6} } T^{ h_{5} a_{0} }_{ p_{0} p_{1} } T^{ h_{7} h_{3} }_{ p_{2} a_{0} } L^{ h_{4} }_{ h_{5} } L^{ h_{6} }_{ h_{7} } a^{ p_{1} }_{ h_{3} }")
# # print(term)
# print(term.canonicalize())

# text = """1/2 & H^{ p_{1} p_{2} }_{ h_{1} h_{2} } T^{ h_{0} h_{3} }_{ p_{0} a_{0} } T^{ h_{4} a_{0} }_{ p_{1} p_{2} } L^{ h_{1} }_{ h_{4} } L^{ h_{2} }_{ h_{3} } a^{ p_{0} }_{ h_{0} } \\
# -7/8 & H^{ p_{1} p_{2} }_{ h_{1} h_{2} } T^{ h_{0} h_{3} }_{ p_{1} a_{0} } T^{ h_{4} a_{0} }_{ p_{0} p_{2} } L^{ h_{1} }_{ h_{4} } L^{ h_{2} }_{ h_{3} } a^{ p_{0} }_{ h_{0} } \\
# -1/8 & H^{ p_{1} p_{2} }_{ h_{1} h_{2} } T^{ h_{0} h_{4} }_{ p_{2} a_{0} } T^{ h_{3} a_{0} }_{ p_{0} p_{1} } L^{ h_{1} }_{ h_{4} } L^{ h_{2} }_{ h_{3} } a^{ p_{0} }_{ h_{0} } \\
# -1/2 & H^{ p_{1} p_{2} }_{ h_{1} h_{2} } T^{ h_{0} a_{0} }_{ p_{0} p_{1} } T^{ h_{3} h_{4} }_{ p_{2} a_{0} } L^{ h_{1} }_{ h_{3} } L^{ h_{2} }_{ h_{4} } a^{ p_{0} }_{ h_{0} } \\
# -1/4 & H^{ p_{1} p_{2} }_{ h_{1} h_{2} } T^{ h_{0} a_{0} }_{ p_{1} p_{2} } T^{ h_{3} h_{4} }_{ p_{0} a_{0} } L^{ h_{1} }_{ h_{3} } L^{ h_{2} }_{ h_{4} } a^{ p_{0} }_{ h_{0} } """

# text = """1 & H^{ g_{2} p_{1} }_{ h_{3} g_{1} } T^{ a_{1} h_{1} }_{ p_{0} p_{1} } T^{ h_{3} h_{4} }_{ p_{3} a_{0} } L^{ a_{0} }_{ a_{1} } a^{ g_{1} p_{0} p_{3} }_{ h_{4} g_{2} h_{1} } \\
#           1 & H^{ p_{3} g_{2} }_{ g_{0} h_{1} } T^{ h_{0} h_{1} }_{ p_{0} a_{0} } T^{ a_{1} h_{4} }_{ p_{3} p_{4} } L^{ a_{0} }_{ a_{1} } a^{ g_{0} p_{0} p_{4} }_{ h_{4} g_{2} h_{0} } """
#
# terms = read_latex_terms(text)
# for term in terms:
#     print(term)
#     print(term.canonicalize())


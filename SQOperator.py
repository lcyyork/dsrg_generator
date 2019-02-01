import copy
from Indices import Indices, IndicesAntisymmetric, IndicesSpinOrbital, IndicesSpinIntegrated, IndicesSpinAdapted
from sympy.utilities.iterables import multiset_permutations


class SecondQuantizedOperator:
    def __init__(self, upper_indices, lower_indices, indices_type):
        """
        The second-quantized operator class assuming the indices are antisymmetric.
        :param upper_indices: a list of Index or string for creation operators
        :param lower_indices: a list of Index or string for annihilation operators
        :param indices_type: the type of indices, check Indices subclasses
        """
        def init_indices(indices, indices_type):
            try:
                return Indices.make_indices(indices_type, indices)
            except TypeError:
                if issubclass(Indices, type(indices))
                    return copy.deepcopy(indices)
                else:
                    raise ValueError("Unable to initialize indices to Indices.")

        self._cre_ops = init_indices(upper_indices, indices_type)
        self._ann_ops = init_indices(lower_indices, indices_type)
        if isinstance(upper_indices, IndicesAntisymmetric):
            self._cre_ops = upper_indices
        else:
            self._cre_ops = IndicesAntisymmetric(upper_indices)

        if isinstance(lower_indices, IndicesAntisymmetric):
            self._ann_ops = lower_indices
        else:
            self._ann_ops = IndicesAntisymmetric(lower_indices)

        self._n_ann = self._ann_ops.size
        self._n_cre = self._cre_ops.size

    @property
    def cre_ops(self):
        return self._cre_ops

    @property
    def ann_ops(self):
        return self._ann_ops

    @property
    def n_ann(self):
        return self._n_ann

    @property
    def n_cre(self):
        return self._n_cre

    def __ne__(self, other):
        return not self == other

    def __eq__(self, other):
        return (self.cre_ops == other.cre_ops) and (self.ann_ops == other.ann_ops)

    def __lt__(self, other):
        if self.cre_ops < other.cre_ops:
            return True
        elif self.cre_ops == other.cre_ops:
            return True if self.ann_ops < other.ann_ops else False
        else:
            return False

    def __le__(self, other):
        return self == other or self < other

    def __gt__(self, other):
        return not self <= other

    def __ge__(self, other):
        return not self < other

    def __repr__(self):
        return self.latex()

    def is_empty(self):
        """ Return True is this SQOperator is empty. """
        return self.n_ops() == 0

    def n_ops(self):
        """ Return the sum of creation and annihilation operators. """
        return self.n_cre + self.n_ann

    def latex(self, dollar=False):
        """ Return the latex form (a string) of this SQOperator object. """
        if self.is_empty():
            return ""
        out = "a^{}_{}".format(self.cre_ops.latex(), self.ann_ops.latex())
        if dollar:
            out = "$ " + out + " $"
        return out

    def latex_permute_format(self):
        """
        Compute the multiset-permutation form of the SQOperator object.
        :return: the sign, the string for permutation, and the string for operator
        """
        if self.is_empty():
            return 1, '', ''
        nperm_cre, perm_cre = self.cre_ops.latex_permute_format()
        nperm_ann, perm_ann = self.ann_ops.latex_permute_format()
        whitespace = ' ' if perm_cre else ''
        return nperm_cre * nperm_ann, perm_cre + whitespace + perm_ann, self.latex()

    def ambit_permute_format(self, reverse_cre_ann=False):
        """ Generate the multiset-permutation form and the corresponding sign for ambit. """
        if self.is_empty():
            return [(1, '')]
        if not reverse_cre_ann:
            for sign_cre, str_cre in self.cre_ops.ambit_permute_format():
                for sign_ann, str_ann in self.ann_ops.ambit_permute_format():
                    yield sign_ann * sign_cre, str_cre + ',' + str_ann
        else:
            for sign_cre, str_cre in self.cre_ops.ambit_permute_format():
                for sign_ann, str_ann in self.ann_ops.ambit_permute_format():
                    yield sign_ann * sign_cre, str_ann + ',' + str_cre

    def n_multiset_permutation(self):
        """ Return the number of multiset permutations. """
        return self.cre_ops.n_multiset_permutation() * self.ann_ops.n_multiset_permutation()

    def exist_permute_format(self):
        """ Return True if there exists a multiset permutation of this SQOperator object. """
        cre_space = [i.space for i in self.cre_ops.indices]
        cre_exist = cre_space.count(cre_space[0]) != self.cre_ops.size
        if cre_exist:
            return True
        else:
            ann_space = [i.space for i in self.ann_ops.indices]
            return ann_space.count(ann_space[0]) != self.ann_ops.size




class SQOperator:
    def __init__(self, upper_indices, lower_indices):
        """
        The spin-orbital second-quantized operator class.
        :param upper_indices: a list of Index or string for creation operators
        :param lower_indices: a list of Index or string for annihilation operators
        """
        if isinstance(upper_indices, Indices):
            self.Uindices = upper_indices
        else:
            self.Uindices = Indices(upper_indices)

        if isinstance(lower_indices, Indices):
            self.Lindices = lower_indices
        else:
            self.Lindices = Indices(lower_indices)

        self.nann = self.Lindices.size
        self.ncre = self.Uindices.size

        return

    def __eq__(self, other):
        return (self.Uindices == other.Uindices) and (self.Lindices == other.Lindices)

    def __ne__(self, other):
        return not self == other

    def __lt__(self, other):
        if self.Uindices < other.Uindices:
            return True
        elif self.Uindices == other.Uindices:
            return True if self.Lindices < other.Lindices else False
        else:
            return False

    def __le__(self, other):
        return self == other or self < other

    def __gt__(self, other):
        return not self <= other

    def __ge__(self, other):
        return not self < other

    def __repr__(self):
        return self.latex()

    def latex(self, dollar=False):
        """ Return the latex form (a string) of this SQOperator object. """
        if self.is_empty_sqop():
            return ""
        out = "a^{}_{}".format(self.Uindices.latex(), self.Lindices.latex())
        if dollar:
            out = "$ " + out + " $"
        return out

    def latex_permute_format(self):
        """
        Compute the multiset-permutation form of the SQOperator object.
        :return: the sign, the string for permutation, and the string for operator
        """
        if self.is_empty_sqop():
            return 1, '', ''
        nperm_cre, perm_cre = self.Uindices.latex_permute_format()
        nperm_ann, perm_ann = self.Lindices.latex_permute_format()
        whitespace = ' ' if perm_cre else ''
        return nperm_cre * nperm_ann, perm_cre + whitespace + perm_ann, self.latex()

    def ambit_permute_format(self, reverse_cre_ann=False):
        """ Generate the multiset-permutation form and the corresponding sign for ambit. """
        if self.is_empty_sqop():
            return [(1, '')]
        if not reverse_cre_ann:
            for sign_cre, str_cre in self.Uindices.ambit_permute_format():
                for sign_ann, str_ann in self.Lindices.ambit_permute_format():
                    yield sign_ann * sign_cre, str_cre + ',' + str_ann
        else:
            for sign_cre, str_cre in self.Uindices.ambit_permute_format():
                for sign_ann, str_ann in self.Lindices.ambit_permute_format():
                    yield sign_ann * sign_cre, str_ann + ',' + str_cre

    def n_multiset_permutation(self):
        """ Return the number of multiset permutations. """
        return self.Uindices.n_multiset_permutation() * self.Lindices.n_multiset_permutation()

    def exist_permute_format(self):
        """ Return True if there exists a multiset permutation of this SQOperator object. """
        cre_space = [i.space for i in self.Uindices.indices]
        cre_exist = cre_space.count(cre_space[0]) != self.Uindices.size
        if cre_exist:
            return True
        else:
            ann_space = [i.space for i in self.Lindices.indices]
            return ann_space.count(ann_space[0]) != self.Lindices.size

    def is_empty_sqop(self):
        """ Return True is this SQOperator is empty. """
        return self.nops() == 0

    def nops(self):
        """ Return the sum of creation and annihilation operators. """
        return self.ncre + self.nann


def test_sqoperator_class():
    empty_sqop = SQOperator([], [])
    sqop1 = SQOperator(["g0", "g1", "g2"], ["p0", "p1", "p2"])
    sqop2 = SQOperator(["g0", "v1", "p2"], ["p0", "a1", "p2"])
    assert empty_sqop.is_empty_sqop() == True, "SQOperator is_empty failed."
    assert sqop1.exist_permute_format() == False, "SQOperator exist_permute_format failed."
    assert sqop1.n_multiset_permutation() == 1, "SQOperator n_multiset_permutation failed."
    assert sqop2.exist_permute_format() == True, "SQOperator exist_permute_format failed."
    assert sqop2.n_multiset_permutation() == 18, "SQOperator n_multiset_permutation failed."
    nperm, perm, latex_str = sqop1.latex_permute_format()
    assert nperm == 1, "SQOperator latex_permute_format failed."
    assert perm == '', "SQOperator latex_permute_format failed."
    assert latex_str == 'a^{ g_{0} g_{1} g_{2} }_{ p_{0} p_{1} p_{2} }', "SQOperator latex_permute_format failed."
    nperm, perm, latex_str = sqop2.latex_permute_format()
    assert nperm == 18, "SQOperator latex_permute_format failed."
    assert perm == '{\\cal P}(g_{0} / p_{2} / v_{1}) {\\cal P}(p_{0} p_{2} / a_{1})', "SQOperator latex_permute_format failed."
    assert latex_str == 'a^{ g_{0} v_{1} p_{2} }_{ p_{0} a_{1} p_{2} }', "SQOperator latex_permute_format failed."
    print("SQOperator tests passed.")

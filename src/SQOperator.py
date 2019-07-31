from src.IndicesPair import IndicesPair


class SecondQuantizedOperator(IndicesPair):
    def __init__(self, cre_ops, ann_ops, indices_type='so'):
        """
        The second-quantized operator class.
        :param cre_ops: a Indices object / anything can be converted to Indices for creation operators
        :param ann_ops: a Indices object / anything can be converted to Indices for annihilation operators
        :param indices_type: the type of indices, used if the indices are not Indices
        """
        IndicesPair.__init__(self, cre_ops, ann_ops, indices_type)

    @classmethod
    def make_empty(cls, indices_type='so'):
        return cls([], [], indices_type)

    @property
    def cre_ops(self):
        return super().upper_indices

    @property
    def ann_ops(self):
        return super().lower_indices

    @property
    def n_cre(self):
        return super().n_upper

    @property
    def n_ann(self):
        return super().n_lower

    @property
    def n_ops(self):
        return super().size

    def _is_valid_operand(self, other):
        if not isinstance(other, SecondQuantizedOperator):
            raise TypeError(f"Cannot compare between 'SecondQuantizedOperator' and '{other.__class__.__name__}'.")
        self._is_valid_operand_indices(other)

    def __repr__(self):
        return self.latex()

    def latex(self, dollar=False):
        """
        Translate to latex form.
        :param dollar: True if use inline math for latex
        :return: a string of latex format
        """
        out = f"a{super().latex()}"
        if dollar:
            out = "$ " + out + " $"
        return out

    def ambit(self, cre_first=False):
        """
        Translate to ambit form.
        :param cre_first: True if creation indices come in front of annihilation indices
        :return: a string of ambit form
        """
        return super().ambit(cre_first)

    def is_empty(self):
        """ Return True is this object is empty. """
        return self.n_ops == 0

    def is_excitation(self):
        """
        Test if this second-quantized operator is a possible excitation operator.
        :return: True if this is a possible excitation operator, otherwise False
        """
        for i in self.cre_ops:
            if 'c' == i.space:
                return False
        for i in self.ann_ops:
            if 'v' == i.space:
                return False
        return True

    def is_particle_conserving(self):
        """ Return True if this object conserves particles. """
        return self.n_ann == self.n_cre

    def exist_permute_format(self, p_cre, p_ann):
        """
        Test if there is a non-trivial multiset permutation.
        :param p_cre: a partition of upper indices, e.g., [[i,j], [k], [l]] for P(ij/k/l)
        :param p_ann: a partition of lower indices, e.g., [[i,j], [k], [l]] for P(ij/k/l)
        :return: True if the number of multiset permutation is not one.
        """
        return self.cre_ops.exist_permute_format(p_cre) or self.ann_ops.exist_permute_format(p_ann)

    def n_multiset_permutation(self, p_cre, p_ann):
        """
        Compute the number of multiset permutations.
        :param p_cre: a partition of upper indices, e.g., [[i,j], [k], [l]] for P(ij/k/l)
        :param p_ann: a partition of lower indices, e.g., [[i,j], [k], [l]] for P(ij/k/l)
        :return: the number of multiset permutations
        """
        return self.cre_ops.n_multiset_permutation(p_cre) * self.ann_ops.n_multiset_permutation(p_ann)

    def latex_permute_format(self, p_cre, p_ann):
        """
        Compute the multiset-permutation form of the SQOperator object.
        :param p_cre: a partition of upper indices, e.g., [[i,j], [k], [l]] for P(ij/k/l)
        :param p_ann: a partition of lower indices, e.g., [[i,j], [k], [l]] for P(ij/k/l)
        :return: a tuple of (the number of multiset permutations, a string for permutation, a string for operator)
        """
        if self.is_empty():
            return 1, '', ''
        n_perm_cre, perm_cre = self.cre_ops.latex_permute_format(p_cre)
        n_perm_ann, perm_ann = self.ann_ops.latex_permute_format(p_ann)
        whitespace = ' ' if perm_cre and perm_ann else ''
        return n_perm_cre * n_perm_ann, perm_cre + whitespace + perm_ann, self.latex()

    def ambit_permute_format(self, p_cre, p_ann, cre_first=False):
        """
        Generate the multiset-permutation form for ambit.
        :param p_cre: a partition of upper indices, e.g., [[i,j], [k], [l]] for P(ij/k/l)
        :param p_ann: a partition of lower indices, e.g., [[i,j], [k], [l]] for P(ij/k/l)
        :param cre_first: True if creation operators comes before annihilation operators
        :return: a tuple of (sign, a string representation of operator)
        """
        if self.is_empty():
            yield 1, ''
        else:
            first, second = (self.cre_ops, self.ann_ops) if cre_first else (self.ann_ops, self.cre_ops)
            p1, p2 = (p_cre, p_ann) if cre_first else (p_ann, p_cre)
            for sign_1, str_1 in first.ambit_permute_format(p1):
                for sign_2, str_2 in second.ambit_permute_format(p2):
                    yield sign_1 * sign_2, f'["{str_1},{str_2}"]'

    def generate_spin_cases(self, particle_conserving=True):
        """
        Generate spin-integrated second-quantized operator from spin-orbital indices.
        :param particle_conserving: True if generated indices preserve the spin
        :return: a SecondQuantizedOperator using spin-integrated indices pair
        """
        for upper, lower in self.generate_spin_cases_indices(particle_conserving):
            yield SecondQuantizedOperator(upper, lower, 'si')

    def canonicalize(self):
        """
        Sort the indices to canonical order.
        :return: a tuple of (sorted SecondQuantizedOperator, sign change)
        """
        upper, lower, sign = self.canonicalize_indices()
        return SecondQuantizedOperator(upper, lower), sign

    def void(self):
        """ Return an empty SecondQuantizedOperator. """
        return SecondQuantizedOperator(self.indices_type([]), self.indices_type([]))

    def base_strong_generating_set(self, hermitian=False):
        """
        Return the base and strong generating set.
        :param hermitian: upper and lower indices can be swapped if True
        :return: a tuple of (base, strong generating set)
        """
        return super().base_strong_generating_set(False)

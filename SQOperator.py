from IndicesPair import IndicesPair, make_indices_pair


def make_sqop(upper_indices, lower_indices, indices_type):
    """
    Initialize a SecondQuantizedOperator object from upper and lower indices.
    :param upper_indices: a list of Index or string for upper indices
    :param lower_indices: a list of Index or string for lower indices
    :param indices_type: the type of indices
    :return: a SecondQuantizedOperator object
    """
    return SecondQuantizedOperator(make_indices_pair(upper_indices, lower_indices, indices_type))


class SecondQuantizedOperator:
    def __init__(self, indices_pair):
        """
        The second-quantized operator class.
        :param indices_pair: a IndicesPair object
        """
        if not isinstance(indices_pair, IndicesPair):
            raise TypeError(f"Invalid indices_pair ('{indices_pair.__class__.__name__}'), requires 'IndicesPair'.")

        self._indices_pair = indices_pair

    @property
    def indices_pair(self):
        return self._indices_pair

    @property
    def cre_ops(self):
        return self.indices_pair.upper_indices

    @property
    def ann_ops(self):
        return self.indices_pair.lower_indices

    @property
    def n_cre(self):
        return self.indices_pair.n_upper

    @property
    def n_ann(self):
        return self.indices_pair.n_lower

    @property
    def n_ops(self):
        return self.n_cre + self.n_ann

    @property
    def type_of_indices(self):
        return self.indices_pair.type_of_indices

    @property
    def string_form(self):
        return self.type_of_indices(self.cre_ops.indices + self.ann_ops.indices[::-1])

    @staticmethod
    def _is_valid_operand(other):
        if not isinstance(other, SecondQuantizedOperator):
            raise TypeError(f"Cannot compare between 'SecondQuantizedOperator' and '{other.__class__.__name__}'.")

    def __eq__(self, other):
        self._is_valid_operand(other)
        return self.indices_pair == other.indices_pair

    def __ne__(self, other):
        self._is_valid_operand(other)
        return self.indices_pair != other.indices_pair

    def __lt__(self, other):
        self._is_valid_operand(other)
        return self.indices_pair < other.indices_pair

    def __le__(self, other):
        self._is_valid_operand(other)
        return self.indices_pair <= other.indices_pair

    def __gt__(self, other):
        self._is_valid_operand(other)
        return self.indices_pair > other.indices_pair

    def __ge__(self, other):
        self._is_valid_operand(other)
        return self.indices_pair >= other.indices_pair

    def __repr__(self):
        return self.latex()

    def latex(self, dollar=False):
        """
        Translate to latex form.
        :param dollar: True if use inline math for latex
        :return: a string of latex format
        """
        if self.is_empty():
            return ""
        out = f"a{self.indices_pair.latex()}"
        if dollar:
            out = "$ " + out + " $"
        return out

    def ambit(self, cre_first=False):
        """
        Translate to ambit form.
        :param cre_first: True if creation indices come in front of annihilation indices
        :return: a string of ambit form
        """
        return self.indices_pair.ambit(cre_first)

    def is_empty(self):
        """ Return True is this object is empty. """
        return self.n_ops == 0

    def is_particle_conserving(self):
        """ Return True if this object conserves particles. """
        return self.n_ann == self.n_cre

    def is_spin_conserving(self):
        """ Return True if conserves spin Ms. """
        if self.n_cre == self.n_ann:
            return self.cre_ops.n_beta() == self.ann_ops.n_beta()
        raise ValueError(f"Invalid quest, n_cre ({self.n_cre}) != n_ann ({self.n_ann}).")

    def exist_permute_format(self):
        """ Return True if there exists a multiset permutation of this object. """
        return self.cre_ops.exist_permute_format() or self.ann_ops.exist_permute_format()

    def n_multiset_permutation(self):
        """ Return the number of multiset permutations. """
        return self.cre_ops.n_multiset_permutation() * self.ann_ops.n_multiset_permutation()

    def latex_permute_format(self):
        """
        Compute the multiset-permutation form of the SQOperator object.
        :return: a tuple of (the number of multiset permutations, a string for permutation, a string for operator)
        """
        if self.is_empty():
            return 1, '', ''
        n_perm_cre, perm_cre = self.cre_ops.latex_permute_format()
        n_perm_ann, perm_ann = self.ann_ops.latex_permute_format()
        whitespace = ' ' if perm_cre and perm_ann else ''
        return n_perm_cre * n_perm_ann, perm_cre + whitespace + perm_ann, self.latex()

    def ambit_permute_format(self, cre_first=False):
        """
        Generate the multiset-permutation form for ambit.
        :param cre_first: True if creation operators comes before annihilation operators
        :return: a tuple of (sign, a string representation of operator)
        """
        if self.is_empty():
            yield 1, ''
        else:
            first, second = (self.cre_ops, self.ann_ops) if cre_first else (self.ann_ops, self.cre_ops)
            for sign_1, str_1 in first.ambit_permute_format():
                for sign_2, str_2 in second.ambit_permute_format():
                    yield sign_1 * sign_2, f'["{str_1},{str_2}"]'

    def generate_spin_cases(self, particle_conserving=True):
        """
        Generate spin-integrated second-quantized operator from spin-orbital indices.
        :param particle_conserving: True if generated indices preserve the spin
        :return: a SecondQuantizedOperator using spin-integrated indices pair
        """
        for indices_pair in self.indices_pair.generate_spin_cases(particle_conserving):
            yield SecondQuantizedOperator(indices_pair)

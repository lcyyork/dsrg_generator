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
        return self.cre_ops.size

    @property
    def n_ann(self):
        return self.ann_ops.size

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
        """ Return the latex form (a string) of this SQOperator object. """
        if self.is_empty():
            return ""
        out = f"a{self.indices_pair.latex()}"
        if dollar:
            out = "$ " + out + " $"
        return out

    def is_empty(self):
        """ Return True is this SQOperator is empty. """
        return self.n_ops() == 0

    def n_ops(self):
        """ Return the sum of creation and annihilation operators. """
        return self.n_cre + self.n_ann

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

    def ambit_permute_format(self, reverse_cre_ann=False):
        """
        Generate the multiset-permutation form for ambit.
        :param reverse_cre_ann: True if ann comes before cre
        :return: a tuple of (sign, a string representation of operator)
        """
        if self.is_empty():
            yield 1, ''
        else:
            first, second = (self.ann_ops, self.cre_ops) if reverse_cre_ann else (self.cre_ops, self.ann_ops)
            print(first, second)
            for sign_1, str_1 in first.ambit_permute_format():
                print(sign_1, str_1)
                for sign_2, str_2 in second.ambit_permute_format():
                    print(sign_2, str_2)
                    yield sign_1 * sign_2, str_1 + ',' + str_2

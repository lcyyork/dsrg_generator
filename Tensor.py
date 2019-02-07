from IndicesPair import IndicesPair


class Tensor:
    def __init__(self, name, indices_pair, priority=0):
        """
        The tensor class.
        :param name: the tensor name
        :param indices_pair: a IndicesPair object for the upper and lower indices
        :param priority: a integer for priority when sorting
        """
        if not isinstance(name, str):
            raise TypeError(f"Invalid tensor::name, given '{name.__class__.__name__}', required 'string'.")
        self._name = name

        if not isinstance(indices_pair, IndicesPair):
            t = f"{indices_pair.__class__.__name__}"
            raise TypeError(f"Invalid tensor::indices_pair, given '{t}', required 'IndicesPair'.")
        self._indices_pair = indices_pair

        if not isinstance(property, int):
            raise TypeError(f"Invalid tensor::priority, given '{priority.__class__.__name__}', required 'int'.")
        self._priority = priority

    @property
    def name(self):
        return self._name

    @property
    def indices_pair(self):
        return self._indices_pair

    @property
    def priority(self):
        return self._priority

    @property
    def upper_indices(self):
        return self.indices_pair.upper_indices

    @property
    def lower_indices(self):
        return self.indices_pair.lower_indices

    @property
    def n_upper(self):
        return self.indices_pair.n_upper

    @property
    def n_lower(self):
        return self.indices_pair.n_lower

    @property
    def size(self):
        return self.indices_pair.size

    @property
    def comparison_tuple(self):
        return self.priority, self.name, self.size, self.indices_pair

    @staticmethod
    def _is_valid_operand(other):
        if not isinstance(other, Tensor):
            raise TypeError(f"Cannot compare between 'Tensor' and '{other.__class__.__name__}'.")

    def __eq__(self, other):
        self._is_valid_operand(other)
        return self.comparison_tuple == other.comparison_tuple

    def __ne__(self, other):
        self._is_valid_operand(other)
        return self.comparison_tuple != other.comparison_tuple

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
        return self.latex()

    def latex(self, dollar=False):
        out = f"{self.name}{self.indices_pair.latex()}"
        if dollar:
            out = "$ " + out + " $"
        return out

    def ambit(self):
        if self.n_lower == self.n_upper:
            return f"{self.name}{self.n_lower}{self.indices_pair.ambit()}"
        return f"{self.name}{self.size}{self.indices_pair.ambit()}"

    def is_permutation(self, other):
        """
        Test if two tensors only differ by permutations of indices.
        :param other: the compared Tensor object
        :return: True if two tensors only differ by permutations of indices
        """
        self._is_valid_operand(other)
        if self.comparison_tuple[:-1] != other.comparison_tuple[:-1]:
            return False
        else:
            return self.upper_indices.is_permutation(other.upper_indices) and \
                   self.lower_indices.is_permutation(other.lower_indices)

    def canonicalize(self):
        """
        Sort the Tensor indices to canonical order.
        :return: a tuple of (tensor with sorted indices, sign)
        """
        indices_pair, sign = self.indices_pair.canonicalize()
        return self.__class__(self.name, indices_pair, self.priority), sign


class Cumulant(Tensor):
    def __init__(self, upper_indices, lower_indices):
        Tensor.__init__(self, "L", upper_indices, lower_indices, priority=3)
        return


class HoleDensity(Tensor):
    def __init__(self, upper_indices, lower_indices):
        if len(upper_indices) != 1 or len(lower_indices) != 1:
            raise ValueError("Hole density should be of 1 body.")
        Tensor.__init__(self, "C", upper_indices, lower_indices, priority=3)
        return


class Kronecker(Tensor):
    def __init__(self, upper_indices, lower_indices):
        if len(upper_indices) != 1 or len(lower_indices) != 1:
            raise ValueError("Kronecker delta should be of 1 body.")
        Tensor.__init__(self, "K", upper_indices, lower_indices, priority=2)
        return


class ClusterAmp(Tensor):
    def __init__(self, upper_indices, lower_indices):
        Tensor.__init__(self, "T", upper_indices, lower_indices, priority=1)
        return


class Hamiltonian(Tensor):
    def __init__(self, upper_indices, lower_indices):
        Tensor.__init__(self, "H", upper_indices, lower_indices, priority=0)
        return


def test_tensor_class():
    tensor = Tensor("T", ['a1', 'a2'], ['c3', 'p0'])
    assert str(tensor) == "T^{ a_{1} a_{2} }_{ c_{3} p_{0} }", "Tensor format failed."
    assert tensor.ambit() == 'T2["a1,a2,c3,p0"]', "Tensor format ambit failed."
    tensor_c, tensor_sign = tensor.canonicalize_copy()
    assert tensor_c == Tensor("T", ['a1', 'a2'], ['p0', 'c3']), "Tensor canonicalize failed."
    assert tensor_sign == -1, "Tensor canonicalize sign failed."
    assert tensor_c <= tensor, "Tensor comparison <= failed."
    print("Tensor tests passed.")

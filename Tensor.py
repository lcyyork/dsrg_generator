from IndicesPair import IndicesPair, make_indices_pair


def make_tensor_preset(upper_indices, lower_indices, indices_type, tensor_type):
    """
    Create a Tensor subclass object from upper and lower indices.
    :param upper_indices: a list of Index or string for upper indices
    :param lower_indices: a list of Index or string for lower indices
    :param indices_type: the preset type of indices
    :param tensor_type: the preset type of tensor type
    :return: a Tensor subclass object
    """
    indices_pair = make_indices_pair(upper_indices, lower_indices, indices_type)
    return Tensor.make_tensor(tensor_type, indices_pair)


def make_tensor(name, upper_indices, lower_indices, indices_type, priority=0):
    """
    Create a Tensor object from upper and lower indices.
    :param name: the name of tensor
    :param upper_indices: a list of Index or string for upper indices
    :param lower_indices: a list of Index or string for lower indices
    :param indices_type: the preset type of indices
    :param priority: the priority of tensor
    :return: a Tensor object
    """
    indices_pair = make_indices_pair(upper_indices, lower_indices, indices_type)
    return Tensor(name, indices_pair, priority)


class Tensor:
    # available choices: 'cumulant', 'hole_density', 'Kronecker', 'cluster_amplitude', 'Hamiltonian'
    subclasses = dict()

    @classmethod
    def register_subclass(cls, tensor_type):
        def decorator(subclass):
            cls.subclasses[tensor_type] = subclass
            return subclass
        return decorator

    @classmethod
    def make_tensor(cls, tensor_type, params):
        if tensor_type not in cls.subclasses:
            raise KeyError(f"Invalid tensor type '{tensor_type}', not in {', '.join(Tensor.subclasses.keys())}.")
        return cls.subclasses[tensor_type](params)

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

    def generate_spin_cases(self, particle_conserving=True):
        """
        Generate tensors labeled by spin-integrated indices from spin-orbital indices.
        :param particle_conserving: True if generated indices preserve the spin
        :return: a Tensor object labeled by spin-integrated indices
        """
        for indices_pair in self.indices_pair.generate_spin_cases(particle_conserving):
            yield self.__class__(self.name, indices_pair, self.priority)


@Tensor.register_subclass('cumulant')
class Cumulant(Tensor):
    def __init__(self, indices_pair):
        Tensor.__init__(self, "L", indices_pair, priority=3)


@Tensor.register_subclass('hole_density')
class HoleDensity(Tensor):
    def __init__(self, indices_pair):
        Tensor.__init__(self, "C", indices_pair, priority=3)
        if self.n_upper != 1 or self.n_lower != 1:
            raise ValueError("Hole density should be of 1 body.")


@Tensor.register_subclass('Kronecker')
class Kronecker(Tensor):
    def __init__(self, indices_pair):
        Tensor.__init__(self, "K", indices_pair, priority=2)
        if self.n_upper != 1 or self.n_lower != 1:
            raise ValueError("Kronecker delta should be of 1 body.")


@Tensor.register_subclass('cluster_amplitude')
class ClusterAmplitude(Tensor):
    def __init__(self, indices_pair):
        Tensor.__init__(self, "T", indices_pair, priority=1)


@Tensor.register_subclass('Hamiltonian')
class Hamiltonian(Tensor):
    def __init__(self, indices_pair):
        Tensor.__init__(self, "H", indices_pair, priority=0)

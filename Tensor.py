from mo_space import space_relation
from IndicesPair import IndicesPair, make_indices_pair


def make_tensor_preset(tensor_type, upper_indices, lower_indices, indices_type=""):
    """
    Create a Tensor subclass object from upper and lower indices.
    :param tensor_type: the preset type of tensor type
    :param upper_indices: a list of Index or string for upper indices
    :param lower_indices: a list of Index or string for lower indices
    :param indices_type: the preset type of indices
    :return: a Tensor subclass object
    """
    indices_pair = make_indices_pair(upper_indices, lower_indices, indices_type)
    return Tensor.make_tensor(tensor_type, indices_pair)


def make_tensor(name, upper_indices, lower_indices, indices_type="", priority=0):
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
    return Tensor(indices_pair, name, priority)


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

    def __init__(self, indices_pair, name, priority=0):
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

        if not isinstance(priority, int):
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
    def n_body(self):
        if self.n_lower != self.n_upper:
            raise ValueError(f"Invalid quest because n_lower ({self.n_lower}) != n_upper ({self.n_upper}).")
        return self.n_lower

    @property
    def size(self):
        return self.indices_pair.size

    @property
    def comparison_tuple(self):
        return self.priority, self.name, self.size, self.indices_pair

    @property
    def type_of_indices(self):
        return self.indices_pair.type_of_indices

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
        return f"{self.name}_{self.n_upper}_{self.n_lower}{self.indices_pair.ambit()}"

    def downgrade_indices(self):
        """
        Downgrade the indices of this object.
        :return: the largest space label of possible downgrades
        """
        raise NotImplementedError("Only available for Cumulant, HoleDensity, and Kronecker.")

    def is_all_active(self):
        """ Return True if this Tensor contains only active indices. """
        return (self.upper_indices.count_index_space(['A', 'a']),
                self.lower_indices.count_index_space(['A', 'a'])) == (self.n_upper, self.n_lower)

    def is_spin_conserving(self):
        """ Return True if spin Ms is preserved. """
        if self.n_upper == self.n_lower:
            return self.upper_indices.n_beta() == self.lower_indices.n_beta()
        raise ValueError(f"Invalid quest, n_upper ({self.n_upper}) != n_lower ({self.n_lower}).")

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

    def any_overlapped_indices(self, other):
        """
        Test if two Tensor objects have same indices.
        :param other: the compared Tensor object
        :return: True if any indices are in common between self and other
        """
        self._is_valid_operand(other)
        return self.upper_indices.any_overlap(other.upper_indices) or \
            self.lower_indices.any_overlap(other.lower_indices)

    def canonicalize(self):
        """
        Sort the Tensor indices to canonical order.
        :return: a tuple of (tensor with sorted indices, sign)
        """
        indices_pair, sign = self.indices_pair.canonicalize()
        return self.__class__(indices_pair, self.name, self.priority), sign

    def generate_spin_cases(self, particle_conserving=True):
        """
        Generate tensors labeled by spin-integrated indices from spin-orbital indices.
        :param particle_conserving: True if generated indices preserve the spin
        :return: a Tensor object labeled by spin-integrated indices
        """
        for indices_pair in self.indices_pair.generate_spin_cases(particle_conserving):
            yield self.__class__(indices_pair, self.name, self.priority)


@Tensor.register_subclass('cumulant')
class Cumulant(Tensor):
    def __init__(self, indices_pair, name='L', priority=2):
        Tensor.__init__(self, indices_pair, name, priority)
        if not self.is_spin_conserving():
            raise ValueError("Cumulant should conserve spin Ms.")

    def downgrade_indices(self):
        """
        Downgrade indices for Cumulant: 1cu -> hole only, 2cu -> active only.
        :return: the largest space label of possible downgrades
        """
        if self.n_body != 1:
            raise NotImplementedError("Indices of higher-order cumulant should be all active."
                                      " No point to implement downgrade_indices here.")
        else:
            u_index, l_index = self.upper_indices[0], self.lower_indices[0]
            overlap = space_relation[u_index.space] & space_relation[l_index.space]
            n_overlap = len(overlap)

            if n_overlap == 1:
                overlap_space = next(iter(overlap))
                if overlap_space.lower() != 'v':
                    return overlap_space
            elif n_overlap > 1:
                hole_label = 'H' if u_index.is_beta() else 'h'
                if space_relation[hole_label] <= overlap:
                    return hole_label
                else:
                    return 'A' if u_index.is_beta() else 'a'

        return ''


@Tensor.register_subclass('hole_density')
class HoleDensity(Tensor):
    def __init__(self, indices_pair, name='C', priority=2):
        Tensor.__init__(self, indices_pair, name, priority)
        if self.n_upper != 1 or self.n_lower != 1:
            raise ValueError("Hole density should be of 1 body.")
        if not self.is_spin_conserving():
            raise ValueError("HoleDensity should converse spin Ms.")

    def downgrade_indices(self):
        """
        Downgrade indices for HoleDensity: particle only.
        :return: the largest space label of possible downgrades
        """
        u_index, l_index = self.upper_indices[0], self.lower_indices[0]
        overlap = space_relation[u_index.space] & space_relation[l_index.space]
        n_overlap = len(overlap)

        if n_overlap == 1:
            overlap_space = next(iter(overlap))
            if overlap_space.lower() != 'c':
                return overlap_space
        elif n_overlap > 1:
            particle_label = 'P' if u_index.is_beta() else 'p'
            if space_relation[particle_label] <= overlap:
                return particle_label
            else:
                return 'A' if u_index.is_beta() else 'a'

        return ''


@Tensor.register_subclass('Kronecker')
class Kronecker(Tensor):
    def __init__(self, indices_pair, name='K', priority=-1):
        Tensor.__init__(self, indices_pair, name, priority)
        if self.n_upper != 1 or self.n_lower != 1:
            raise ValueError("Kronecker delta should be of 1 body.")
        if not self.is_spin_conserving():
            raise ValueError("Kronecker should converse spin Ms.")

    def downgrade_indices(self):
        """
        Downgrade indices for Kronecker: (hole, particle) -> active, high priority -> low priority.
        :return: the largest space label of possible downgrades
        """
        high, low = sorted([self.upper_indices[0], self.lower_indices[0]])

        if len(space_relation[high.space] & space_relation[low.space]) == 0:
            return ''

        if (high.space.lower(), low.space.lower()) == ('p', 'h'):
            return 'A' if high.is_beta() else 'a'
        else:
            return low.space


@Tensor.register_subclass('cluster_amplitude')
class ClusterAmplitude(Tensor):
    def __init__(self, indices_pair, name='T', priority=1):
        Tensor.__init__(self, indices_pair, name, priority)
        if not self.is_spin_conserving():
            raise ValueError("ClusterAmplitude should converse spin Ms.")


@Tensor.register_subclass('Hamiltonian')
class Hamiltonian(Tensor):
    def __init__(self, indices_pair, name='H', priority=0):
        Tensor.__init__(self, indices_pair, name, priority)
        if not self.is_spin_conserving():
            raise ValueError("Hamiltonian should converse spin Ms.")

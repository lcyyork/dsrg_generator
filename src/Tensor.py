from src.mo_space import space_relation
from src.IndicesPair import IndicesPair


class Tensor(IndicesPair):
    # available choices: 'cumulant', 'hole_density', 'Kronecker', 'cluster_amplitude', 'Hamiltonian'
    subclasses = dict()
    subclasses_alias = {'cumulant': 'cumulant', 'lambda': 'cumulant', 'L': 'cumulant',
                        'hole_density': 'hole_density', 'eta': 'hole_density', 'C': 'hole_density',
                        'Kronecker': 'Kronecker', 'delta': 'Kronecker', 'K': 'Kronecker',
                        'cluster_amplitude': 'cluster_amplitude', 'T': 'cluster_amplitude',
                        'Hamiltonian': 'Hamiltonian', 'H': 'Hamiltonian'}

    @classmethod
    def register_subclass(cls, tensor_type):
        def decorator(subclass):
            cls.subclasses[tensor_type] = subclass
            return subclass
        return decorator

    @classmethod
    def make_tensor(cls, tensor_type, *params):
        if tensor_type not in cls.subclasses_alias:
            raise KeyError(f"Invalid tensor type '{tensor_type}', not in {', '.join(Tensor.subclasses.keys())}.")
        return cls.subclasses[cls.subclasses_alias[tensor_type]](*params)

    def __init__(self, upper_indices, lower_indices, indices_type='so', name='Tensor', priority=0):
        """
        The tensor class.
        :param upper_indices: Indices object / anything can be converted for upper indices
        :param lower_indices: Indices object / anything can be converted for lower indices
        :param indices_type: the type of indices, used if the indices are not Indices
        :param name: the tensor name
        :param priority: a integer for priority when sorting
        """
        if not isinstance(name, str):
            raise TypeError(f"Invalid tensor::name, given '{name.__class__.__name__}', required 'string'.")
        self._name = name

        if not isinstance(priority, int):
            raise TypeError(f"Invalid tensor::priority, given '{priority.__class__.__name__}', required 'int'.")
        self._priority = priority

        IndicesPair.__init__(self, upper_indices, lower_indices, indices_type)

    @property
    def name(self):
        return self._name

    @property
    def priority(self):
        return self._priority

    def clone(self):
        """ Make a copy. """
        return self.__class__(self.upper_indices, self.lower_indices, name=self.name, priority=self.priority)

    def from_indices(self, upper_indices, lower_indices, indices_type='so'):
        """
        Make a new tensor of same type using the input indices.
        :param upper_indices: Indices object / anything can be converted for upper indices
        :param lower_indices: Indices object / anything can be converted for lower indices
        :param indices_type: the type of indices, used if the indices are not Indices
        :return: a new tensor
        """
        return self.__class__(upper_indices, lower_indices, indices_type, self.name, self.priority)

    @property
    def comparison_tuple(self):
        return self.priority, self.name, self.size, self.upper_indices, self.lower_indices

    def _is_valid_operand(self, other):
        if not isinstance(other, Tensor):
            raise TypeError(f"Cannot compare between 'Tensor' and '{other.__class__.__name__}'.")
        self._is_valid_operand_indices(other)

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

    def __hash__(self):
        return hash(self.comparison_tuple)

    def __repr__(self):
        return self.latex()

    def latex(self, dollar=False):
        """
        Translate to latex form.
        :param dollar: True if use inline math for latex
        :return: a string of latex format
        """
        out = f"{self.name}{super().latex()}"
        if dollar:
            out = "$ " + out + " $"
        return out

    def ambit(self, upper_first=True):
        """
        Translate to ambit code form.
        :param upper_first: True if put upper indices in front of lower indices
        :return: a string in ambit format
        """
        if self.n_upper == self.n_lower:
            return f"{self.name}{self.n_upper}{super().ambit(upper_first)}"
        return f"{self.name}_{self.n_upper}_{self.n_lower}{super().ambit(upper_first)}"

    def downgrade_indices(self):
        """
        Downgrade the indices of this object.
        :return: the largest space label of possible downgrades
        """
        raise NotImplementedError("Only available for Cumulant, HoleDensity, and Kronecker.")

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
        upper_indices, lower_indices, sign = self.canonicalize_indices()
        tensor = self.__class__(upper_indices, lower_indices, name=self.name, priority=self.priority)
        return tensor, sign

    def generate_spin_cases(self, particle_conserving=True):
        """
        Generate tensors labeled by spin-integrated indices from spin-orbital indices.
        :param particle_conserving: True if generated indices preserve the spin
        :return: a Tensor object labeled by spin-integrated indices
        """
        for upper, lower in self.generate_spin_cases_indices(particle_conserving):
            yield self.__class__(upper, lower, name=self.name, priority=self.priority)


@Tensor.register_subclass('cumulant')
class Cumulant(Tensor):
    def __init__(self, upper_indices, lower_indices, indices_type='so', name='L', priority=2):
        Tensor.__init__(self, upper_indices, lower_indices, indices_type, name, priority)
        if not self.is_spin_conserving():
            raise ValueError("Cumulant should conserve spin Ms.")

    def downgrade_indices(self):
        """
        Downgrade indices for Cumulant: 1cu -> hole only, 2cu -> active only.
        :return: the largest space label of possible downgrades
        """
        if self.n_body == 1:
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

        raise NotImplementedError("Indices of higher-order cumulant should be all active."
                                  " No point to implement downgrade_indices here.")


@Tensor.register_subclass('hole_density')
class HoleDensity(Tensor):
    def __init__(self, upper_indices, lower_indices, indices_type='so', name='C', priority=2):
        Tensor.__init__(self, upper_indices, lower_indices, indices_type, name, priority)
        if self.n_upper != 1 or self.n_lower != 1:
            raise ValueError("Hole density should be of 1 body.")
        if not self.is_spin_conserving():
            raise ValueError("Hole density should converse spin Ms.")

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

    def expand(self):
        """
        Expand hole density to Kronecker delta minus one cumulant.
        :return: a list of possible expansions
        """
        upper, lower = self.upper_indices, self.lower_indices
        if upper[0].space.lower() == 'v' or lower[0].space.lower() == 'v':
            return [Kronecker(upper, lower)]
        return [Kronecker(upper, lower), Cumulant(upper, lower)]


@Tensor.register_subclass('Kronecker')
class Kronecker(Tensor):
    def __init__(self, upper_indices, lower_indices, indices_type='so', name='K', priority=-1):
        Tensor.__init__(self, upper_indices, lower_indices, indices_type, name, priority)
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
    def __init__(self, upper_indices, lower_indices, indices_type='so', name='T', priority=1):
        Tensor.__init__(self, upper_indices, lower_indices, indices_type, name, priority)

        if not self.is_spin_conserving():
            raise ValueError("ClusterAmplitude should converse spin Ms.")

        # hole and particle indices cannot appear in upper/lower at the same time
        def test_indices(block):
            indices = self.upper_indices if block == 'upper' else self.lower_indices
            de_ex_space = 'v' if block == 'upper' else 'c'

            hole, part, de_ex = False, False, False

            for i in indices:
                space = space_relation[i.space.lower()]
                if space == {'a'}:
                    continue
                if space <= space_relation['h']:
                    hole = True
                if space <= space_relation['p']:
                    part = True
                if de_ex_space in space:
                    de_ex = True

            if hole and part:
                raise ValueError(f"Invalid {block} indices for cluster amplitudes: "
                                 f"{indices} contains both hole and particle indices")

            return de_ex

        de_ex_upper = test_indices('upper')
        de_ex_lower = test_indices('lower')
        self._excitation = False if de_ex_upper or de_ex_lower else True

    def ambit(self, upper_first=True):
        return super().ambit(self._excitation)

    def downgrade_indices(self):
        raise NotImplementedError("Cannot downgrade indices for ClusterAmplitudes.")

    def is_all_active(self):
        return all(i.space.lower() == 'a' for i in self.indices)


@Tensor.register_subclass('Hamiltonian')
class Hamiltonian(Tensor):
    def __init__(self, upper_indices, lower_indices, indices_type='so', name='H', priority=0):
        Tensor.__init__(self, upper_indices, lower_indices, indices_type, name, priority)
        if not self.is_spin_conserving():
            raise ValueError("Hamiltonian should converse spin Ms.")

    def downgrade_indices(self):
        raise NotImplementedError("Cannot downgrade indices for Hamiltonian.")

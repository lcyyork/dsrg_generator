from typing import Type
from Indices import Indices, IndicesSpinOrbital


def make_indices_pair(upper_indices, lower_indices, indices_type: str):
        """
        Initialize a IndicesPair object from upper and lower indices.
        :param upper_indices: a list of Index or string for upper indices
        :param lower_indices: a list of Index or string for lower indices
        :param indices_type: the type of indices
        :return: a IndicesPair object
        """
        if indices_type not in Indices.subclasses:
            raise KeyError(f"Invalid indices type {indices_type}. Choices: {', '.join(Indices.subclasses.keys())}.")

        upper = Indices.make_indices(upper_indices, indices_type)
        lower = Indices.make_indices(lower_indices, indices_type)

        return IndicesPair(upper, lower)


class IndicesPair:
    def __init__(self, upper_indices: Type[Indices], lower_indices: Type[Indices]):
        """
        The IndicesPair class to handle upper and lower indices for tensors or second-quantized operators.
        :param upper_indices: a Indices object for upper indices
        :param lower_indices: a Indices object for lower indices
        """
        if not isinstance(upper_indices, Indices):
            raise TypeError(f"upper_indices ('{upper_indices.__class__.__name__}') is not of 'Indices' type.")
        if not isinstance(lower_indices, Indices):
            raise TypeError(f"lower_indices ('{lower_indices.__class__.__name__}') is not of 'Indices' type.")
        if type(upper_indices) != type(lower_indices):
            raise TypeError(f"Inconsistent type for upper_indices ('{upper_indices.__class__.__name__}') "
                            f"and lower_indices ('{lower_indices.__class__.__name__}').")

        self._upper_indices = upper_indices
        self._lower_indices = lower_indices
        self._n_upper = self._upper_indices.size
        self._n_lower = self._lower_indices.size

    @property
    def upper_indices(self):
        return self._upper_indices

    @property
    def lower_indices(self):
        return self._lower_indices

    @property
    def n_upper(self):
        return self._n_upper

    @property
    def n_lower(self):
        return self._n_lower

    @property
    def size(self):
        return self._n_upper + self._n_lower

    def is_empty(self):
        return self.size == 0

    @staticmethod
    def _is_valid_operand(other):
        if not isinstance(other, IndicesPair):
            raise TypeError(f"Cannot compare between 'IndicesPair' and '{other.__class__.__name__}'.")

    def __eq__(self, other):
        self._is_valid_operand(other)
        return (self.upper_indices, self.lower_indices) == (other.upper_indices, other.lower_indices)

    def __ne__(self, other):
        self._is_valid_operand(other)
        return (self.upper_indices, self.lower_indices) != (other.upper_indices, other.lower_indices)

    def __lt__(self, other):
        self._is_valid_operand(other)
        return (self.upper_indices, self.lower_indices) < (other.upper_indices, other.lower_indices)

    def __le__(self, other):
        self._is_valid_operand(other)
        return (self.upper_indices, self.lower_indices) <= (other.upper_indices, other.lower_indices)

    def __gt__(self, other):
        self._is_valid_operand(other)
        return (self.upper_indices, self.lower_indices) > (other.upper_indices, other.lower_indices)

    def __ge__(self, other):
        self._is_valid_operand(other)
        return (self.upper_indices, self.lower_indices) >= (other.upper_indices, other.lower_indices)

    def __repr__(self):
        return self.latex()

    def latex(self):
        """
        The latex form of IndicesPair
        :return: a string in latex format
        """
        return f"^{self.upper_indices.latex()}_{self.lower_indices.latex()}"

    def ambit(self):
        """
        The ambit form of IndicesPair
        :return: a string in ambit format
        """
        return f'["{self.upper_indices.ambit()},{self.lower_indices.ambit()}"]'

    def generate_spin_cases(self, particle_conserving=True):
        """
        Generate spin-integrated indices pair from spin-orbital indices.
        :param particle_conserving: True if generated indices preserve the spin
        :return: Spin-integrated indices pair
        """
        if not isinstance(self.upper_indices, IndicesSpinOrbital):
            raise TypeError("Only available for spin-orbital indices.")
        for upper_indices in self.upper_indices.generate_spin_cases():
            upper_spin = upper_indices.spin_count
            for lower_indices in self.lower_indices.generate_spin_cases():
                if particle_conserving:
                    if upper_spin == lower_indices.spin_count:
                        yield IndicesPair(upper_indices, lower_indices)
                else:
                    yield IndicesPair(upper_indices, lower_indices)

    def canonicalize(self):
        """
        Sort the IndicesPair to canonical order.
        :return: a tuple of (sorted IndicesPair, sign change)
        """
        upper_indices, upper_sign = self.upper_indices.canonicalize()
        lower_indices, lower_sign = self.lower_indices.canonicalize()
        return IndicesPair(upper_indices, lower_indices), upper_sign * lower_sign

from mo_space import space_priority, space_priority_so
from Index import Index


class SpaceCounter:
    def __init__(self, spin_orbital=True):
        """
        The space counter to count the MO spaces in a given indices.
        :param spin_orbital: True if using spin-orbital space_priority
        """
        self._space_priority = space_priority_so if spin_orbital else space_priority
        length = len(self._space_priority)
        self._upper = [0] * length
        self._lower = [0] * length
        self._n_upper = 0
        self._n_lower = 0

    @property
    def upper(self):
        return self._upper

    @property
    def lower(self):
        return self._lower

    @property
    def n_upper(self):
        return self._n_upper

    @property
    def n_lower(self):
        return self._n_lower

    @property
    def size(self):
        return self.n_upper + self.n_lower

    @staticmethod
    def _is_valid_indices_set(indices_set):
        if any([isinstance(i, Index) for i in indices_set]):
            raise TypeError("Invalid type in indices set.")

    def add_upper(self, indices_set):
        """
        Add the MO spaces of indices to upper set.
        :param indices_set: a set of Index objects
        """
        self._is_valid_indices_set(indices_set)
        for index in indices_set:
            self._upper[self._space_priority[index.space]] += 1
            self._n_upper += 1

    def add_lower(self, indices_set):
        """
        Add the MO spaces of indices to lower set.
        :param indices_set: a set of Index objects
        """
        self._is_valid_indices_set(indices_set)
        for index in indices_set:
            self._lower[self._space_priority[index.space]] += 1
            self._n_lower += 1

    def __eq__(self, other):
        return (self.upper, self.lower) == (other.upper, other.lower)

    def __ne__(self, other):
        return (self.upper, self.lower) != (other.upper, other.lower)

    def __lt__(self, other):
        return (self.size, self.upper, self.lower) < (other.size, other.upper, other.lower)

    def __le__(self, other):
        return (self.size, self.upper, self.lower) <= (other.size, other.upper, other.lower)

    def __gt__(self, other):
        return (self.size, self.upper, self.lower) > (other.size, other.upper, other.lower)

    def __ge__(self, other):
        return (self.size, self.upper, self.lower) >= (other.size, other.upper, other.lower)

    def __repr__(self):
        return f"SpaceCounter ({self.upper}, {self.lower})"

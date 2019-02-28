from mo_space import space_priority, space_priority_so
from Indices import IndicesSpinOrbital
from Tensor import Tensor


class SpaceCounter:
    def __init__(self, tensor1, tensor2):
        """
        The space counter class to count the MO spaces between two Tensors.
        :param tensor1: the 1st Tensor
        :param tensor2: the 2nd Tensor
        """
        if not isinstance(tensor1, Tensor):
            raise TypeError(f"Invalid input for tensor1, given '{tensor1.__class__.__name__}',"
                            f" required 'Tensor'.")
        if not isinstance(tensor2, Tensor):
            raise TypeError(f"Invalid input for tensor2, given '{tensor2.__class__.__name__}',"
                            f" required 'Tensor'.")
        if tensor1.type_of_indices is not tensor2.type_of_indices:
            raise ValueError(f"Two tensors are of different indices type:"
                             f" '{tensor1.type_of_indices}' and '{tensor2.type_of_indices}'.")

        _space_priority = space_priority_so if tensor1.type_of_indices is IndicesSpinOrbital else space_priority
        length = len(_space_priority)
        self._upper = [0] * length
        self._lower = [0] * length

        upper = tensor1.upper_indices.indices_set & tensor2.lower_indices.indices_set
        lower = tensor1.lower_indices.indices_set & tensor2.upper_indices.indices_set

        for index in upper:
            self._upper[_space_priority[index.space]] += 1
        for index in lower:
            self._lower[_space_priority[index.space]] += 1

        self._n_upper = sum(self._upper)
        self._n_lower = sum(self._lower)

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

    def _is_valid_operand(self, other):
        if other.__class__ is not SpaceCounter:
            raise TypeError(f"Cannot compare between '{other.__class__.__name__}' and 'SpaceCounter'.")
        if len(self.upper) != len(other.upper):
            raise ValueError(f"Two SpaceCounter objects assume different indices type.")

    def __eq__(self, other):
        self._is_valid_operand(other)
        return (self.upper, self.lower) == (other.upper, other.lower)

    def __ne__(self, other):
        self._is_valid_operand(other)
        return (self.upper, self.lower) != (other.upper, other.lower)

    def __lt__(self, other):
        self._is_valid_operand(other)
        return (self.size, self.upper, self.lower) < (other.size, other.upper, other.lower)

    def __le__(self, other):
        self._is_valid_operand(other)
        return (self.size, self.upper, self.lower) <= (other.size, other.upper, other.lower)

    def __gt__(self, other):
        self._is_valid_operand(other)
        return (self.size, self.upper, self.lower) > (other.size, other.upper, other.lower)

    def __ge__(self, other):
        self._is_valid_operand(other)
        return (self.size, self.upper, self.lower) >= (other.size, other.upper, other.lower)

    def __repr__(self):
        return f"SpaceCounter ({','.join(map(str, self.upper))}; {','.join(map(str, self.lower))})"


# class SpaceCounter:
#     def __init__(self, spin_orbital=True):
#         """
#         The space counter to count the MO spaces in a given indices.
#         :param spin_orbital: True if using spin-orbital space_priority
#         """
#         self._space_priority = space_priority_so if spin_orbital else space_priority
#         length = len(self._space_priority)
#         self._upper = [0] * length
#         self._lower = [0] * length
#         self._n_upper = 0
#         self._n_lower = 0
#
#     @property
#     def upper(self):
#         return self._upper
#
#     @property
#     def lower(self):
#         return self._lower
#
#     @property
#     def n_upper(self):
#         return self._n_upper
#
#     @property
#     def n_lower(self):
#         return self._n_lower
#
#     @property
#     def size(self):
#         return self.n_upper + self.n_lower
#
#     @staticmethod
#     def _is_valid_indices_set(indices_set):
#         if not all([isinstance(i, Index) for i in indices_set]):
#             raise TypeError("Invalid type in indices set.")
#
#     def add_upper(self, indices_set):
#         """
#         Add the MO spaces of indices to upper set.
#         :param indices_set: a set of Index objects
#         """
#         self._is_valid_indices_set(indices_set)
#         for index in indices_set:
#             self._upper[self._space_priority[index.space]] += 1
#             self._n_upper += 1
#
#     def add_lower(self, indices_set):
#         """
#         Add the MO spaces of indices to lower set.
#         :param indices_set: a set of Index objects
#         """
#         self._is_valid_indices_set(indices_set)
#         for index in indices_set:
#             self._lower[self._space_priority[index.space]] += 1
#             self._n_lower += 1
#
#     def __eq__(self, other):
#         return (self.upper, self.lower) == (other.upper, other.lower)
#
#     def __ne__(self, other):
#         return (self.upper, self.lower) != (other.upper, other.lower)
#
#     def __lt__(self, other):
#         return (self.size, self.upper, self.lower) < (other.size, other.upper, other.lower)
#
#     def __le__(self, other):
#         return (self.size, self.upper, self.lower) <= (other.size, other.upper, other.lower)
#
#     def __gt__(self, other):
#         return (self.size, self.upper, self.lower) > (other.size, other.upper, other.lower)
#
#     def __ge__(self, other):
#         return (self.size, self.upper, self.lower) >= (other.size, other.upper, other.lower)
#
#     def __repr__(self):
#         return f"SpaceCounter ({self.upper}, {self.lower})"

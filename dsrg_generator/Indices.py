import collections
from itertools import product, combinations
from sympy.combinatorics import Permutation
from sympy.utilities.iterables import multiset_permutations

from dsrg_generator.mo_space import space_priority
from dsrg_generator.Index import Index


def sort_and_count_inversions(array):
    """ Sort an array and count the number of inversions via merge sort. """
    n = len(array)
    if n <= 1:
        return array, 0
    else:
        n_half = n // 2

        left, x = sort_and_count_inversions(array[:n_half])
        right, y = sort_and_count_inversions(array[n_half:])
        result, z = merge_and_count_split_inversion(left, right)

        return result, x + y + z


def merge_and_count_split_inversion(left, right):
    """ Merge two lists (left, right) into sorted_result and count inversions. """
    i, j = 0, 0
    count = 0
    n_left = len(left)
    n_right = len(right)
    n = n_left + n_right
    sorted_result = [0] * n

    for k in range(n):
        if i == n_left:
            sorted_result[k] = right[j]
            j += 1
            continue

        if j == n_right:
            sorted_result[k] = left[i]
            i += 1
            continue

        if left[i] <= right[j]:
            sorted_result[k] = left[i]
            i += 1
        else:
            sorted_result[k] = right[j]
            count += n_left - i
            j += 1

    return sorted_result, count


class Indices:
    # available keys: antisymmetric, spin-orbital, spin-integrated, spin-adapted
    subclasses = dict()
    subclasses_alias = {'antisymmetric': 'antisymmetric', 'asymm': 'antisymmetric',
                        'spin-orbital': 'spin-orbital', 'so': 'spin-orbital',
                        'spin-integrated': 'spin-integrated', 'si': 'spin-integrated',
                        'spin-adapted': 'spin-adapted', 'sa': 'spin-adapted'}

    @classmethod
    def register_subclass(cls, indices_type):
        def decorator(subclass):
            cls.subclasses[indices_type] = subclass
            return subclass
        return decorator

    @classmethod
    def make_indices(cls, params, indices_type):
        if indices_type not in cls.subclasses_alias:
            raise KeyError(f"Invalid indices type '{indices_type}'. "
                           f"Available: {', '.join(Indices.subclasses_alias.keys())}.")
        return cls.subclasses[cls.subclasses_alias[indices_type]](params)

    def __init__(self, list_of_indices):
        """
        The IndicesBase class to handle a list of Index.
        :param list_of_indices: a list of indices where each index can be casted to Index

        Examples
        --------
        viable formats: 1) ["g0", "g1", ...]
                        2) [Index("g0"), Index("g1"), ...],
                        3) "g0, g1, ..." (separated by comma)
        """
        if not isinstance(list_of_indices, collections.Sequence):
            raise TypeError("Indices only accepts sequence type.")

        if isinstance(list_of_indices, str):
            list_of_indices = [] if len(list_of_indices) == 0 else [i.strip() for i in list_of_indices.split(',')]

        # check if list_of_indices contains any non Index type entries
        indices = []
        for i in list_of_indices:
            if isinstance(i, Index):
                indices.append(i)
            else:
                try:
                    indices.append(Index(i))
                except (ValueError, TypeError):
                    msg = f"Invalid input for Indices initialization: {list_of_indices}\n"
                    msg += "Incoming list of Indices contains improper entries.\n"
                    msg += f"Cannot convert {i} to Index type."
                    raise ValueError(msg)

        # check if list_of_indices contains repeated indices
        indices_set = set(indices)
        size = len(indices)
        if len(indices_set) != size:
            raise ValueError("Indices class does not support repeated indices.")

        self._indices = indices
        self._size = size
        self._indices_set = indices_set

        self._spin_count = [None, None]  # the number of alpha and beta indices
        self._spin_pure = None

    @property
    def indices(self):
        return self._indices

    @property
    def size(self):
        return self._size

    @property
    def indices_set(self):
        return self._indices_set

    def __repr__(self):
        return ", ".join(map(str, self.indices))

    def _is_valid_operand(self, other):
        if self.__class__ is not other.__class__:
            raise TypeError(f"Cannot compare between '{self.__class__.__name__}' and '{other.__class__.__name__}'.")

    @staticmethod
    def _is_valid_operand_weak(other):
        if not isinstance(other, Indices):
            raise TypeError(f"'{other.__class__.__name__}' is not a subclass of 'Indices'.")

    def __eq__(self, other):
        self._is_valid_operand(other)
        return (self.size, self.indices) == (other.size, other.indices)

    def __ne__(self, other):
        self._is_valid_operand(other)
        return (self.size, self.indices) != (other.size, other.indices)

    def __lt__(self, other):
        self._is_valid_operand(other)
        return (self.size, self.indices) < (other.size, other.indices)

    def __le__(self, other):
        self._is_valid_operand(other)
        return (self.size, self.indices) <= (other.size, other.indices)

    def __gt__(self, other):
        self._is_valid_operand(other)
        return (self.size, self.indices) > (other.size, other.indices)

    def __ge__(self, other):
        self._is_valid_operand(other)
        return (self.size, self.indices) >= (other.size, other.indices)

    def __iter__(self):
        self.n = 0
        return self

    def __next__(self):
        if self.n < self.size:
            self.n += 1
            return self.indices[self.n - 1]
        else:
            raise StopIteration

    def __getitem__(self, key):
        # index out-of-range handled by python list
        return self.indices[key]

    def __len__(self):
        return self.size

    def __hash__(self):
        return hash(tuple(self))

    def __contains__(self, value):
        return value in self.indices_set

    def __add__(self, other):
        self._is_valid_operand_weak(other)
        indices_list = self.indices + other.indices
        return self.__class__(indices_list)

    def clone(self):
        """ Clone this object. """
        return self.__class__(self.indices)

    def is_permutation(self, other):
        """
        Test if this and other can be related via a permutation.
        :param other: the other Indices object
        :return: True if there exists a permutation to bring self to other
        """
        self._is_valid_operand_weak(other)
        return self.indices_set == other.indices_set

    def count_permutations(self, other):
        """
        Count the number of permutations needed from self to other.
        :param other: the other Indices object
        :return: the number of inversions
        """

        # check if list1 is a permutation of list2
        if not self.is_permutation(other):
            print("list1: ", str(self))
            print("list2: ", str(other))
            raise ValueError("Two Indices are not in permutations.")

        sequence = {}
        for i, v in enumerate(self):
            sequence[v] = i
        permuted = [0] * other.size
        for i, v in enumerate(other):
            permuted[i] = sequence[v]

        return Permutation(permuted).inversions()

    def latex(self, dollar=False):
        """
        The latex form of this Indices.
        :param dollar: add '$' at both ends for math mode
        :return: a string for the latex form
        """
        out = "{{ {} }}".format(" ".join([i.latex() for i in self]))
        if dollar:
            out = "$" + out + "$"
        return out

    def ambit(self):
        """ Return the ambit form (a string) of this Indices object. """
        return ",".join(map(str, self.indices))

    def count_index_space(self, list_of_space):
        """
        Count the number Index whose MO space lie in the given list.
        :param list_of_space: a list of MO spaces
        :return: the sum of matches
        """
        if not set(list_of_space) <= set(space_priority.keys()):
            print("Given space list:", list_of_space)
            print("Allowed space list:", space_priority.keys())
            raise ValueError("Given list of MO space contains invalid elements.")

        counter = collections.Counter([i.space for i in self.indices])
        return sum([counter[i] for i in list_of_space])

    def canonicalize(self):
        """
        Bring the indices to canonical form.
        :return: the sorted indices and a sign change
        """
        return self.clone(), 1

    def any_overlap(self, other):
        """
        Test if there are any overlapped indices between two Indices
        :param other: the compared Indices object
        :return: True if find any overlap
        """
        self._is_valid_operand(other)
        return not self.indices_set.isdisjoint(other.indices_set)

    def n_multiset_permutation(self, part):
        """
        Compute the number of multiset permutations.
        :param part: a partition of indices, e.g., [[i,j], [k], [l]] for P(ij/k/l)
        :return: the number of multiset permutations
        """
        if not self.exist_permute_format(part):
            return 1
        n = len(part)
        list_index = []
        for i in range(n):
            list_index += [i] * len(part[i])
        return len(list(multiset_permutations(list_index)))

    def exist_permute_format(self, part):
        """
        Return True if there is a non-trivial multiset permutation for antisymmetric indices.
        :param part: a partition of indices, e.g., [[i,j], [k], [l]] for P(ij/k/l)
        :return: always False for non-antisymmetric indices
        """
        return False

    def latex_permute_format(self, part):
        """
        Compute the multiset-permutation form of this Indices object for latex.
        :param part: a partition of indices, e.g., [[i,j], [k], [l]] for P(ij/k/l)
        :return: the number of multiset permutations and a string of permutation for latex
        """
        return 1, ""

    def ambit_permute_format(self, part):
        """
        Generate the multiset-permutation form and the corresponding sign of this Indices object for ambit.
        :param part: a partition of indices, e.g., [[i,j], [k], [l]] for P(ij/k/l)
        :return: sign change, a string of permutation for ambit
        """
        yield 1, ",".join(map(str, self.indices))

    @property
    def spin_count(self):
        raise TypeError("Only available for spin-integrated indices.")

    @property
    def spin_pure(self):
        raise TypeError("Only available for spin-integrated indices.")

    def n_alpha(self):
        return self.spin_count[0]

    def n_beta(self):
        return sum([i.is_beta() for i in self.indices])

    def generate_spin_cases(self, n_beta=None):
        """
        Generate spin-integrated indices from spin-orbital indices.
        :param n_beta: the number of beta indices asked, all possible values by default
        :return: IndicesSpinIntegrated object, e.g., a0,g0 -> a0,g0; a0,G0; A0,g0; A0,G0
        """
        raise TypeError("Only available for spin-orbital indices.")


@Indices.register_subclass('spin-adapted')
class IndicesSpinAdapted(Indices):
    def __init__(self, list_of_indices):
        """
        The spin-adapted indices (cannot change ordering).
        :param list_of_indices: list of indices
        """
        Indices.__init__(self, list_of_indices)


@Indices.register_subclass('antisymmetric')
class IndicesAntisymmetric(Indices):
    def __init__(self, list_of_indices):
        """
        The base class for antisymmetric indices.
        :param list_of_indices: list of indices (see IndicesBase)
        """
        Indices.__init__(self, list_of_indices)

    def canonicalize(self):
        """
        Sort the Indices to canonical form.
        :return: a tuple of (sorted Indices, sign change)
        """
        if self.size <= 1:
            return self.clone(), 1

        list_index, permutation_count = sort_and_count_inversions(self)
        return self.__class__(list_index), (-1) ** permutation_count

    def exist_permute_format(self, part):
        """
        Test if the indices partitioning is non-trivial (i.e., 1).
        :param part: a partition of indices, e.g., [[i,j], [k], [l]] for P(ij/k/l)
        :return: True if the partition of indices yields a multiset permutation greater than one
        """
        if self.size == 0:
            return False
        return len(part) != 1

    def latex_permute_format(self, part):
        """
        Compute the multiset-permutation form of this Indices object for latex.
        :param part: a partition of indices, e.g., [[i,j], [k], [l]] for P(ij/k/l)
        :return: the number of multiset permutations and a string of permutation for latex
        """
        n_perm = self.n_multiset_permutation(part)
        if n_perm == 1:
            return 1, ""
        else:
            perm = ' / '.join([' '.join([i.latex() for i in indices]) for indices in part])
            return n_perm, f"{{\\cal P}}({perm})"

    def ambit_permute_format(self, part):
        """
        Generate the multiset-permutation form and the corresponding sign of this Indices object for ambit.
        :param part: a partition of indices, e.g., [[i,j], [k], [l]] for P(ij/k/l)
        :return: sign change, a string of permutation for ambit
        """
        n = len(part)
        list_index = []
        for i in range(n):
            list_index += [i] * len(part[i])

        for perm in multiset_permutations(list_index):
            next_available = [0] * n
            list_of_indices = []
            for i in perm:
                list_of_indices.append(part[i][next_available[i]])
                next_available[i] += 1
            permuted = self.__class__(list_of_indices)
            yield (-1) ** self.count_permutations(permuted), ",".join(map(str, list_of_indices))


@Indices.register_subclass('spin-orbital')
class IndicesSpinOrbital(IndicesAntisymmetric):
    def __init__(self, list_of_indices):
        """
        Indices represented by spin orbitals.
        :param list_of_indices: list of indices (see IndicesBase)
        """
        IndicesAntisymmetric.__init__(self, list_of_indices)
        for index in self.indices:
            if index.is_beta():
                raise ValueError("Spin-orbital indices should all be in lowercase.")

    def generate_spin_cases(self, n_beta=None):
        """
        Generate spin-integrated indices from spin-orbital indices.
        :param n_beta: the number of beta indices asked, all possible values by default
        :return: IndicesSpinIntegrated object, e.g., a0,g0 -> a0,g0; a0,G0; A0,g0; A0,G0
        """
        if n_beta is None:
            for spins in product(range(2), repeat=self.size):
                indices = list(map(lambda i, s: i if not s else i.to_beta(), self.indices, spins))
                yield IndicesSpinIntegrated(indices)
        else:
            if not isinstance(n_beta, int):
                raise TypeError(f"Invalid n_beta, given '{n_beta.__class__.__name__}', required 'int'.")
            if n_beta > self.size:
                raise ValueError(f"Invalid n_beta value, given '{n_beta}', required 0 <= n_beta <= {self.size}.")

            for beta_indices in combinations(range(self.size), n_beta):
                indices = [i for i in self.indices]
                for idx in beta_indices:
                    indices[idx] = indices[idx].to_beta()
                yield IndicesSpinIntegrated(indices)


@Indices.register_subclass('spin-integrated')
class IndicesSpinIntegrated(IndicesAntisymmetric):
    def __init__(self, list_of_indices):
        """
        Spin integrated indices.
        :param list_of_indices: list of indices (see IndicesBase)
        """
        IndicesAntisymmetric.__init__(self, list_of_indices)
        self._spin_count = [0, 0]
        for index in self.indices:
            self._spin_count[index.is_beta()] += 1
        self._spin_pure = self._spin_count[0] == self.size or self._spin_count[1] == self.size

    @property
    def spin_count(self):
        return self._spin_count

    @property
    def spin_pure(self):
        return self._spin_pure

    def exist_permute_format(self, part):
        """
        Test if the indices partitioning is non-trivial (i.e., 1).
        :param part: a partition of indices, e.g., [[i,j], [k], [l]] for P(ij/k/l)
        :return: True if the partition of indices yields a multiset permutation greater than one
        """
        if not self.spin_pure:
            return False
        else:
            return super().exist_permute_format(part)

    def latex_permute_format(self, part):
        """
        Compute the multiset-permutation form of this Indices object for latex.
        :param part: a partition of indices, e.g., [[i,j], [k], [l]] for P(ij/k/l)
        :return: the number of multiset permutations and a string of permutation for latex
        """
        if not self.spin_pure:
            return 1, ""
        else:
            return super().latex_permute_format(part)

    def ambit_permute_format(self, part):
        """
        Generate the multiset-permutation form and the corresponding sign of this Indices object for ambit.
        :param part: a partition of indices, e.g., [[i,j], [k], [l]] for P(ij/k/l)
        :return: sign change, a string of permutation for ambit
        """
        if not self.spin_pure:
            yield 1, ",".join(map(str, self.indices))
        else:
            yield from super().ambit_permute_format(part)

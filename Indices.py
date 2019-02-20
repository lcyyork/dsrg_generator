import collections
from itertools import product, combinations
from sympy.utilities.iterables import multiset_permutations
from mo_space import space_priority
from Index import Index


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

    @classmethod
    def register_subclass(cls, indices_type):
        def decorator(subclass):
            cls.subclasses[indices_type] = subclass
            return subclass
        return decorator

    @classmethod
    def make_indices(cls, params, indices_type):
        if indices_type not in cls.subclasses:
            raise KeyError(f"Invalid indices type '{indices_type}'. Available: {', '.join(Indices.subclasses.keys())}.")
        return cls.subclasses[indices_type](params)

    def __init__(self, list_of_indices):
        """
        The IndicesBase class to handle a list of Index.
        :param list_of_indices: viable format: ["g0", "g1", ...],
                                               [Index("g0"), Index("g1"), ...],
                                               "g0, g1, ..." (separated by comma)
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
                    print("Incoming list of Indices contains improper entries.")
                    print(f"Cannot convert {i} to Index type.")
                    raise ValueError("Invalid input for Indices initialization.")

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

    def __iadd__(self, other):
        self._is_valid_operand_weak(other)
        if len(self.indices_set.intersection(other.indices_set)) != 0:
            raise ValueError("Two Indices objects contain common Index, thus cannot be added.")
        self._indices += other.indices
        self._size += other.size
        self._indices_set = self.indices_set.union(other.indices_set)
        return self

    def clone(self):
        return self.__class__(self.indices)

    def remove(self, idx):
        """ Return a Indices object without the idx-th element of this Indices object. """
        temp = self.indices[:idx] + self.indices[idx + 1:]
        return self.__class__(temp)

    def is_permutation(self, other):
        """ Return True if there exists a permutation to bring self to other. """
        self._is_valid_operand_weak(other)
        return self.indices_set == other.indices_set

    def count_permutations(self, other):
        """ Count the number of permutations needed from self to other. """

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
        n_inversions = sort_and_count_inversions(permuted)[1]

        return n_inversions

    def latex(self, dollar=False):
        """ Return the latex form (a string) of this Indices object. """
        out = "{{ {} }}".format(" ".join([i.latex() for i in self]))
        if dollar:
            out = "$" + out + "$"
        return out

    def ambit(self):
        """ Return the ambit form (a string) of this Indices object. """
        return ",".join(map(str, self.indices))

    def count_index_space(self, list_of_space):
        """ Count the number Index whose MO space lie in the given list. """
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
        return len(self.indices_set & other.indices_set) != 0

    def n_multiset_permutation(self):
        """ Return the number of multiset permutations of this Indices object. """
        if not self.exist_permute_format():
            return 1
        return len(list(multiset_permutations([index.space for index in self.indices])))

    def exist_permute_format(self):
        """ Return True if there is a valid multiset permutation. """
        return False

    def latex_permute_format(self):
        """
        Compute the multiset-permutation form of this Indices object for latex.
        :return: the number of multiset permutations and a string of permutation for latex
        """
        return 1, ""

    def ambit_permute_format(self):
        """
        Generate the multiset-permutation form and the corresponding sign of this Indices object for ambit.
        :return: sign change, a string of permutation for ambit
        """
        yield 1, ",".join(map(str, self.indices))

    @property
    def spin_count(self):
        raise TypeError("Only available for spin-integrated indices.")

    @property
    def spin_pure(self):
        raise TypeError("Only available for spin-integrated indices.")

    def is_spin_pure(self):
        return self.spin_pure

    def n_alpha(self):
        return self.spin_count[0]

    def n_beta(self):
        return self.spin_count[1]

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

    def exist_permute_format(self):
        """ Return True if there is a valid multiset permutation. """
        if self.size == 0:
            return False
        space = [i.space for i in self.indices]
        return space.count(space[0]) != self.size

    def latex_permute_format(self):
        """
        Compute the multiset-permutation form of this Indices object for latex.
        :return: the number of multiset permutations and a string of permutation for latex
        """
        nperm = self.n_multiset_permutation()
        if nperm == 1:
            return 1, ""
        else:
            indices = sorted(self.indices)
            perm = indices[0].latex()
            for i, index in enumerate(indices[1:], 1):
                if index.space != indices[i - 1].space:
                    perm += ' /'
                perm += ' ' + index.latex()
            return nperm, f"{{\\cal P}}({perm})"

    def ambit_permute_format(self):
        """
        Generate the multiset-permutation form and the corresponding sign of this Indices object for ambit.
        :return: sign change, a string of permutation for ambit
        """
        space_map = collections.defaultdict(list)
        for index in self.indices:
            space_map[index.space].append(index)
        original_ordering = {v: i for i, v in enumerate(self)}
        for space_perm in multiset_permutations([index.space for index in self.indices]):
            permuted = [0] * self.size
            next_available = {space: 0 for space in space_map.keys()}
            indices = []
            for i, space in enumerate(space_perm):
                index = space_map[space][next_available[space]]
                indices.append(index)
                next_available[space] += 1
                permuted[i] = original_ordering[index]
            yield (-1) ** (sort_and_count_inversions(permuted)[1]), ",".join(map(str, indices))


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
                indices = list(map(lambda idx, s: idx if not s else idx.to_beta(), self.indices, spins))
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

    def exist_permute_format(self):
        """ Return True if there is a valid multiset permutation. """
        if not self.spin_pure:
            return False
        else:
            return super().exist_permute_format()

    def latex_permute_format(self):
        """
        Compute the multiset-permutation form of this Indices object for latex.
        :return: the number of multiset permutations and a string of permutation for latex
        """
        if not self.spin_pure:
            return 1, ""
        else:
            return super().latex_permute_format()

    def ambit_permute_format(self):
        """
        Generate the multiset-permutation form and the corresponding sign of this Indices object for ambit.
        :return: sign change, a string of permutation for ambit
        """
        if not self.spin_pure:
            yield 1, ",".join(map(str, self.indices))
        else:
            yield from super().ambit_permute_format()

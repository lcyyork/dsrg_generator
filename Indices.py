import collections
from itertools import product
from sympy.utilities.iterables import multiset_permutations
from mo_space import space_priority
from Index import Index


class Indices:
    # available keys: antisymmetric, spin-orbital, spin-integrated
    subclasses = dict()

    @classmethod
    def register_subclass(cls, indices_type):
        def decorator(subclass):
            cls.subclasses[indices_type] = subclass
            return subclass
        return decorator

    @classmethod
    def make_indices(cls, indices_type, params):
        if indices_type not in cls.subclasses:
            print(f"Warning: {indices_type} not valid. Use Indices base class.")
            return Indices(params)
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
            list_of_indices = [i.strip() for i in list_of_indices.split(',')]

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
        self._set = indices_set

    @property
    def indices(self):
        return self._indices

    @property
    def size(self):
        return self._size

    @property
    def set(self):
        return self._set

    def __repr__(self):
        return ", ".join(map(str, self.indices))

    def __eq__(self, other):
        return self.indices == other.indices

    def __ne__(self, other):
        return not self == other

    def __lt__(self, other):
        if self.size != other.size:
            return True if self.size < other.size else False

        for i, j in zip(self.indices, other.indices):
            if i != j:
                return True if i < j else False
        return False

    def __le__(self, other):
        return self < other or self == other

    def __gt__(self, other):
        return not self <= other

    def __ge__(self, other):
        return not self < other

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
        return value in self.set

    def __add__(self, other):
        indices_list = self.indices + other.indices
        return Indices(indices_list)

    def __iadd__(self, other):
        if len(self.set.intersection(other.set)) != 0:
            raise ValueError("Two Indices objects contain common Index, thus cannot be added.")
        self._indices += other.indices
        self._size += other.size
        self._set = self.set.union(other.set)
        return self

    def clone(self):
        return Indices(self.indices)

    def remove(self, idx):
        """ Return a Indices object without the idx-th element of this Indices object. """
        temp = self.indices[:idx] + self.indices[idx + 1:]
        return Indices(temp)

    def is_permutation(self, other):
        """ Return True if there exists a permutation to bring self to other. """
        return self.set == other.set

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


@Indices.register_subclass('antisymmetric')
class IndicesAntisymmetric(Indices):
    def __init__(self, list_of_indices):
        """
        The base class for antisymmetric indices.
        :param list_of_indices: list of indices (see IndicesBase)
        """
        Indices.__init__(self, list_of_indices)

    @staticmethod
    def sort_and_count_inversions(A):
        """ Sort an array A and count the number of inversions via merge sort. """
        n = len(A)
        if n <= 1:
            return A, 0
        else:
            nhalf = n // 2

            left, x = IndicesAntisymmetric.sort_and_count_inversions(A[:nhalf])
            right, y = IndicesAntisymmetric.sort_and_count_inversions(A[nhalf:])
            result, z = IndicesAntisymmetric.merge_and_count_split_inversion(left, right)

            return result, x + y + z

    @staticmethod
    def merge_and_count_split_inversion(left, right):
        """ Merge two lists (left, right) into sorted_result and count inversions. """
        i, j = 0, 0
        count = 0
        nleft = len(left)
        nright = len(right)
        n = nleft + nright
        sorted_result = [0] * n

        for k in range(n):
            if i == nleft:
                sorted_result[k] = right[j]
                j += 1
                continue

            if j == nright:
                sorted_result[k] = left[i]
                i += 1
                continue

            if left[i] <= right[j]:
                sorted_result[k] = left[i]
                i += 1
            else:
                sorted_result[k] = right[j]
                count += nleft - i
                j += 1

        return sorted_result, count

    def count_permutations(self, other):
        """ Count the number of permutations needed from self to other. """

        # check if list1 is a permutation of list2
        if not self.is_permutation(other):
            print("list1: ", str(self))
            print("list2: ", str(other))
            raise ValueError("Two Indices are not permutational.")

        sequence = {}
        for i, v in enumerate(self):
            sequence[v] = i
        permuted = [0] * other.size
        for i, v in enumerate(other):
            permuted[i] = sequence[v]
        n_inversions = self.sort_and_count_inversions(permuted)[1]

        return n_inversions

    def canonicalize(self):
        """ Sort this Indices and return the sorted Indices with the proper sign change. """
        if self.size <= 1:
            return self.clone(), 1

        list_index, permutation_count = self.sort_and_count_inversions(self)
        return Indices(list_index), (-1) ** permutation_count

    def n_multiset_permutation(self):
        """ Return the number of multiset permutations of this Indices object. """
        return len(list(multiset_permutations([index.space for index in self.indices])))


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
        :yield: sign change, a string of permutation for ambit
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
            yield (-1) ** (self.sort_and_count_inversions(permuted)[1]), ",".join(map(str, indices))

    def generate_spin_cases(self):
        """
        Generate spin-integrated indices from spin-orbital indices.
        :yield: IndicesSpinIntegrated object, e.g., a0,g0 -> a0,g0; a0,G0; A0,g0; A0,G0
        """
        for spins in product(range(2), repeat=self.size):
            indices = list(map(lambda i, s: i if not s else i.to_beta(), self.indices, spins))
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

    def latex_permute_format(self):
        """
        Compute the multiset-permutation form of this Indices object for latex.
        :return: the number of multiset permutations and a string of permutation for latex
        """
        nperm = self.n_multiset_permutation()
        if nperm == 1 or (not self.spin_pure):
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
        :yield: sign change, a string of permutation for ambit
        """
        if not self.spin_pure:
            yield 1, ",".join(map(str, self.indices))
            return

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
            yield (-1) ** (self.sort_and_count_inversions(permuted)[1]), ",".join(map(str, indices))

    def is_spin_pure(self):
        return self.spin_pure

    def n_alpha(self):
        return self.spin_count[0]

    def n_beta(self):
        return self.spin_count[1]

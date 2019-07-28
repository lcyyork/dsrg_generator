from sympy.combinatorics.tensor_can import get_symmetric_group_sgs, bsgs_direct_product, get_minimal_bsgs
from sympy.combinatorics import Permutation
from sympy.combinatorics.perm_groups import PermutationGroup

from src.Indices import Indices, IndicesSpinOrbital, IndicesAntisymmetric


# TODO: delete this
def make_indices_pair(upper_indices, lower_indices, indices_type=""):
        """
        Initialize a IndicesPair object from upper and lower indices.
        :param upper_indices: a list of Index or string for upper indices
        :param lower_indices: a list of Index or string for lower indices
        :param indices_type: the type of indices, used if the indices are not Indices
        :return: a IndicesPair object
        """
        def test_indices_type():
            if indices_type not in Indices.subclasses_alias:
                raise KeyError(f"Invalid indices type {indices_type}."
                               f" Choices: {', '.join(Indices.subclasses.keys())}.")

        if isinstance(upper_indices, Indices):
            upper = upper_indices
        else:
            test_indices_type()
            upper = Indices.make_indices(upper_indices, indices_type)

        if isinstance(lower_indices, Indices):
            lower = lower_indices
        else:
            test_indices_type()
            lower = Indices.make_indices(lower_indices, indices_type)

        return IndicesPair(upper, lower)


class IndicesPair:
    def __init__(self, upper_indices, lower_indices, indices_type='so'):
        """
        The IndicesPair class to handle upper and lower indices for tensors or second-quantized operators.
        :param upper_indices: a Indices object for upper indices
        :param lower_indices: a Indices object for lower indices
        :param indices_type: the type of indices, used if the indices are not Indices
        """
        upper = upper_indices if isinstance(upper_indices, Indices) else \
            Indices.make_indices(upper_indices, indices_type)
        lower = lower_indices if isinstance(lower_indices, Indices) else \
            Indices.make_indices(lower_indices, indices_type)

        if type(upper_indices) != type(lower_indices):
            raise TypeError(f"Inconsistent type for upper_indices ('{upper_indices.__class__.__name__}') "
                            f"and lower_indices ('{lower_indices.__class__.__name__}').")

        self._upper_indices = upper
        self._lower_indices = lower
        self._indices_type = Indices.subclasses_alias[indices_type]

    @classmethod
    def from_indices_pair(cls, other):
        """ Make a copy from a IndicesPair. """
        IndicesPair._is_valid_operand(other)
        return cls(other.upper_indices, other.lower_indices, other.indices_type)

    @property
    def upper_indices(self):
        return self._upper_indices

    @property
    def lower_indices(self):
        return self._lower_indices

    @property
    def n_upper(self):
        return self.upper_indices.size

    @property
    def n_lower(self):
        return self.lower_indices.size

    @property
    def size(self):
        return self.n_upper + self.n_lower

    @property
    def n_body(self):
        if self.n_lower != self.n_upper:
            raise ValueError(f"Invalid quest because n_lower ({self.n_lower}) != n_upper ({self.n_upper}).")
        return self.n_lower

    @property
    def indices(self, upper_first=True):
        if upper_first:
            return self.upper_indices.indices + self.lower_indices.indices
        else:
            return self.lower_indices.indices + self.upper_indices.indices

    @property
    def indices_type(self):
        return self._indices_type

    # TODO: delete this
    @property
    def type_of_indices(self):
        return self.upper_indices.__class__

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

    def ambit(self, upper_first=True):
        """
        The ambit form of IndicesPair
        :param upper_first: True if put upper indices in front of lower indices
        :return: a string in ambit format
        """
        if self.is_empty():
            return ""
        if not upper_first:
            return f'["{self.lower_indices.ambit()},{self.upper_indices.ambit()}"]'
        return f'["{self.upper_indices.ambit()},{self.lower_indices.ambit()}"]'

    def generate_spin_cases(self, particle_conserving=True):
        """
        Generate spin-integrated indices pair from spin-orbital indices.
        :param particle_conserving: True if generated indices preserve the spin
        :return: Spin-integrated indices pair
        """
        if not isinstance(self.upper_indices, IndicesSpinOrbital):
            raise TypeError("Only available for spin-orbital indices.")

        if particle_conserving and (self.n_upper != self.n_lower):
            raise ValueError("Invalid option. Number of particles cannot be preserved.")

        if particle_conserving:
            for upper_indices in self.upper_indices.generate_spin_cases():
                n_beta = upper_indices.n_beta()
                for lower_indices in self.lower_indices.generate_spin_cases(n_beta):
                    yield IndicesPair(upper_indices, lower_indices)
        else:
            for upper_indices in self.upper_indices.generate_spin_cases():
                for lower_indices in self.lower_indices.generate_spin_cases():
                    yield IndicesPair(upper_indices, lower_indices)

    def canonicalize(self):
        """
        Sort the IndicesPair to canonical order.
        :return: a tuple of (sorted IndicesPair, sign change)
        """
        upper_indices, upper_sign = self.upper_indices.canonicalize()
        lower_indices, lower_sign = self.lower_indices.canonicalize()
        return IndicesPair(upper_indices, lower_indices), upper_sign * lower_sign

    def void_indices_pair(self):
        """ Return an empty IndicesPair. """
        return IndicesPair('', '', self.indices_type)

    def base_strong_generating_set(self, hermitian):
        """
        Return minimal base and strong generating set of this IndicesPair.
        :param hermitian: upper and lower indices can be swapped if True
        :return: a tuple of (base, strong generating set)
        """
        if not isinstance(self.upper_indices, IndicesAntisymmetric):
            return self.sym_bsgs(hermitian)
        else:
            return self.asym_bsgs(hermitian)

    def sym_bsgs(self, hermitian):
        """
        Return minimal base and strong generating set for non-antisymmetric indices.
        :param hermitian: upper and lower indices can be swapped if True
        :return: a tuple of (base, strong generating set)
        """
        if self.n_lower != self.n_upper:
            raise ValueError(f"Invalid: n_lower ({self.n_lower}) != n_upper ({self.n_upper}).")

        n_body = self.n_upper
        cre = list(range(n_body))
        ann = list(range(n_body, 2 * n_body))

        perms = []
        for i, j in zip(cre[:-1], cre[1:]):
            perms.append(Permutation(2 * n_body + 1)(i, j)(i + n_body, j + n_body))

        p = list(range(2 * n_body + 2))
        if hermitian:
            for i, j in zip(cre, ann):
                p[i] = j
                p[j] = i
        perms.append(Permutation(p))

        sym = PermutationGroup(*perms)
        sym.schreier_sims()
        return get_minimal_bsgs(sym.base, sym.strong_gens)

    def asym_bsgs(self, hermitian):
        """
        Return minimal base and strong generating set for antisymmetric indices.
        :param hermitian: upper and lower indices can be swapped if True
        :return: a tuple of (base, strong generating set)
        """
        if not hermitian:
            u_base, u_gens = get_symmetric_group_sgs(self.n_upper, 1)
            l_base, l_gens = get_symmetric_group_sgs(self.n_lower, 1)
            return bsgs_direct_product(u_base, u_gens, l_base, l_gens)

        if self.n_upper != self.n_lower:
            raise ValueError(f"{self} cannot be Hermitian.")

        upper = list(range(self.n_upper))
        lower = list(range(self.n_upper, self.size))
        sign = [self.size, self.size + 1]

        perms = [Permutation(i, j)(sign[0], sign[1]) for i, j in zip(upper[:-1], upper[1:])]
        perms += [Permutation(i, j)(sign[0], sign[1]) for i, j in zip(lower[:-1], lower[1:])]

        p = list(range(self.size + 2))
        for i, j in zip(upper, lower):
            p[i] = j
            p[j] = i
        perms.append(Permutation(p))

        asymmetric = PermutationGroup(*perms)
        asymmetric.schreier_sims()
        return get_minimal_bsgs(asymmetric.base, asymmetric.strong_gens)
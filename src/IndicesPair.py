from sympy.combinatorics.tensor_can import get_symmetric_group_sgs, bsgs_direct_product, get_minimal_bsgs
from sympy.combinatorics import Permutation
from sympy.combinatorics.perm_groups import PermutationGroup

from src.Indices import Indices, IndicesSpinOrbital, IndicesAntisymmetric


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
        return self.upper_indices.__class__

    def is_empty(self):
        return self.size == 0

    def clone(self):
        """ Make a copy. """
        return self.__class__(self.upper_indices, self.lower_indices)

    def _is_valid_operand_indices(self, other):
        if self.indices_type != other.indices_type:
            raise TypeError(f"Cannot compare between indices pairs due to different indices types: "
                            f"{self.indices_type.__name__} vs {other.indices_type.__name__}")

    def _is_valid_operand(self, other):
        if not isinstance(other, IndicesPair):
            raise TypeError(f"Cannot compare between 'IndicesPair' and '{other.__class__.__name__}'.")
        self._is_valid_operand_indices(other)

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

    def __hash__(self):
        return hash((self.upper_indices, self.lower_indices))

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

    def diagonal_indices(self):
        """
        Test if upper and lower indices contain common elements.
        :return: a set of common indices
        """
        return self.upper_indices.indices_set & self.lower_indices.indices_set

    def generate_spin_cases(self, particle_conserving=True):
        """
        Generate spin-integrated indices pair from spin-orbital indices.
        :param particle_conserving: True if generated indices preserve the spin
        :return: spin-integrated IndicesPair
        """
        for upper, lower in self.generate_spin_cases_indices(particle_conserving):
            yield IndicesPair(upper, lower, 'si')

    def generate_spin_cases_indices(self, particle_conserving):
        """
        Generate spin-integrated indices pair from spin-orbital indices.
        :param particle_conserving: True if generated indices preserve the spin
        :return: a tuple of spin-integrated upper and lower indices
        """
        if not isinstance(self.upper_indices, IndicesSpinOrbital):
            raise TypeError("Only available for spin-orbital indices.")

        if particle_conserving and (self.n_upper != self.n_lower):
            raise ValueError("Invalid option. Number of particles cannot be preserved.")

        if particle_conserving:
            for upper_indices in self.upper_indices.generate_spin_cases():
                n_beta = upper_indices.n_beta()
                for lower_indices in self.lower_indices.generate_spin_cases(n_beta):
                    yield upper_indices, lower_indices
        else:
            for upper_indices in self.upper_indices.generate_spin_cases():
                for lower_indices in self.lower_indices.generate_spin_cases():
                    yield upper_indices, lower_indices

    def is_spin_conserving(self):
        """ Return True if spin Ms is preserved. """
        if self.n_upper == self.n_lower:
            return self.upper_indices.n_beta() == self.lower_indices.n_beta()
        raise ValueError(f"Invalid quest, n_upper ({self.n_upper}) != n_lower ({self.n_lower}).")

    def canonicalize(self):
        """
        Sort the IndicesPair to canonical order.
        :return: a tuple of (sorted IndicesPair, sign change)
        """
        upper_indices, lower_indices, sign = self.canonicalize_indices()
        return IndicesPair(upper_indices, lower_indices), sign

    def canonicalize_indices(self):
        """
        Sort the upper and lower indices to canonical order.
        :return: a tuple of (sorted upper indices, sorted lower indices, sign change)
        """
        upper_indices, upper_sign = self.upper_indices.canonicalize()
        lower_indices, lower_sign = self.lower_indices.canonicalize()
        return upper_indices, lower_indices, upper_sign * lower_sign

    def base_strong_generating_set(self, hermitian=True):
        """
        Return minimal base and strong generating set of this IndicesPair.
        :param hermitian: upper and lower indices can be swapped if True
        :return: a tuple of (base, strong generating set)
        """
        if self.size == 0:
            raise ValueError("Cannot perform BSGS on empty indices pair.")

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

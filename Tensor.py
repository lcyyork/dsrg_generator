from Indices import Indices


class Tensor:
    """ The antisymmetric many-body Tensor class.
    capability: a single tensor term with lower and upper indices """

    def __init__(self, name, upper_indices, lower_indices, priority=0):
        """
        The antisymmetric many-body Tensor class.
        :param name: tensor name
        :param upper_indices: upper indices of the tensor, a list of string/Index
        :param lower_indices: lower indices of the tensor, a list of string/Index
        :param priority: tensor priority, smaller value has higher priority
        """
        if not isinstance(name, str):
            raise ValueError("Tensor name has to be string type.")
        self.name = name

        self.nbody = len(upper_indices)
        if self.nbody != len(lower_indices):
            raise ValueError("Cannot decide Tensor nbody. Upper and lower indices are of different sizes.")

        if isinstance(upper_indices, Indices):
            self.Uindices = upper_indices
        else:
            self.Uindices = Indices(upper_indices)

        if isinstance(lower_indices, Indices):
            self.Lindices = lower_indices
        else:
            self.Lindices = Indices(lower_indices)

        self.priority = priority

        return

    def __eq__(self, other):
        if self.priority != other.priority:
            return False
        if self.nbody != other.nbody:
            return False
        if self.name != other.name:
            return False
        if self.Uindices != other.Uindices:
            return False
        if self.Lindices != other.Lindices:
            return False
        return True

    def __ne__(self, other):
        return not self == other

    def is_permutation(self, other):
        """ Return True if two tensors differ in permutations. """
        if self.priority != other.priority:
            return False
        if self.nbody != other.nbody:
            return False
        if self.name != other.name:
            return False
        return self.Uindices.is_permutation(other.Uindices) and self.Lindices.is_permutation(other.Lindices)

    def __lt__(self, other):
        if self.priority != other.priority:
            return True if self.priority < other.priority else False
        if self.nbody != other.nbody:
            return True if self.nbody < other.nbody else False
        if self.name != other.name:
            return True if self.name < other.name else False
        if self.Uindices != other.Uindices:
            return True if self.Uindices < other.Uindices else False
        if self.Lindices != other.Lindices:
            return True if self.Lindices < other.Lindices else False
        return False

    def __le__(self, other):
        return self < other or self == other

    def __gt__(self, other):
        return not self <= other

    def __ge__(self, other):
        return not self < other

    def __repr__(self):
        return self.latex()

    def latex(self, dollar=False):
        out = "{}^{}_{}".format(self.name, self.Uindices.latex(), self.Lindices.latex())
        if dollar:
            out = "$ " + out + " $"
        return out

    def ambit(self):
        return '{}{}["{},{}"]'.format(self.name, self.nbody, self.Uindices.ambit(), self.Lindices.ambit())

    def canonicalize_inplace(self):
        self.Uindices, sign0 = self.Uindices.canonicalize()
        self.Lindices, sign1 = self.Lindices.canonicalize()
        return sign0 * sign1

    def canonicalize_copy(self):
        Uindices, sign0 = self.Uindices.canonicalize()
        Lindices, sign1 = self.Lindices.canonicalize()
        return Tensor(self.name, Uindices, Lindices, self.priority), sign0 * sign1


class Cumulant(Tensor):
    def __init__(self, upper_indices, lower_indices):
        Tensor.__init__(self, "L", upper_indices, lower_indices, priority=3)
        return


class HoleDensity(Tensor):
    def __init__(self, upper_indices, lower_indices):
        if len(upper_indices) != 1 or len(lower_indices) != 1:
            raise ValueError("Hole density should be of 1 body.")
        Tensor.__init__(self, "C", upper_indices, lower_indices, priority=3)
        return


class Kronecker(Tensor):
    def __init__(self, upper_indices, lower_indices):
        if len(upper_indices) != 1 or len(lower_indices) != 1:
            raise ValueError("Kronecker delta should be of 1 body.")
        Tensor.__init__(self, "K", upper_indices, lower_indices, priority=2)
        return


class ClusterAmp(Tensor):
    def __init__(self, upper_indices, lower_indices):
        Tensor.__init__(self, "T", upper_indices, lower_indices, priority=1)
        return


class Hamiltonian(Tensor):
    def __init__(self, upper_indices, lower_indices):
        Tensor.__init__(self, "H", upper_indices, lower_indices, priority=0)
        return


def test_tensor_class():
    tensor = Tensor("T", ['a1', 'a2'], ['c3', 'p0'])
    assert str(tensor) == "T^{ a_{1} a_{2} }_{ c_{3} p_{0} }", "Tensor format failed."
    assert tensor.ambit() == 'T2["a1,a2,c3,p0"]', "Tensor format ambit failed."
    tensor_c, tensor_sign = tensor.canonicalize_copy()
    assert tensor_c == Tensor("T", ['a1', 'a2'], ['p0', 'c3']), "Tensor canonicalize failed."
    assert tensor_sign == -1, "Tensor canonicalize sign failed."
    assert tensor_c <= tensor, "Tensor comparison <= failed."
    print("Tensor tests passed.")

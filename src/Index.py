from src.mo_space import space_priority


class Index:
    def __init__(self, name):
        """
        The Index class to handle MO space and label
        :param name: the name of index, e.g., p0, a1
        """

        check_str = isinstance(name, str)
        if not check_str:
            raise TypeError("Index name has to be string type.")

        s = name[0]
        n = name[1:]
        if s not in space_priority:
            msg = "Improper Index name:\n"
            msg += f"{name} is not in the available MO spaces (g, p, v, h, c, a)."
            raise ValueError(msg)
        if not n.isdigit():
            msg = "Improper Index name:\n"
            msg += f"{name} does not have an integer after the space label."
            raise ValueError(msg)

        self._name = name
        self._space = s
        self._number = int(n)
        return

    @property
    def name(self):
        return self._name

    @property
    def space(self):
        return self._space

    @property
    def number(self):
        return self._number

    def __repr__(self):
        return self.name

    @staticmethod
    def _is_valid_operand(other):
        if not isinstance(other, Index):
            raise TypeError(f"Cannot compare between 'Index' and '{type(other).__name__}'.")

    def __eq__(self, other):
        self._is_valid_operand(other)
        return (self.space, self.number) == (other.space, other.number)

    def __ne__(self, other):
        self._is_valid_operand(other)
        return (self.space, self.number) != (other.space, other.number)

    def __lt__(self, other):
        self._is_valid_operand(other)
        return (space_priority[self.space], self.number) < (space_priority[other.space], other.number)

    def __le__(self, other):
        self._is_valid_operand(other)
        return (space_priority[self.space], self.number) <= (space_priority[other.space], other.number)

    def __gt__(self, other):
        self._is_valid_operand(other)
        return (space_priority[self.space], self.number) > (space_priority[other.space], other.number)

    def __ge__(self, other):
        self._is_valid_operand(other)
        return (space_priority[self.space], self.number) >= (space_priority[other.space], other.number)

    def __hash__(self):
        return hash(self.name)

    def latex(self, dollar=False):
        out = f"{self.space}_{{{self.number}}}"
        if dollar:
            out = "$" + out + "$"
        return out

    def is_beta(self):
        return self.space.isupper()

    def to_beta(self):
        return Index(self.name.upper())

    def to_alpha(self):
        return Index(self.name.lower())

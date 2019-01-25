from mo_space import space_priority


class Index:
    def __init__(self, name):
        """
        The Index class to handle MO space and label
        :param name: the name of index, e.g., p0, a1
        """

        check_str = isinstance(name, str)
        if not check_str:
            raise ValueError("Index name has to be string type.")

        s = name[0]
        n = name[1:]
        if s not in space_priority:
            print("{0} is not in the available MO spaces (g, p, v, h, c, a).".format(name))
            raise ValueError("Improper Index name.")
        if not n.isdigit():
            print("{0} does not have an integer after the space label.".format(name))
            raise ValueError("Improper Index name.")

        self.name = name
        self.space = s
        self.number = int(n)
        return

    def __repr__(self):
        return self.name

    def __eq__(self, other):
        return self.name == other.name

    def __ne__(self, other):
        return not self == other

    def __lt__(self, other):
        if self.space != other.space:
            return space_priority[self.space] < space_priority[other.space]
        else:
            return self.number < other.number

    def __le__(self, other):
        if self.space != other.space:
            return space_priority[self.space] < space_priority[other.space]
        else:
            return self.number <= other.number

    def __gt__(self, other):
        return not self <= other

    def __ge__(self, other):
        return not self < other

    def __hash__(self):
        return hash(self.name)

    def latex(self, dollar=False):
        out = "{0}_{{{1}}}".format(self.space, self.number)
        if dollar:
            out = "$" + out + "$"
        return out

from mo_space import space_priority


class SpaceCounter:
    def __init__(self):
        """
        The space counter to count the MO spaces in a given indices.
        """
        self.upper = [0] * len(space_priority)
        self.lower = [0] * len(space_priority)
        self.nupper = 0
        self.nlower = 0

    def add(self, indices_set, is_upper):
        """
        Add the MO spaces of indices to self.
        :param indices_set: a set of Index objects
        :param is_upper: add to self.upper/self.lower if True/False
        """
        if is_upper:
            for index in indices_set:
                self.upper[space_priority.index(index.space)] += 1
                self.nupper += 1
        else:
            for index in indices_set:
                self.lower[space_priority.index(index.space)] += 1
                self.nlower += 1

    def __eq__(self, other):
        return self.upper == other.upper and self.lower == other.lower

    def __ne__(self, other):
        return self.upper != other.upper or self.lower != other.lower

    def __lt__(self, other):
        total0 = self.nupper + self.nlower
        total1 = other.nupper + other.nlower
        if total0 != total1:
            return total0 < total1
        if self.upper != other.upper:
            return self.upper < other.upper
        if self.lower != other.lower:
            return self.lower < other.lower
        return False

    def __le__(self, other):
        return self == other or self < other

    def __gt__(self, other):
        total0 = self.nupper + self.nlower
        total1 = other.nupper + other.nlower
        if total0 != total1:
            return total0 > total1
        if self.upper != other.upper:
            return self.upper > other.upper
        if self.lower != other.lower:
            return self.lower > other.lower
        return False

    def __ge__(self, other):
        return not self < other

    def __repr__(self):
        return "SpaceCounter upper {} lower {}".format(self.upper, self.lower)

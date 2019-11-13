# spin-orbital space
so_space = ['g', 'p', 'h', 'v', 'c', 'a']
space_priority_so = {k: v for v, k in enumerate(so_space)}
space_relation_so = {'c': {'c'}, 'a': {'a'}, 'v': {'v'},
                     'p': {'a', 'v'}, 'h': {'c', 'a'},
                     'g': {'c', 'a', 'v'}
                     }

# spin-integrated alpha subspace
space_priority_a = space_priority_so
space_relation_a = space_relation_so

# spin-integrated beta subspace
mo_space_b = [i.upper() for i in so_space]
space_priority_b = {k: v for v, k in enumerate(mo_space_b)}
space_relation_b = {'C': {'C'}, 'A': {'A'}, 'V': {'V'},
                    'P': {'A', 'V'}, 'H': {'C', 'A'},
                    'G': {'C', 'A', 'V'}
                    }

# spin-integrated space
mo_space = so_space + mo_space_b
space_priority = {k: v for v, k in enumerate(mo_space)}
space_relation = {**space_relation_a, **space_relation_b}


def find_space_label(space_set):
    """
    Find the space label for the input set of space labels.
    :param space_set: a set of space labels, e.g., {'a', 'v'}
    :return: the composite space label, e.g., 'p' for {'a', 'v'}
    """
    for i in space_set:
        if i not in space_relation:
            raise ValueError(f"Invalid space set ({space_set}): {i} not in {space_relation}")

    size = len(space_set)

    if size == 0:
        raise ValueError(f"Invalid space set: empty")

    is_beta = next(iter(space_set)).isupper()

    if size == 1:
        return next(iter(space_set))
    elif size == 2:
        if 'v' in space_set or 'V' in space_set:
            return 'P' if is_beta else 'p'
        if 'c' in space_set or 'C' in space_set:
            return 'H' if is_beta else 'h'
    else:
        return 'G' if is_beta else 'g'


class MOSpaceCounter:
    """
    The MOSpaceCounter class.
    The main purpose of this is to simplify comparisons between computational cost of a Term.
    For example, one term scales as V^4 O^2, which is more expensive than another of V^5.
    """

    def __init__(self, input_counter):
        """
        Constructor of MOSpaceCounter object.
        :param input_counter: a Counter or a dictionary of format {'c': nc, 'v': nv, 'a': na, 'C': nC, 'h': nh}
        """
        if len(set(i.lower() for i in input_counter.keys()) - set(so_space)) != 0:
            raise ValueError(f"Invalid space label in {input_counter}.")

        idict = {'c': 0, 'a': 0, 'v': 0}
        for k, n in input_counter.items():
            k = k.lower()
            if k in idict:
                idict[k] += n
            else:
                if k == 'h':
                    idict['c'] += n
                else:
                    idict['v'] += n

        self._input_counter = {k.lower(): v for k, v in input_counter.items()}

        self._parsed_counter = [idict['v'], idict['c'], idict['a']]
        self._overall_scaling = sum(self._parsed_counter)

    @property
    def parsed_counter(self):
        return self._parsed_counter

    @property
    def overall_scaling(self):
        return self._overall_scaling

    @property
    def raw_counter(self):
        return self._input_counter

    def __repr__(self):
        out = " ".join([f"{i}^{self._input_counter[i]}"
                        for i in sorted(self._input_counter.keys(), key=lambda x: space_priority[x])])
        return out

    @staticmethod
    def _is_valid_operand(other):
        if not isinstance(other, MOSpaceCounter):
            raise TypeError(f"Cannot compare between 'MOSpaceCounter' and '{type(other).__name__}'.")

    def __eq__(self, other):
        self._is_valid_operand(other)
        return self.raw_counter == other.raw_counter

    def __ne__(self, other):
        self._is_valid_operand(other)
        return self.raw_counter != other.raw_counter

    def __lt__(self, other):
        self._is_valid_operand(other)
        return (self.overall_scaling, self.parsed_counter) < (other.overall_scaling, other.parsed_counter)

    def __le__(self, other):
        self._is_valid_operand(other)
        return (self.overall_scaling, self.parsed_counter) <= (other.overall_scaling, other.parsed_counter)

    def __gt__(self, other):
        self._is_valid_operand(other)
        return (self.overall_scaling, self.parsed_counter) > (other.overall_scaling, other.parsed_counter)

    def __ge__(self, other):
        self._is_valid_operand(other)
        return (self.overall_scaling, self.parsed_counter) >= (other.overall_scaling, other.parsed_counter)

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

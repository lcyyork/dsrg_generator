def integer_partition(n):
    """
    Partition an integer n to small components using ZS1 algorithm.
    see Intern. J. Computer Math., Vol. 70. pp. 319 by A. Zoghbiu and I. Stojmenovic

    other implementations see
    https://stackoverflow.com/questions/18503096/python-integer-partitioning-with-given-k-partitions
    """
    x = [1] * n
    x[0] = n
    m, h = 1, 1
    while x[0] != 1:
        if x[0] == n:
            yield x[:1]

        if x[h - 1] == 2:
            m += 1
            x[h - 1] = 1
            h -= 1
        else:
            r = x[h - 1] - 1
            t = m - h + 1
            x[h - 1] = r
            while t >= r:
                h += 1
                x[h - 1] = r
                t -= r
            if t == 0:
                m = h
            else:
                m = h + 1
                if t > 1:
                    h += 1
                    x[h - 1] = t
        yield x[:m]


def integer_separation(integer, include_zero=False):
    """ Separate an integer to two small numbers. """
    out = []
    start = 0 if include_zero else 1
    end = 1 if include_zero else 0
    for i, j in zip(range(start, integer + end), range(integer - start, 0 - end, -1)):
        out.append((i, j))
    return out

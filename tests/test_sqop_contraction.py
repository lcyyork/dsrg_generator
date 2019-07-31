from itertools import combinations

from src.Indices import Indices
from src.SQOperator import SecondQuantizedOperator
from src.Tensor import Tensor
from src.sqop_contraction import expand_hole_densities
from src.sqop_contraction import compute_elementary_contractions_list, compute_elementary_contractions_categorized
from src.sqop_contraction import compute_operator_contractions, compute_operator_contractions_general


SQ = SecondQuantizedOperator
make_tensor = Tensor.make_tensor


def test_ele_con_list_1():
    h = SQ("g0", "g1", 'spin-orbital')
    t = SQ("p0", "h0", 'spin-orbital')
    ref = {make_tensor('cumulant', "g0", "h0"),
           make_tensor('eta', "p0", "g1"),
           make_tensor('lambda', "g0, p0", "g1, h0")}
    elementary_contractions = set(compute_elementary_contractions_list([h, t]))
    assert elementary_contractions == ref

    h = SQ("c0", "g1", 'spin-orbital')
    ref = {make_tensor('cumulant', "c0", "h0"),
           make_tensor('eta', "p0", "g1")}
    elementary_contractions = set(compute_elementary_contractions_list([h, t]))
    assert elementary_contractions == ref

    h = SQ("v0", "g1", 'spin-orbital')
    ref = {make_tensor('eta', "p0", "g1")}
    elementary_contractions = set(compute_elementary_contractions_list([h, t]))
    assert elementary_contractions == ref


def test_ele_con_list_2():
    h = SQ("g0, g1", "g2, g3", 'spin-orbital')
    t = SQ("p0, p1", "h0, h1", 'spin-orbital')
    elementary_contractions = set(compute_elementary_contractions_list([h, t], max_cu=4))

    ref = {make_tensor('L', 'g0', 'h0'), make_tensor('L', 'g0', 'h1'),
           make_tensor('L', 'g1', 'h0'), make_tensor('L', 'g1', 'h1'),
           make_tensor('C', 'p0', 'g2'), make_tensor('C', 'p0', 'g3'),
           make_tensor('C', 'p1', 'g2'), make_tensor('C', 'p1', 'g3'),
           make_tensor('L', "g0, g1", "h0, h1"), make_tensor('L', "p0, p1", "g2, g3"),
           make_tensor('L', "g0, g1", "g2, h0"), make_tensor('L', "g0, g1", "g2, h1"),
           make_tensor('L', "g0, g1", "g3, h0"), make_tensor('L', "g0, g1", "g3, h1"),
           make_tensor('L', "p0, p1", "g2, h0"), make_tensor('L', "p0, p1", "g2, h1"),
           make_tensor('L', "p0, p1", "g3, h0"), make_tensor('L', "p0, p1", "g3, h1"),
           make_tensor('L', "g0, p0", "g2, g3"), make_tensor('L', "g0, p1", "g2, g3"),
           make_tensor('L', "g1, p0", "g2, g3"), make_tensor('L', "g1, p1", "g2, g3"),
           make_tensor('L', "g0, p0", "h0, h1"), make_tensor('L', "g0, p1", "h0, h1"),
           make_tensor('L', "g1, p0", "h0, h1"), make_tensor('L', "g1, p1", "h0, h1"),
           make_tensor('L', "g0, p0", "g2, h0"), make_tensor('L', "g0, p0", "g2, h1"),
           make_tensor('L', "g0, p0", "g3, h0"), make_tensor('L', "g0, p0", "g3, h1"),
           make_tensor('L', "g0, p1", "g2, h0"), make_tensor('L', "g0, p1", "g2, h1"),
           make_tensor('L', "g0, p1", "g3, h0"), make_tensor('L', "g0, p1", "g3, h1"),
           make_tensor('L', "g1, p0", "g2, h0"), make_tensor('L', "g1, p0", "g2, h1"),
           make_tensor('L', "g1, p0", "g3, h0"), make_tensor('L', "g1, p0", "g3, h1"),
           make_tensor('L', "g1, p1", "g2, h0"), make_tensor('L', "g1, p1", "g2, h1"),
           make_tensor('L', "g1, p1", "g3, h0"), make_tensor('L', "g1, p1", "g3, h1"),
           make_tensor('L', 'g0, p0, p1', 'g2, h0, h1'), make_tensor('L', 'g0, p0, p1', 'g3, h0, h1'),
           make_tensor('L', 'g1, p0, p1', 'g2, h0, h1'), make_tensor('L', 'g1, p0, p1', 'g3, h0, h1'),
           make_tensor('L', 'g0, p0, p1', 'g2, g3, h0'), make_tensor('L', 'g0, p0, p1', 'g2, g3, h1'),
           make_tensor('L', 'g1, p0, p1', 'g2, g3, h0'), make_tensor('L', 'g1, p0, p1', 'g2, g3, h1'),
           make_tensor('L', 'g0, g1, p0', 'g2, h0, h1'), make_tensor('L', 'g0, g1, p0', 'g3, h0, h1'),
           make_tensor('L', 'g0, g1, p1', 'g2, h0, h1'), make_tensor('L', 'g0, g1, p1', 'g3, h0, h1'),
           make_tensor('L', 'g0, g1, p0', 'g2, g3, h0'), make_tensor('L', 'g0, g1, p0', 'g2, g3, h1'),
           make_tensor('L', 'g0, g1, p1', 'g2, g3, h0'), make_tensor('L', 'g0, g1, p1', 'g2, g3, h1'),
           make_tensor('L', 'g0, g1, p0, p1', 'g2, g3, h0, h1')}

    assert elementary_contractions == ref


def test_ele_con_list_3():
    h = SQ("g0, h0", "g1, p0", 'spin-orbital')
    t1d = SQ("c0", "v0")
    t2e = SQ("p1, p2", "h1, h2")
    elementary_contractions = set(compute_elementary_contractions_list([t1d, h, t2e], max_cu=4))

    ref = set(compute_elementary_contractions_list([h, t2e], max_cu=4))
    ref.update(compute_elementary_contractions_list([t1d, h], max_cu=1))
    ref.update(compute_elementary_contractions_list([t1d, t2e], max_cu=1))

    assert elementary_contractions == ref


def test_ele_con_list_4():
    h = SQ("g0, h0", "g1, p0", 'spin-orbital')
    t1d = SQ("h3", "p3")
    t2e = SQ("p1, p2", "h1, h2")
    elementary_contractions = compute_elementary_contractions_list([t1d, h, t2e], max_cu=5)

    ref = set()
    upper_indices = t1d.upper_indices + h.upper_indices + t2e.upper_indices
    lower_indices = t1d.lower_indices + h.lower_indices + t2e.lower_indices
    invalid_contractions = {make_tensor('L', 'h3', 'p3'),
                            make_tensor('L', 'g0', 'g1'), make_tensor('L', 'g0', 'p0'),
                            make_tensor('L', 'h0', 'g1'), make_tensor('L', 'h0', 'p0'),
                            make_tensor('L', 'p1', 'h1'), make_tensor('L', 'p1', 'h2'),
                            make_tensor('L', 'p2', 'h1'), make_tensor('L', 'p2', 'h2'),
                            make_tensor('L', 'g0, h0', 'g1, p0'), make_tensor('L', 'p1, p2', 'h1, h2'),
                            make_tensor('L', 'g0', 'p3'), make_tensor('L', 'h0', 'p3'),
                            make_tensor('L', 'p1', 'p3'), make_tensor('L', 'p2', 'p3'),
                            make_tensor('L', 'p1', 'p0'), make_tensor('L', 'p2', 'p0'),
                            make_tensor('L', 'p1', 'g1'), make_tensor('L', 'p2', 'g1')}
    for cu in range(1, 6):
        for upper in combinations(upper_indices, cu):
            u_indices = Indices.make_indices(upper, 'so')
            for lower in combinations(lower_indices, cu):
                l_indices = Indices.make_indices(lower, 'so')
                tensor = make_tensor('L', u_indices, l_indices)
                if tensor not in invalid_contractions:
                    ref.add(tensor)
    ref |= {make_tensor('C', 'g0', 'p3'), make_tensor('C', 'h0', 'p3'),
            make_tensor('C', 'p1', 'p3'), make_tensor('C', 'p2', 'p3'),
            make_tensor('C', 'p1', 'p0'), make_tensor('C', 'p2', 'p0'),
            make_tensor('C', 'p1', 'g1'), make_tensor('C', 'p2', 'g1')}

    assert ref == set(elementary_contractions)


def test_ele_con_1():
    h = SQ("g0, g1", "g2, g3", 'spin-orbital')
    t = SQ("p0, p1", "h0, h1", 'spin-orbital')
    ele_con_cat = compute_elementary_contractions_categorized([h, t], max_cu=4)

    ref_cons = set(compute_elementary_contractions_list([h, t], max_cu=4))
    ref_blocks = {(0, 1), (1, 0), (0, 0, 1, 1), (0, 0, 0, 1), (1, 1, 0, 0),
                  (1, 1, 0, 1), (0, 1, 0, 0), (0, 1, 1, 1), (0, 1, 0, 1),
                  (0, 1, 1, 0, 1, 1), (0, 1, 1, 0, 0, 1), (0, 0, 1, 0, 1, 1),
                  (0, 0, 1, 0, 0, 1), (0, 0, 1, 1, 0, 0, 1, 1)}

    for block, cons in ele_con_cat.items():
        assert set(cons) <= ref_cons
        assert block in ref_blocks


def test_ele_con_2():
    h = SQ("g0", "g0")
    t1d = SQ("h0", "p0")
    t1e = SQ("p1", "h1")
    ele_con_cat = compute_elementary_contractions_categorized([t1d, h, t1e], 2)

    ref = {(0, 1): [make_tensor('L', 'h0', 'g0')],
           (1, 0): [make_tensor('C', 'g0', 'p0')],
           (0, 2): [make_tensor('L', 'h0', 'h1')],
           (2, 0): [make_tensor('C', 'p1', 'p0')],
           (1, 2): [make_tensor('L', 'g0', 'h1')],
           (2, 1): [make_tensor('C', 'p1', 'g0')],
           (0, 1, 0, 1): [make_tensor('L', 'h0, g0', 'p0, g0')],
           (0, 1, 0, 2): [make_tensor('L', 'h0, g0', 'p0, h1')],
           (0, 1, 1, 2): [make_tensor('L', 'h0, g0', 'g0, h1')],
           (0, 2, 0, 1): [make_tensor('L', 'h0, p1', 'p0, g0')],
           (0, 2, 0, 2): [make_tensor('L', 'h0, p1', 'p0, h1')],
           (0, 2, 1, 2): [make_tensor('L', 'h0, p1', 'g0, h1')],
           (1, 2, 0, 1): [make_tensor('L', 'g0, p1', 'p0, g0')],
           (1, 2, 0, 2): [make_tensor('L', 'g0, p1', 'p0, h1')],
           (1, 2, 1, 2): [make_tensor('L', 'g0, p1', 'g0, h1')]}

    for block, cons in ele_con_cat.items():
        assert cons == ref[block]


def test_contraction_1():
    h = SQ("g0", "g1")
    t = SQ("p0", "h0")
    ref = [[(1, [], SQ("g0, p0", "g1, h0"))],
           [(1, [make_tensor('L', "g0", "h0"), make_tensor('K', "p0", "g1")], SQ.make_empty()),
            (-1, [make_tensor('L', "g0", "h0"), make_tensor('L', "p0", "g1")], SQ.make_empty())],
           [(-1, [make_tensor('L', "g0", "h0")], SQ("p0", "g1"))],
           [(1, [make_tensor('K', "p0", "g1")], SQ("g0", "h0")),
            (-1, [make_tensor('L', "p0", "g1")], SQ("g0", "h0"))],
           [(1, [make_tensor('L', "g0, p0", "g1, h0")], SQ.make_empty())]]
    a = list(compute_operator_contractions_general([h, t], max_cu=2, max_n_open=4))
    for i in a:
        assert i in ref
    assert len(a) == len(ref)


def test_contraction_2():
    h = SQ("g0", "g0")
    t2e = SQ("p0, p1", "h0, h1")

    ref = [[(1, [], SQ('g0, p0, p1', 'g0, h0, h1'))],
           [(-1, [make_tensor('L', 'g0', 'h0')], SQ('p0, p1', 'g0, h1'))],
           [(1, [make_tensor('L', 'g0', 'h1')], SQ('p0, p1', 'g0, h0'))],
           [(1, [make_tensor('K', 'p0', 'g0')], SQ('g0, p1', 'h0, h1')),
            (-1, [make_tensor('L', 'p0', 'g0')], SQ('g0, p1', 'h0, h1'))],
           [(-1, [make_tensor('K', 'p1', 'g0')], SQ('g0, p0', 'h0, h1')),
            (1, [make_tensor('L', 'p1', 'g0')], SQ('g0, p0', 'h0, h1'))],
           [(1, [make_tensor('L', 'p0, p1', 'g0, h0')], SQ('g0', 'h1'))],
           [(-1, [make_tensor('L', 'p0, p1', 'g0, h1')], SQ('g0', 'h0'))],
           [(1, [make_tensor('L', 'g0, p0', 'h0, h1')], SQ('p1', 'g0'))],
           [(-1, [make_tensor('L', 'g0, p1', 'h0, h1')], SQ('p0', 'g0'))],
           [(1, [make_tensor('L', 'g0, p0', 'g0, h0')], SQ('p1', 'h1'))],
           [(-1, [make_tensor('L', 'g0, p0', 'g0, h1')], SQ('p1', 'h0'))],
           [(-1, [make_tensor('L', 'g0, p1', 'g0, h0')], SQ('p0', 'h1'))],
           [(1, [make_tensor('L', 'g0, p1', 'g0, h1')], SQ('p0', 'h0'))],
           [(1, [make_tensor('L', 'g0', 'h0'), make_tensor('K', 'p0', 'g0')], SQ('p1', 'h1')),
            (-1, [make_tensor('L', 'g0', 'h0'), make_tensor('L', 'p0', 'g0')], SQ('p1', 'h1'))],
           [(-1, [make_tensor('L', 'g0', 'h0'), make_tensor('K', 'p1', 'g0')], SQ('p0', 'h1')),
            (1, [make_tensor('L', 'g0', 'h0'), make_tensor('L', 'p1', 'g0')], SQ('p0', 'h1'))],
           [(-1, [make_tensor('L', 'g0', 'h1'), make_tensor('K', 'p0', 'g0')], SQ('p1', 'h0')),
            (1, [make_tensor('L', 'g0', 'h1'), make_tensor('L', 'p0', 'g0')], SQ('p1', 'h0'))],
           [(1, [make_tensor('L', 'g0', 'h1'), make_tensor('K', 'p1', 'g0')], SQ('p0', 'h0')),
            (-1, [make_tensor('L', 'g0', 'h1'), make_tensor('L', 'p1', 'g0')], SQ('p0', 'h0'))],
           [(-1, [make_tensor('L', 'g0', 'h0'), make_tensor('L', 'p0, p1', 'g0, h1')], SQ.make_empty())],
           [(1, [make_tensor('L', 'g0', 'h1'), make_tensor('L', 'p0, p1', 'g0, h0')], SQ.make_empty())],
           [(1, [make_tensor('L', 'g0, p1', 'h0, h1'), make_tensor('K', 'p0', 'g0')], SQ.make_empty()),
            (-1, [make_tensor('L', 'g0, p1', 'h0, h1'), make_tensor('L', 'p0', 'g0')], SQ.make_empty())],
           [(-1, [make_tensor('L', 'g0, p0', 'h0, h1'), make_tensor('K', 'p1', 'g0')], SQ.make_empty()),
            (1, [make_tensor('L', 'g0, p0', 'h0, h1'), make_tensor('L', 'p1', 'g0')], SQ.make_empty())],
           [(1, [make_tensor('L', 'g0, p0, p1', 'g0, h0, h1')], SQ.make_empty())]]

    a = list(compute_operator_contractions_general([h, t2e], max_cu=3, n_process=2, batch_size=0))
    for i in a:
        assert i in ref
    assert len(a) == len(ref)  # 22

    a = list(compute_operator_contractions_general([h, t2e], max_cu=2, max_n_open=4, min_n_open=2,
                                                   n_process=2, batch_size=0))
    for i in a:
        assert i in ref
    assert len(a) == (len(ref) - 6)  # 16


def test_contraction_3():
    def canonicalize_densities(_densities, _sign):
        out = []
        for d in _densities:
            d_new, _s = d.canonicalize()
            out.append(d_new)
            _sign *= _s
        return _sign, out

    h = SQ("g0", "g0")
    t1d = SQ("h0", "p0")
    t1e = SQ("p1", "h1")
    ref = []
    for step_1 in compute_operator_contractions_general([t1d, h], max_cu=2):
        for sign_1, densities_1, sq_op_1 in step_1:
            sign_1, densities_1 = canonicalize_densities(densities_1, sign_1)
            max_cu = sq_op_1.n_body + 1

            for step_2 in compute_operator_contractions_general([sq_op_1, t1e], max_cu=max_cu, n_process=2):
                for sign_2, densities_2, sq_op_2 in step_2:
                    sign_2, densities_2 = canonicalize_densities(densities_2, sign_2)
                    ref.append((sign_1 * sign_2, sorted(densities_1 + densities_2), sq_op_2))

    a = []
    for con in compute_operator_contractions_general([t1d, h, t1e], max_cu=3, n_process=2, batch_size=0):
        for sign, densities, sq_op in con:
            sign, densities = canonicalize_densities(densities, sign)
            a.append((sign, sorted(densities), sq_op))
            assert a[-1] in ref
    assert len(a) == len(ref)


def test_contraction_4():
    from timeit import default_timer as timer

    h = SQ("g0, g1", "g2, g3")
    t2e = SQ("v0, v1", "c0, c1")
    t2d = SQ("c2, c3", "v2, v3")
    t2ee = SQ("v4, v5", "c4, c5")

    start = timer()
    a = list(compute_operator_contractions_general([t2d, h, t2e, t2ee], max_cu=1, n_process=4, batch_size=0))
    print(f"Time to compute T2^+ * H * T2 * T2: {timer() - start:.3f} s")
    assert len(a) == 72832


def test_contraction_categorized_1():
    h = SQ("g0", "g1")
    t = SQ("p0", "h0")
    ref = list(compute_operator_contractions([h, t], max_cu=2, max_n_open=4, for_commutator=False))
    a = list(compute_operator_contractions([h, t], max_cu=2, max_n_open=4, for_commutator=True))
    for i in a:
        assert i in ref
    assert len(a) == (len(ref) - 2)  # minus pure L2 and un-contracted


def test_contraction_categorized_2():
    h = SQ("g0", "g0")
    t2e = SQ("p0, p1", "h0, h1")

    ref = list(compute_operator_contractions_general([h, t2e], max_cu=3, n_process=2, batch_size=0))
    ref = [(sign, sorted(densities), sq_op) for con in ref for sign, densities, sq_op in con]

    a = list(compute_operator_contractions([h, t2e], max_cu=3, n_process=2, batch_size=0, for_commutator=True))
    a = [(sign, sorted(densities), sq_op) for con in a for sign, densities, sq_op in con]

    for i in a:
        assert i in ref
    assert len(a) == (len(ref) - 1 - 9)  # 1 un-contracted, 9 pure cumulant


def test_contraction_categorized_3():
    MT = Tensor.make_tensor

    h = SQ("g0", "g0")
    t1d = SQ("h0", "p0")
    t1e = SQ("p1", "h1")

    # 13 double pairwise, 9 single pairwise with 2-cumulant, 6 triple pairwise
    ref = [(1, [MT('L', 'g0', 'h1'), MT('L', 'h0', 'g0')], SQ('p1', 'p0')),
           (-1, [MT('K', 'p1', 'p0'), MT('L', 'h0', 'g0')], SQ('g0', 'h1')),
           (1, [MT('L', 'p1', 'p0'), MT('L', 'h0', 'g0')], SQ('g0', 'h1')),
           (-1, [MT('K', 'g0', 'p0'), MT('L', 'h0', 'h1')], SQ('p1', 'g0')),
           (1, [MT('L', 'g0', 'p0'), MT('L', 'h0', 'h1')], SQ('p1', 'g0')),
           (-1, [MT('K', 'p1', 'g0'), MT('L', 'h0', 'h1')], SQ('g0', 'p0')),
           (1, [MT('L', 'p1', 'g0'), MT('L', 'h0', 'h1')], SQ('g0', 'p0')),
           (1, [MT('K', 'g0', 'p0'), MT('K', 'p1', 'g0')], SQ('h0', 'h1')),
           (-1, [MT('K', 'p1', 'g0'), MT('L', 'g0', 'p0')], SQ('h0', 'h1')),
           (-1, [MT('K', 'g0', 'p0'), MT('L', 'p1', 'g0')], SQ('h0', 'h1')),
           (1, [MT('L', 'g0', 'p0'), MT('L', 'p1', 'g0')], SQ('h0', 'h1')),
           (-1, [MT('K', 'p1', 'p0'), MT('L', 'g0', 'h1')], SQ('h0', 'g0')),
           (1, [MT('L', 'g0', 'h1'), MT('L', 'p1', 'p0')], SQ('h0', 'g0')),
           (-1, [MT('L', 'h0', 'g0'), MT('L', 'g0, p1', 'p0, h1')], SQ.make_empty()),
           (1, [MT('L', 'h0', 'h1'), MT('L', 'g0, p1', 'p0, g0')], SQ.make_empty()),
           (1, [MT('K', 'g0', 'p0'), MT('L', 'h0, p1', 'g0, h1')], SQ.make_empty()),
           (-1, [MT('L', 'g0', 'p0'), MT('L', 'h0, p1', 'g0, h1')], SQ.make_empty()),
           (-1, [MT('L', 'g0', 'h1'), MT('L', 'h0, p1', 'p0, g0')], SQ.make_empty()),
           (-1, [MT('K', 'p1', 'p0'), MT('L', 'h0, g0', 'g0, h1')], SQ.make_empty()),
           (1, [MT('L', 'p1', 'p0'), MT('L', 'h0, g0', 'g0, h1')], SQ.make_empty()),
           (1, [MT('K', 'p1', 'g0'), MT('L', 'h0, g0', 'p0, h1')], SQ.make_empty()),
           (-1, [MT('L', 'p1', 'g0'), MT('L', 'h0, g0', 'p0, h1')], SQ.make_empty()),
           (-1, [MT('K', 'p1', 'p0'), MT('L', 'g0', 'h1'), MT('L', 'h0', 'g0')], SQ.make_empty()),
           (1, [MT('L', 'g0', 'h1'), MT('L', 'p1', 'p0'), MT('L', 'h0', 'g0')], SQ.make_empty()),
           (1, [MT('K', 'g0', 'p0'), MT('K', 'p1', 'g0'), MT('L', 'h0', 'h1')], SQ.make_empty()),
           (-1, [MT('K', 'p1', 'g0'), MT('L', 'g0', 'p0'), MT('L', 'h0', 'h1')], SQ.make_empty()),
           (-1, [MT('K', 'g0', 'p0'), MT('L', 'p1', 'g0'), MT('L', 'h0', 'h1')], SQ.make_empty()),
           (1, [MT('L', 'g0', 'p0'), MT('L', 'p1', 'g0'), MT('L', 'h0', 'h1')], SQ.make_empty())]

    a = list(compute_operator_contractions([t1d, h, t1e], max_cu=3, n_process=2, batch_size=0, for_commutator=True))
    a = [(sign, sorted(densities), sq_op) for con in a for sign, densities, sq_op in con]

    for i in a:
        assert i in ref
    assert len(a) == len(ref)


def test_expand_hole():
    tensors = [make_tensor('Hamiltonian', "g0", "h1", 'spin-integrated'),
               make_tensor('hole_density', "a4", "p3", 'spin-orbital'),
               make_tensor('cluster_amplitude', "a1,a2", "p0,p1", 'spin-orbital'),
               make_tensor('hole_density', "v0", "p6", 'spin-orbital'),
               make_tensor('hole_density', "h0", "p4", 'spin-orbital')]
    ref = [(1, [make_tensor('Hamiltonian', "g0", "h1", 'spin-integrated'),
                make_tensor('cluster_amplitude', "a1,a2", "p0,p1", 'spin-orbital'),
                make_tensor('Kronecker', "a4", "p3", 'spin-orbital'),
                make_tensor('Kronecker', "v0", "p6", 'spin-orbital'),
                make_tensor('Kronecker', "h0", "p4", 'spin-orbital')]),
           (-1, [make_tensor('Hamiltonian', "g0", "h1", 'spin-integrated'),
                 make_tensor('cluster_amplitude', "a1,a2", "p0,p1", 'spin-orbital'),
                 make_tensor('Kronecker', "a4", "p3", 'spin-orbital'),
                 make_tensor('Kronecker', "v0", "p6", 'spin-orbital'),
                 make_tensor('cumulant', "h0", "p4", 'spin-orbital')]),
           (-1, [make_tensor('Hamiltonian', "g0", "h1", 'spin-integrated'),
                 make_tensor('cluster_amplitude', "a1,a2", "p0,p1", 'spin-orbital'),
                 make_tensor('cumulant', "a4", "p3", 'spin-orbital'),
                 make_tensor('Kronecker', "v0", "p6", 'spin-orbital'),
                 make_tensor('Kronecker', "h0", "p4", 'spin-orbital')]),
           (1, [make_tensor('Hamiltonian', "g0", "h1", 'spin-integrated'),
                make_tensor('cluster_amplitude', "a1,a2", "p0,p1", 'spin-orbital'),
                make_tensor('cumulant', "a4", "p3", 'spin-orbital'),
                make_tensor('Kronecker', "v0", "p6", 'spin-orbital'),
                make_tensor('cumulant', "h0", "p4", 'spin-orbital')])]

    expanded = list(expand_hole_densities(tensors))
    for sign_tensors in expanded:
        assert sign_tensors in ref
    assert len(expanded) == len(ref)

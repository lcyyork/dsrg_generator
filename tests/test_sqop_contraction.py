from itertools import combinations

from src.Indices import Indices
from src.SQOperator import SecondQuantizedOperator
from src.Tensor import Tensor
from src.sqop_contraction import expand_hole_densities
from src.sqop_contraction import compute_elementary_contractions_list, compute_elementary_contractions_categorized
from src.sqop_contraction import compute_operator_contractions, compute_operator_contractions_general


def test_ele_con_list_1():
    h = SecondQuantizedOperator("g0", "g1", 'spin-orbital')
    t = SecondQuantizedOperator("p0", "h0", 'spin-orbital')
    ref = {Tensor.make_tensor('cumulant', "g0", "h0"),
           Tensor.make_tensor('eta', "p0", "g1"),
           Tensor.make_tensor('lambda', "g0, p0", "g1, h0")}
    elementary_contractions = set(compute_elementary_contractions_list([h, t]))
    assert elementary_contractions == ref

    h = SecondQuantizedOperator("c0", "g1", 'spin-orbital')
    ref = {Tensor.make_tensor('cumulant', "c0", "h0"),
           Tensor.make_tensor('eta', "p0", "g1")}
    elementary_contractions = set(compute_elementary_contractions_list([h, t]))
    assert elementary_contractions == ref

    h = SecondQuantizedOperator("v0", "g1", 'spin-orbital')
    ref = {Tensor.make_tensor('eta', "p0", "g1")}
    elementary_contractions = set(compute_elementary_contractions_list([h, t]))
    assert elementary_contractions == ref


def test_ele_con_list_2():
    h = SecondQuantizedOperator("g0, g1", "g2, g3", 'spin-orbital')
    t = SecondQuantizedOperator("p0, p1", "h0, h1", 'spin-orbital')
    elementary_contractions = set(compute_elementary_contractions_list([h, t], max_cu=4))

    ref = {Tensor.make_tensor('L', 'g0', 'h0'), Tensor.make_tensor('L', 'g0', 'h1'),
           Tensor.make_tensor('L', 'g1', 'h0'), Tensor.make_tensor('L', 'g1', 'h1'),
           Tensor.make_tensor('C', 'p0', 'g2'), Tensor.make_tensor('C', 'p0', 'g3'),
           Tensor.make_tensor('C', 'p1', 'g2'), Tensor.make_tensor('C', 'p1', 'g3'),
           Tensor.make_tensor('L', "g0, g1", "h0, h1"), Tensor.make_tensor('L', "p0, p1", "g2, g3"),
           Tensor.make_tensor('L', "g0, g1", "g2, h0"), Tensor.make_tensor('L', "g0, g1", "g2, h1"),
           Tensor.make_tensor('L', "g0, g1", "g3, h0"), Tensor.make_tensor('L', "g0, g1", "g3, h1"),
           Tensor.make_tensor('L', "p0, p1", "g2, h0"), Tensor.make_tensor('L', "p0, p1", "g2, h1"),
           Tensor.make_tensor('L', "p0, p1", "g3, h0"), Tensor.make_tensor('L', "p0, p1", "g3, h1"),
           Tensor.make_tensor('L', "g0, p0", "g2, g3"), Tensor.make_tensor('L', "g0, p1", "g2, g3"),
           Tensor.make_tensor('L', "g1, p0", "g2, g3"), Tensor.make_tensor('L', "g1, p1", "g2, g3"),
           Tensor.make_tensor('L', "g0, p0", "h0, h1"), Tensor.make_tensor('L', "g0, p1", "h0, h1"),
           Tensor.make_tensor('L', "g1, p0", "h0, h1"), Tensor.make_tensor('L', "g1, p1", "h0, h1"),
           Tensor.make_tensor('L', "g0, p0", "g2, h0"), Tensor.make_tensor('L', "g0, p0", "g2, h1"),
           Tensor.make_tensor('L', "g0, p0", "g3, h0"), Tensor.make_tensor('L', "g0, p0", "g3, h1"),
           Tensor.make_tensor('L', "g0, p1", "g2, h0"), Tensor.make_tensor('L', "g0, p1", "g2, h1"),
           Tensor.make_tensor('L', "g0, p1", "g3, h0"), Tensor.make_tensor('L', "g0, p1", "g3, h1"),
           Tensor.make_tensor('L', "g1, p0", "g2, h0"), Tensor.make_tensor('L', "g1, p0", "g2, h1"),
           Tensor.make_tensor('L', "g1, p0", "g3, h0"), Tensor.make_tensor('L', "g1, p0", "g3, h1"),
           Tensor.make_tensor('L', "g1, p1", "g2, h0"), Tensor.make_tensor('L', "g1, p1", "g2, h1"),
           Tensor.make_tensor('L', "g1, p1", "g3, h0"), Tensor.make_tensor('L', "g1, p1", "g3, h1"),
           Tensor.make_tensor('L', 'g0, p0, p1', 'g2, h0, h1'), Tensor.make_tensor('L', 'g0, p0, p1', 'g3, h0, h1'),
           Tensor.make_tensor('L', 'g1, p0, p1', 'g2, h0, h1'), Tensor.make_tensor('L', 'g1, p0, p1', 'g3, h0, h1'),
           Tensor.make_tensor('L', 'g0, p0, p1', 'g2, g3, h0'), Tensor.make_tensor('L', 'g0, p0, p1', 'g2, g3, h1'),
           Tensor.make_tensor('L', 'g1, p0, p1', 'g2, g3, h0'), Tensor.make_tensor('L', 'g1, p0, p1', 'g2, g3, h1'),
           Tensor.make_tensor('L', 'g0, g1, p0', 'g2, h0, h1'), Tensor.make_tensor('L', 'g0, g1, p0', 'g3, h0, h1'),
           Tensor.make_tensor('L', 'g0, g1, p1', 'g2, h0, h1'), Tensor.make_tensor('L', 'g0, g1, p1', 'g3, h0, h1'),
           Tensor.make_tensor('L', 'g0, g1, p0', 'g2, g3, h0'), Tensor.make_tensor('L', 'g0, g1, p0', 'g2, g3, h1'),
           Tensor.make_tensor('L', 'g0, g1, p1', 'g2, g3, h0'), Tensor.make_tensor('L', 'g0, g1, p1', 'g2, g3, h1'),
           Tensor.make_tensor('L', 'g0, g1, p0, p1', 'g2, g3, h0, h1')}

    assert elementary_contractions == ref


def test_ele_con_list_3():
    h = SecondQuantizedOperator("g0, h0", "g1, p0", 'spin-orbital')
    t1d = SecondQuantizedOperator("c0", "v0")
    t2e = SecondQuantizedOperator("p1, p2", "h1, h2")
    elementary_contractions = set(compute_elementary_contractions_list([t1d, h, t2e], max_cu=4))

    ref = set(compute_elementary_contractions_list([h, t2e], max_cu=4))
    ref.update(compute_elementary_contractions_list([t1d, h], max_cu=1))
    ref.update(compute_elementary_contractions_list([t1d, t2e], max_cu=1))

    assert elementary_contractions == ref


def test_ele_con_list_4():
    h = SecondQuantizedOperator("g0, h0", "g1, p0", 'spin-orbital')
    t1d = SecondQuantizedOperator("h3", "p3")
    t2e = SecondQuantizedOperator("p1, p2", "h1, h2")
    elementary_contractions = compute_elementary_contractions_list([t1d, h, t2e], max_cu=5)

    ref = set()
    upper_indices = t1d.upper_indices + h.upper_indices + t2e.upper_indices
    lower_indices = t1d.lower_indices + h.lower_indices + t2e.lower_indices
    invalid_contractions = {Tensor.make_tensor('L', 'h3', 'p3'),
                            Tensor.make_tensor('L', 'g0', 'g1'), Tensor.make_tensor('L', 'g0', 'p0'),
                            Tensor.make_tensor('L', 'h0', 'g1'), Tensor.make_tensor('L', 'h0', 'p0'),
                            Tensor.make_tensor('L', 'p1', 'h1'), Tensor.make_tensor('L', 'p1', 'h2'),
                            Tensor.make_tensor('L', 'p2', 'h1'), Tensor.make_tensor('L', 'p2', 'h2'),
                            Tensor.make_tensor('L', 'g0, h0', 'g1, p0'), Tensor.make_tensor('L', 'p1, p2', 'h1, h2'),
                            Tensor.make_tensor('L', 'g0', 'p3'), Tensor.make_tensor('L', 'h0', 'p3'),
                            Tensor.make_tensor('L', 'p1', 'p3'), Tensor.make_tensor('L', 'p2', 'p3'),
                            Tensor.make_tensor('L', 'p1', 'p0'), Tensor.make_tensor('L', 'p2', 'p0'),
                            Tensor.make_tensor('L', 'p1', 'g1'), Tensor.make_tensor('L', 'p2', 'g1')}
    for cu in range(1, 6):
        for upper in combinations(upper_indices, cu):
            u_indices = Indices.make_indices(upper, 'so')
            for lower in combinations(lower_indices, cu):
                l_indices = Indices.make_indices(lower, 'so')
                tensor = Tensor.make_tensor('L', u_indices, l_indices)
                if tensor not in invalid_contractions:
                    ref.add(tensor)
    ref |= {Tensor.make_tensor('C', 'g0', 'p3'), Tensor.make_tensor('C', 'h0', 'p3'),
            Tensor.make_tensor('C', 'p1', 'p3'), Tensor.make_tensor('C', 'p2', 'p3'),
            Tensor.make_tensor('C', 'p1', 'p0'), Tensor.make_tensor('C', 'p2', 'p0'),
            Tensor.make_tensor('C', 'p1', 'g1'), Tensor.make_tensor('C', 'p2', 'g1')}

    assert ref == set(elementary_contractions)


def test_ele_con_1():
    h = SecondQuantizedOperator("g0, g1", "g2, g3", 'spin-orbital')
    t = SecondQuantizedOperator("p0, p1", "h0, h1", 'spin-orbital')
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
    h = SecondQuantizedOperator("g0", "g0")
    t1d = SecondQuantizedOperator("h0", "p0")
    t1e = SecondQuantizedOperator("p1", "h1")
    ele_con_cat = compute_elementary_contractions_categorized([t1d, h, t1e], 2)

    ref = {(0, 1): [Tensor.make_tensor('L', 'h0', 'g0')],
           (1, 0): [Tensor.make_tensor('C', 'g0', 'p0')],
           (0, 2): [Tensor.make_tensor('L', 'h0', 'h1')],
           (2, 0): [Tensor.make_tensor('C', 'p1', 'p0')],
           (1, 2): [Tensor.make_tensor('L', 'g0', 'h1')],
           (2, 1): [Tensor.make_tensor('C', 'p1', 'g0')],
           (0, 1, 0, 1): [Tensor.make_tensor('L', 'h0, g0', 'p0, g0')],
           (0, 1, 0, 2): [Tensor.make_tensor('L', 'h0, g0', 'p0, h1')],
           (0, 1, 1, 2): [Tensor.make_tensor('L', 'h0, g0', 'g0, h1')],
           (0, 2, 0, 1): [Tensor.make_tensor('L', 'h0, p1', 'p0, g0')],
           (0, 2, 0, 2): [Tensor.make_tensor('L', 'h0, p1', 'p0, h1')],
           (0, 2, 1, 2): [Tensor.make_tensor('L', 'h0, p1', 'g0, h1')],
           (1, 2, 0, 1): [Tensor.make_tensor('L', 'g0, p1', 'p0, g0')],
           (1, 2, 0, 2): [Tensor.make_tensor('L', 'g0, p1', 'p0, h1')],
           (1, 2, 1, 2): [Tensor.make_tensor('L', 'g0, p1', 'g0, h1')]}

    for block, cons in ele_con_cat.items():
        assert cons == ref[block]


def test_contraction_1():
    h = SecondQuantizedOperator("g0", "g1")
    t = SecondQuantizedOperator("p0", "h0")
    ref = [[(1, [], SecondQuantizedOperator("g0, p0", "g1, h0"))],
           [(1, [Tensor.make_tensor('L', "g0", "h0"), Tensor.make_tensor('K', "p0", "g1")],
             SecondQuantizedOperator([], [])),
            (-1, [Tensor.make_tensor('L', "g0", "h0"), Tensor.make_tensor('L', "p0", "g1")],
             SecondQuantizedOperator([], []))],
           [(-1, [Tensor.make_tensor('L', "g0", "h0")], SecondQuantizedOperator("p0", "g1"))],
           [(1, [Tensor.make_tensor('K', "p0", "g1")], SecondQuantizedOperator("g0", "h0")),
            (-1, [Tensor.make_tensor('L', "p0", "g1")], SecondQuantizedOperator("g0", "h0"))],
           [(1, [Tensor.make_tensor('L', "g0, p0", "g1, h0")], SecondQuantizedOperator([], []))]
           ]
    count = 0
    for i in compute_operator_contractions_general([h, t], max_cu=2, max_n_open=4):
        assert i in ref
        count += 1
    assert count == len(ref)


def test_contraction_2():
    from timeit import default_timer as timer
    h = SecondQuantizedOperator("g0", "g0")
    # t2d = SecondQuantizedOperator("h0", "p0")
    # t2e = SecondQuantizedOperator("p1", "h1")
    # h = SecondQuantizedOperator("g0", "g0")
    t2e = SecondQuantizedOperator("p0, p1", "h0, h1")
    # t2d = SecondQuantizedOperator("h2, h3", "p2, p3")
    start = timer()
    a = list(compute_operator_contractions_general([h, t2e], max_cu=4, n_process=1, batch_size=0))
    for i in a:
        print(i)
    print(timer() - start)
    print(f"len(a) = {len(a)}")


def test_contraction_3():
    from timeit import default_timer as timer
    from sympy import binomial
    h = SecondQuantizedOperator("g0, g1", "g2, g3")
    t2e = SecondQuantizedOperator("p0, p1", "h0, h1")
    t2d = SecondQuantizedOperator("h2, h3", "p2, p3")
    t1e = SecondQuantizedOperator("p4", "h4")
    t2ee = SecondQuantizedOperator("p4, p5", "h4, h5")
    start = timer()
    a = list(compute_operator_contractions_general([t2d, h, t2e, t1e], max_cu=1, n_process=4, batch_size=20))
    print(timer() - start)
    print(f"len(a) = {len(a)}")

    start = timer()
    count = 0
    for i in compute_operator_contractions_general([t2d, h, t2e, t2ee], max_cu=1, n_process=4, batch_size=0):
        i[0] = binomial(15, 8)
        count += 1
    print(timer() - start)
    print(f"count = {count}")


def test_expand_hole():
    tensors = [Tensor.make_tensor('Hamiltonian', "g0", "h1", 'spin-integrated'),
               Tensor.make_tensor('hole_density', "a4", "p3", 'spin-orbital'),
               Tensor.make_tensor('cluster_amplitude', "a1,a2", "p0,p1", 'spin-orbital'),
               Tensor.make_tensor('hole_density', "v0", "p6", 'spin-orbital'),
               Tensor.make_tensor('hole_density', "h0", "p4", 'spin-orbital')]
    ref = [(1, [Tensor.make_tensor('Hamiltonian', "g0", "h1", 'spin-integrated'),
                Tensor.make_tensor('cluster_amplitude', "a1,a2", "p0,p1", 'spin-orbital'),
                Tensor.make_tensor('Kronecker', "a4", "p3", 'spin-orbital'),
                Tensor.make_tensor('Kronecker', "v0", "p6", 'spin-orbital'),
                Tensor.make_tensor('Kronecker', "h0", "p4", 'spin-orbital')]),
           (-1, [Tensor.make_tensor('Hamiltonian', "g0", "h1", 'spin-integrated'),
                 Tensor.make_tensor('cluster_amplitude', "a1,a2", "p0,p1", 'spin-orbital'),
                 Tensor.make_tensor('Kronecker', "a4", "p3", 'spin-orbital'),
                 Tensor.make_tensor('Kronecker', "v0", "p6", 'spin-orbital'),
                 Tensor.make_tensor('cumulant', "h0", "p4", 'spin-orbital')]),
           (-1, [Tensor.make_tensor('Hamiltonian', "g0", "h1", 'spin-integrated'),
                 Tensor.make_tensor('cluster_amplitude', "a1,a2", "p0,p1", 'spin-orbital'),
                 Tensor.make_tensor('cumulant', "a4", "p3", 'spin-orbital'),
                 Tensor.make_tensor('Kronecker', "v0", "p6", 'spin-orbital'),
                 Tensor.make_tensor('Kronecker', "h0", "p4", 'spin-orbital')]),
           (1, [Tensor.make_tensor('Hamiltonian', "g0", "h1", 'spin-integrated'),
                Tensor.make_tensor('cluster_amplitude', "a1,a2", "p0,p1", 'spin-orbital'),
                Tensor.make_tensor('cumulant', "a4", "p3", 'spin-orbital'),
                Tensor.make_tensor('Kronecker', "v0", "p6", 'spin-orbital'),
                Tensor.make_tensor('cumulant', "h0", "p4", 'spin-orbital')])]

    expanded = list(expand_hole_densities(tensors))
    for sign_tensors in expanded:
        assert sign_tensors in ref
    assert len(expanded) == len(ref)


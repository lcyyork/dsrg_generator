from src.Index import Index
from src.Term import Term
from src.Tensor import Tensor
from src.SQOperator import SecondQuantizedOperator


make_tensor = Tensor.make_tensor
make_sq = SecondQuantizedOperator


def test_init():
    indices_type = 'spin-orbital'
    list_of_tensors = [Tensor.make_tensor('Hamiltonian', "g0,g1,c0", "g2,p0,v0", indices_type),
                       Tensor.make_tensor('cluster_amplitude', "p0,p1,g3", "a0,h1,a1", indices_type),
                       Tensor.make_tensor('Kronecker', "v0", "p1", indices_type),
                       Tensor.make_tensor('cumulant', "h1", "c0", indices_type),
                       Tensor.make_tensor('cumulant', "g2,a0", "g0,g1", indices_type),
                       Tensor.make_tensor('cumulant', "a1", "g3", indices_type)]
    sq_op = SecondQuantizedOperator.make_empty(indices_type)
    a = Term(list_of_tensors, sq_op)
    print(a)
    # print(a.next_index_number)
    # print(a._downgrade_cumulant_indices())
    # print(a)
    # print(a.next_index_number)
    # a._remove_kronecker_delta()
    # print(a)
    # print(a.next_index_number)
    a.simplify(simplify_core_cumulant=True)
    print(a)
    print(a.next_index_number)

    # for row in a.build_adjacency_matrix():
    #     print(row)
    # for row in a.order_tensors():
    #     print(row)
    print(a)
    print(a.canonicalize_sympy())

    list_of_tensors = [Tensor.make_tensor('Hamiltonian', "v0,c0", "v1,c1", indices_type),
                       Tensor.make_tensor('cluster_amplitude', "v1", "c2", indices_type),
                       Tensor.make_tensor('cluster_amplitude', "v2,v3", "c0,c3", indices_type),
                       Tensor.make_tensor('cluster_amplitude', "c2,c3", "v0,v3", indices_type)]
    sq_op = SecondQuantizedOperator("c1", "v2", indices_type)

    a = Term(list_of_tensors, sq_op)
    # print(a.canonicalize())

    print(a)
    print(a.canonicalize_sympy())

    # for i in a.canonicalize().generate_spin_cases_naive():
    #     print(i)


def test_init_2():
    a = Term([make_tensor('H', 'g0', 'g0')], make_sq("g0", "g0"))
    print(a)

    b = Term.from_term(a, True)
    print(b)


def test_latex_1():
    list_of_tensors = [make_tensor('H', 'v0, v1', 'c0, c1'), make_tensor('t', 'c0, c1', 'v0, v1')]
    sq_op = SecondQuantizedOperator.make_empty()
    a = Term(list_of_tensors, sq_op, 0.25)
    assert a.latex() == "1/4 H^{ v_{0} v_{1} }_{ c_{0} c_{1} } T^{ c_{0} c_{1} }_{ v_{0} v_{1} }"
    assert a.latex(dollar=True) == "$1/4 H^{ v_{0} v_{1} }_{ c_{0} c_{1} } T^{ c_{0} c_{1} }_{ v_{0} v_{1} }$"


def test_latex_2():
    a = Term([make_tensor('H', 'g0, h1', 'g0, p1')],
             make_sq('g0, p1', 'g0, h1'))
    ref = "1/4 & {\\cal P}(g_{0} / p_{1}) {\\cal P}(g_{0} / h_{1}) " \
          "H^{ g_{0} h_{1} }_{ g_{0} p_{1} } a^{ g_{0} p_{1} }_{ g_{0} h_{1} }"
    assert a.latex(delimiter=True) == ref


def test_latex_3():
    list_of_tensors = [make_tensor('H', 'v3, v4', 'v0, c3'),
                       make_tensor('T', 'c0, c1', 'v1, v3'),
                       make_tensor('T', 'c2, c3', 'v2, v4')]
    sq_op = make_sq('v0, v1, v2', 'c0, c1, c2')
    a = Term(list_of_tensors, sq_op, -1)

    ref = "-1/18 & {\\cal P}(v_{0} / v_{1} / v_{2}) {\\cal P}(c_{0} c_{1} / c_{2}) " \
          "H^{ v_{3} v_{4} }_{ v_{0} c_{3} } " \
          "T^{ c_{0} c_{1} }_{ v_{1} v_{3} } " \
          "T^{ c_{2} c_{3} }_{ v_{2} v_{4} } " \
          "a^{ v_{0} v_{1} v_{2} }_{ c_{0} c_{1} c_{2} } \\\\"

    assert a.latex(delimiter=True, backslash=True) == ref

    ref = "-1 H^{ v_{3} v_{4} }_{ v_{0} c_{3} } T^{ c_{0} c_{1} }_{ v_{1} v_{3} } " \
          "T^{ c_{2} c_{3} }_{ v_{2} v_{4} } a^{ v_{0} v_{1} v_{2} }_{ c_{0} c_{1} c_{2} }"

    assert a.latex(permute_format=False) == str(a) == ref


def test_ambit_1():
    list_of_tensors = [make_tensor('H', 'v0, v1', 'c0, c1'), make_tensor('t', 'c0, c1', 'v0, v1')]
    sq_op = SecondQuantizedOperator.make_empty()
    a = Term(list_of_tensors, sq_op, 0.25)
    assert a.ambit(name='X') == 'X0 += (1.0 / 4.0) * H2["v0,v1,c0,c1"] * T2["c0,c1,v0,v1"];'

    list_of_tensors = [make_tensor('H', 'v0, v1', 'g0, c1'), make_tensor('t', 'c0, c1', 'v0, v1')]
    sq_op = make_sq('g0', 'c0')
    a = Term(list_of_tensors, sq_op, 0.5)
    assert a.ambit() == 'C1["c0,g0"] += (1.0 / 2.0) * H2["v0,v1,g0,c1"] * T2["c0,c1,v0,v1"];'


def test_ambit_2():
    a = Term([make_tensor('H', 'g0, h1', 'g0, p1')], make_sq('g0, p1', 'g0, h1'))
    ref = '// Error: diagonal indices are not supported by ambit.\n' \
          'temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"ghgp"});\n' \
          'temp["g0,h1,g0,p1"] += 1.0 * H2["g0,h1,g0,p1"];\n' \
          'C2["g0,h1,g0,p1"] += temp["g0,h1,g0,p1"];\n' \
          'C2["g0,h1,p1,g0"] -= temp["g0,h1,g0,p1"];\n' \
          'C2["h1,g0,g0,p1"] -= temp["g0,h1,g0,p1"];\n' \
          'C2["h1,g0,p1,g0"] += temp["g0,h1,g0,p1"];\n'
    assert a.ambit() == ref


def test_ambit_3():
    a = Term([make_tensor('H', 'g0, h1', 'g1, p1')], make_sq('g1, p1', 'g0, h1'))
    ref = 'temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"ghgp"});\n' \
          'temp["g0,h1,g1,p1"] += 1.0 * H2["g0,h1,g1,p1"];'
    assert a.ambit(ignore_permutations=True) == ref

    assert a.ambit(ignore_permutations=True, init_temp=False) == 'temp["g0,h1,g1,p1"] += 1.0 * H2["g0,h1,g1,p1"];'

    assert a.ambit(init_temp=False, declared_temp=False) == 'C2["g0,h1,g1,p1"] += 1.0 * H2["g0,h1,g1,p1"];\n'


def test_almost_eq():
    a = Term([make_tensor('H', 'g0, h1', 'g1, p1')], make_sq('g1, p1', 'g0, h1'))
    b = Term.from_term(a, flip_sign=True)
    assert a != b
    assert a.almost_equal(b)
    assert a.coeff + b.coeff == 0


def test_lt():
    pass


def test_le():
    pass


def test_gt():
    pass


def test_ge():
    pass


def test_void():
    a = Term([make_tensor('H', 'g0, h1', 'g0, p1')], make_sq('g0, p1', 'g0, h1'), 0.0)
    assert a.is_void()

    b = a.void()
    assert b.is_void()
    assert b is not a
    assert b != a

    c = Term.make_empty('so')
    assert b == c


def test_perm_part():
    list_of_tensors = [make_tensor('H', 'g0, g1', 'g2, p0'), make_tensor('T', 'h0, h1', 'p0, p1')]
    sq_op = make_sq('g0, g1, p1', 'g2, h0, h1')
    a = Term(list_of_tensors, sq_op)
    cre_part, ann_part = a.perm_partition_open()
    assert cre_part == [[Index('g0'), Index('g1')], [Index('p1')]]
    assert ann_part == [[Index('g2')], [Index('h0'), Index('h1')]]

    list_of_tensors = [make_tensor('H', 'v4, c2', 'v3, c1'),
                       make_tensor('T', 'c1', 'v0'),
                       make_tensor('T', 'c0, c3', 'v1, v4'),
                       make_tensor('T', 'c2, c3', 'v2, v3')]
    sq_op = make_sq('v2, c0', 'v0, v1')
    a = Term(list_of_tensors, sq_op)
    cre_part, ann_part = a.perm_partition_open()
    assert cre_part == [[Index('c0')], [Index('v2')]]
    assert ann_part == [[Index('v0')], [Index('v1')]]

    list_of_tensors = [make_tensor('H', 'v3, v4', 'v0, c3'),
                       make_tensor('T', 'c0, c1', 'v1, v3'),
                       make_tensor('T', 'c2, c3', 'v2, v4')]
    sq_op = make_sq('v0, v1, v2', 'c0, c1, c2')
    a = Term(list_of_tensors, sq_op)
    cre_part, ann_part = a.perm_partition_open()
    assert cre_part == [[Index('v0')], [Index('v1')], [Index('v2')]]
    assert ann_part == [[Index('c0'), Index('c1')], [Index('c2')]]

    list_of_tensors = [make_tensor('H', 'g0, h1', 'p1, g0')]
    sq_op = make_sq('g0, p1', 'g0, h1')
    a = Term(list_of_tensors, sq_op, -1)
    cre_part, ann_part = a.perm_partition_open()
    assert cre_part == [[Index('g0')], [Index('p1')]]
    assert ann_part == [[Index('g0')], [Index('h1')]]


def test_simplify():
    list_of_tensors = [make_tensor('H', "g0", "g0"),
                       make_tensor('t', "h0", "p0"), make_tensor('t', "p1", "h1"),
                       make_tensor('L', 'h0', 'g0'), make_tensor('L', 'g0, p1', 'p0, h1')]
    a = Term(list_of_tensors, SecondQuantizedOperator.make_empty(), -1)
    a.simplify()
    assert a.is_void()

    list_of_tensors = [make_tensor('H', "g0", "g0"), make_tensor('t', "h0", "p0"), make_tensor('t', "p1", "h1"),
                       make_tensor('L', 'g0', 'h1'), make_tensor('L', 'h0', 'g0'), make_tensor('K', 'p0', 'p1')]
    a = Term(list_of_tensors, SecondQuantizedOperator.make_empty())
    a.simplify()
    assert a.list_of_tensors == [make_tensor('H', 'h6', 'h6'),
                                 make_tensor('t', 'p1', 'h3'), make_tensor('t', 'h4', 'p1'),
                                 make_tensor('L', 'h6', 'h3'), make_tensor('L', 'h4', 'h6')]
    assert a.sq_op == SecondQuantizedOperator.make_empty()
    assert a.indices_set == {Index(i) for i in ['h6', 'p1', 'h3', 'h4']}
    assert a.diagonal_indices == {Index('h6')}

    list_of_tensors = [make_tensor('H', "g0", "g0"), make_tensor('t', "h0", "p0"), make_tensor('t', "p1", "c1"),
                       make_tensor('L', 'g0', 'c1'), make_tensor('K', 'p0', 'p1')]
    a = Term(list_of_tensors, SecondQuantizedOperator('h0', 'g0'))
    a.simplify()
    assert a.list_of_tensors == [make_tensor('H', "c2", "c2"),
                                 make_tensor('t', "p1", "c2"), make_tensor('t', "h0", "p1")]
    assert a.sq_op == SecondQuantizedOperator('h0', 'c2')
    assert a.indices_set == {Index(i) for i in ['c2', 'p1', 'h0']}
    assert a.diagonal_indices == {Index('c2')}


def test_canonicalize_1():
    list_of_tensors = [make_tensor('H', "g0", "g0"), make_tensor('t', "h0", "p0"), make_tensor('t', "p1", "c1"),
                       make_tensor('L', 'g0', 'c1'), make_tensor('K', 'p0', 'p1')]
    a = Term(list_of_tensors, SecondQuantizedOperator('h0', 'g0'))
    ref = Term([make_tensor('H', 'c0', 'c0'), make_tensor('t', 'c0', 'p0'), make_tensor('t', 'h0', 'p0')],
               SecondQuantizedOperator('h0', 'c0'))
    assert a.canonicalize() == ref

    index_type = 'so'
    a = Term([make_tensor('H', "g0", "g0", index_type),
              make_tensor('t', "h0, a0", "p0, a0", index_type), make_tensor('t', "p1", "h1", index_type),
              make_tensor('L', 'g0, a0', 'h1, a0', index_type), make_tensor('L', 'h0', 'g0', index_type),
              make_tensor('K', 'p0', 'p1', index_type)],
             SecondQuantizedOperator.make_empty(index_type))
    ref = Term([make_tensor('H', "h0", "h0"), make_tensor('t', "h1", "p0"), make_tensor('t', "h2", "p0"),
                make_tensor('L', 'h0', 'h1'), make_tensor('L', 'h2', 'h0')],
               SecondQuantizedOperator.make_empty())
    # print(a.canonicalize_simple())
    # print(a.canonicalize_simple().canonicalize_simple())
    print(a.canonicalize())
    # # assert a.canonicalize() == ref


def test_problem():
    """
    -1/2 & H^{ v_{0} }_{ v_{1} } T^{ v_{1} }_{ c_{1} } T^{ v_{2} }_{ c_{0} } T^{ c_{0} c_{1} }_{ v_{0} v_{2} }
    -1/2 & H^{ v_{0} }_{ v_{1} } T^{ c_{0} }_{ v_{2} } T^{ c_{1} }_{ v_{0} } T^{ v_{1} v_{2} }_{ c_{0} c_{1} }
    """
    "1 H^{ v_{1} }_{ v_{0} } T^{ c_{0} }_{ v_{0} } T^{ c_{1} }_{ v_{2} } T^{ c_{1} c_{0} }_{ v_{1} v_{2} } a^{  }_{  }"
    indices_type = 'so'
    list_of_tensors = [Tensor.make_tensor('Hamiltonian', "v0", "v1", indices_type),
                       Tensor.make_tensor('cluster_amplitude', "v1", "c1", indices_type),
                       Tensor.make_tensor('cluster_amplitude', "v2", "c0", indices_type),
                       Tensor.make_tensor('cluster_amplitude', "c0,c1", "v0,v2", indices_type)]
    sq_op = SecondQuantizedOperator.make_empty(indices_type)
    a = Term(list_of_tensors, sq_op)

    list_of_tensors = [Tensor.make_tensor('Hamiltonian', "v0", "v1", indices_type),
                       Tensor.make_tensor('cluster_amplitude', "c0", "v2", indices_type),
                       Tensor.make_tensor('cluster_amplitude', "c1", "v0", indices_type),
                       Tensor.make_tensor('cluster_amplitude', "v1,v2", "c0,c1", indices_type)]
    b = Term(list_of_tensors, sq_op)

    ac = a.canonicalize()
    bc = b.canonicalize()
    print(ac)
    print(bc)
    print(ac == bc)

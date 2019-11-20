import pytest
from dsrg_generator.Index import Index
from dsrg_generator.Term import Term
from dsrg_generator.Tensor import Tensor
from dsrg_generator.SQOperator import SecondQuantizedOperator


make_tensor = Tensor.make_tensor
make_sq = SecondQuantizedOperator


def test_init_1():
    # coeff cannot be converted to float
    with pytest.raises(ValueError):
        assert Term([], [], 'a')


def test_init_2():
    # sq_op is not a SecondQuantizedOperator
    with pytest.raises(TypeError):
        assert Term([], [], '1')


def test_init_3():
    # inconsistent indices type
    with pytest.raises(TypeError):
        assert Term([make_tensor('H', 'g0', 'g1')], make_sq('g1', 'g0', 'si'))


def test_init_4():
    # indices not appear in pairs
    with pytest.raises(ValueError):
        assert Term([make_tensor('H', 'c0', 'g1'), make_tensor('t', 'c0', 't0')], make_sq('t0', 't1'))


def test_init_5():
    # OK to have diagonal indices, but attributes not fully functional
    a = Term([make_tensor('H', 'g0', 'g0')], make_sq("g0", "g0"))
    b = Term.from_term(a, flip_sign=True)
    assert a.coeff == -b.coeff
    b.coeff = 1
    assert a == b


def test_latex_1():
    list_of_tensors = [make_tensor('H', 'v0, v1', 'c0, c1'), make_tensor('t', 'c0, c1', 'v0, v1')]
    sq_op = SecondQuantizedOperator.make_empty()
    a = Term(list_of_tensors, sq_op, 0.25)
    assert a.latex() == "1/4 H^{ v_{0} v_{1} }_{ c_{0} c_{1} } T^{ c_{0} c_{1} }_{ v_{0} v_{1} }"
    assert a.latex(dollar=True) == "$1/4 H^{ v_{0} v_{1} }_{ c_{0} c_{1} } T^{ c_{0} c_{1} }_{ v_{0} v_{1} }$"


def test_latex_2():
    a = Term([make_tensor('H', 'g0, h1', 'g0, p1')],
             make_sq('g0, p1', 'g0, h1'))
    ref = "1/4 & {\\cal P} ( g_{0} / p_{1} ) {\\cal P} ( g_{0} / h_{1} ) " \
          "H^{ g_{0} h_{1} }_{ g_{0} p_{1} } a^{ g_{0} p_{1} }_{ g_{0} h_{1} }"
    assert a.latex(delimiter=True) == ref


def test_latex_3():
    list_of_tensors = [make_tensor('H', 'v3, v4', 'v0, c3'),
                       make_tensor('T', 'c0, c1', 'v1, v3'),
                       make_tensor('T', 'c2, c3', 'v2, v4')]
    sq_op = make_sq('v0, v1, v2', 'c0, c1, c2')
    a = Term(list_of_tensors, sq_op, -1)

    ref = "-1/18 & {\\cal P} ( v_{0} / v_{1} / v_{2} ) {\\cal P} ( c_{0} c_{1} / c_{2} ) " \
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


def test_comparision():
    # sequence: sq_op, number of tensors, tensors, absolute value of coeff, coeff
    a = Term([make_tensor('H', 'g0,h1', 'g1,p1')], make_sq('g1,p1', 'g0,h1'))
    b = Term([make_tensor('H', 'g0,h1', 'g1,p1'), make_tensor('t', 'h1', 'p1')], make_sq('g1', 'g0'))
    assert a > b

    b = Term([make_tensor('H', 'g0,h1,c0', 'g1,p1,v0'), make_tensor('t', 'c0', 'v0')], make_sq('g1,p1', 'g0,h1'))
    assert a < b

    c = Term([make_tensor('H', 'g0,v0,c0', 'g1,p1,c0'), make_tensor('t', 'h1', 'v0')], make_sq('g1,p1', 'g0,h1'))
    assert c > b
    assert c.list_of_tensors > b.list_of_tensors

    b = Term.from_term(a, flip_sign=True)
    c = Term.from_term(a)
    d = sorted([a, b, c])
    assert d[0] == b
    assert d[1] == d[2] == a == c


def test_void():
    a = Term([make_tensor('H', 'g0, h1', 'g0, p1')], make_sq('g0, p1', 'g0, h1'), 0.0)
    assert a.is_void()

    b = a.void()
    assert b.is_void()
    assert b is not a
    assert b != a

    c = Term.make_empty('so')
    assert b == c

    a.void_self()
    assert a == b


def test_perm_part_1():
    list_of_tensors = [make_tensor('H', 'g0, g1', 'g2, p0'), make_tensor('T', 'h0, h1', 'p0, p1')]
    sq_op = make_sq('g0, g1, p1', 'g2, h0, h1')
    a = Term(list_of_tensors, sq_op)
    cre_part, ann_part = a.perm_partition_open()
    assert cre_part == [[Index('g0'), Index('g1')], [Index('p1')]]
    assert ann_part == [[Index('g2')], [Index('h0'), Index('h1')]]


def test_perm_part_2():
    list_of_tensors = [make_tensor('H', 'v4, c2', 'v3, c1'),
                       make_tensor('T', 'c1', 'v0'),
                       make_tensor('T', 'c0, c3', 'v1, v4'),
                       make_tensor('T', 'c2, c3', 'v2, v3')]
    sq_op = make_sq('v2, c0', 'v0, v1')
    a = Term(list_of_tensors, sq_op)
    cre_part, ann_part = a.perm_partition_open()
    assert cre_part == [[Index('v2')], [Index('c0')]]
    assert ann_part == [[Index('v0')], [Index('v1')]]

    list_of_tensors = [make_tensor('H', 'v3, v4', 'v0, c3'),
                       make_tensor('T', 'c0, c1', 'v1, v3'),
                       make_tensor('T', 'c2, c3', 'v2, v4')]
    sq_op = make_sq('v0, v1, v2', 'c0, c1, c2')
    a = Term(list_of_tensors, sq_op)
    cre_part, ann_part = a.perm_partition_open()
    assert cre_part == [[Index('v0')], [Index('v1')], [Index('v2')]]
    assert ann_part == [[Index('c0'), Index('c1')], [Index('c2')]]


def test_perm_part_3():
    list_of_tensors = [make_tensor('H', 'g0, h1', 'p1, g0')]
    sq_op = make_sq('g0, p1', 'g0, h1')
    a = Term(list_of_tensors, sq_op, -1)
    cre_part, ann_part = a.perm_partition_open()
    assert cre_part == [[Index('g0')], [Index('p1')]]
    assert ann_part == [[Index('g0')], [Index('h1')]]

    list_of_tensors = [make_tensor('H', 'g0, h0, a0', 'g1, p0, p1'), make_tensor('t', 'p0, p1', 'a0,a1')]
    sq_op = make_sq('g1, a1', 'g0, h0')
    a = Term(list_of_tensors, sq_op, -1)
    cre_part, ann_part = a.perm_partition_open()
    assert cre_part == [[Index('g1')], [Index('a1')]]
    assert ann_part == [[Index('g0')], [Index('h0')]]


def test_perm_part_4():
    list_of_tensors = [make_tensor('H', 'c1', 'c0'), make_tensor('t', 'c1,a0', 'v0,v1')]
    sq_op = make_sq('v0,v1', 'c0,a0')
    a = Term(list_of_tensors, sq_op)
    cre_part_1, ann_part_1 = a.perm_partition_open()

    list_of_tensors = [make_tensor('H', 'a1', 'a0'), make_tensor('t', 'c0,a1', 'v0,v1')]
    sq_op = make_sq('v0,v1', 'c0,a0')
    a = Term(list_of_tensors, sq_op)
    cre_part_2, ann_part_2 = a.perm_partition_open()

    assert cre_part_1 == cre_part_2
    assert ann_part_1 == ann_part_2


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
    assert a.diagonal_indices == {Index('h6'): 4}

    list_of_tensors = [make_tensor('H', "g0", "g0"), make_tensor('t', "h0", "p0"), make_tensor('t', "p1", "c1"),
                       make_tensor('L', 'g0', 'c1'), make_tensor('K', 'p0', 'p1')]
    a = Term(list_of_tensors, SecondQuantizedOperator('h0', 'g0'))
    a.simplify()
    assert a.list_of_tensors == [make_tensor('H', "c2", "c2"),
                                 make_tensor('t', "p1", "c2"), make_tensor('t', "h0", "p1")]
    assert a.sq_op == SecondQuantizedOperator('h0', 'c2')
    assert a.indices_set == {Index(i) for i in ['c2', 'p1', 'h0']}
    assert a.diagonal_indices == {Index('c2'): 4}


def test_canonicalize_1():
    list_of_tensors = [make_tensor('H', "g0", "g0"), make_tensor('t', "h0", "p0"), make_tensor('t', "p1", "c1"),
                       make_tensor('L', 'g0', 'c1'), make_tensor('K', 'p0', 'p1')]
    a = Term(list_of_tensors, SecondQuantizedOperator('h0', 'g0'))
    with pytest.raises(NotImplementedError):
        assert a.canonicalize()


def test_canonicalize_2():
    # H^{ v_{0} }_{ v_{1} } T^{ v_{1} }_{ c_{1} } T^{ v_{2} }_{ c_{0} } T^{ c_{0} c_{1} }_{ v_{0} v_{2} }
    indices_type = 'so'
    list_of_tensors = [make_tensor('Hamiltonian', "v0", "v1", indices_type),
                       make_tensor('cluster_amplitude', "v1", "c1", indices_type),
                       make_tensor('cluster_amplitude', "v2", "c0", indices_type),
                       make_tensor('cluster_amplitude', "c0,c1", "v0,v2", indices_type)]
    sq_op = SecondQuantizedOperator.make_empty(indices_type)
    a = Term(list_of_tensors, sq_op)

    # H^{ v_{0} }_{ v_{1} } T^{ c_{0} }_{ v_{2} } T^{ c_{1} }_{ v_{0} } T^{ v_{1} v_{2} }_{ c_{0} c_{1} }
    list_of_tensors = [make_tensor('Hamiltonian', "v0", "v1", indices_type),
                       make_tensor('cluster_amplitude', "c0", "v2", indices_type),
                       make_tensor('cluster_amplitude', "c1", "v0", indices_type),
                       make_tensor('cluster_amplitude', "v1,v2", "c0,c1", indices_type)]
    b = Term(list_of_tensors, sq_op)

    ref = Term([make_tensor('Hamiltonian', "v1", "v0", indices_type),
                make_tensor('cluster_amplitude', "c0", "v0", indices_type),
                make_tensor('cluster_amplitude', "c1", "v2", indices_type),
                make_tensor('cluster_amplitude', "c0,c1", "v1,v2", indices_type)],
               sq_op, -1)
    assert ref == a.canonicalize() == b.canonicalize()


def test_canonicalize_3():
    # -1 * H^{g0,g1}_{g2,g3} * T^{h0,h1}_{p0,p1} * L^{p1}_{g2} * L^{p0}_{g3} * a^{g0,g1}_{h0, h1}
    list_of_tensors = [make_tensor('H', 'g0, g1', 'g2, g3'), make_tensor('t', 'p0, p1', 'h0, h1'),
                       make_tensor('L', 'p1', 'g2'), make_tensor('L', 'p0', 'g3')]
    sq_op = make_sq('g0, g1', 'h0, h1')
    a = Term(list_of_tensors, sq_op, -1)

    ref = Term([make_tensor('H', 'a0, a1', 'g0, g1'), make_tensor('t', 'a2, a3', 'h0, h1'),
                make_tensor('L', 'a2', 'a0'), make_tensor('L', 'a3', 'a1')], sq_op, 1)
    assert a.canonicalize() == ref

    list_of_tensors = [make_tensor('H', 'g0, G1', 'g2, G3', 'si'), make_tensor('t', 'P0, p1', 'H0, h1', 'si'),
                       make_tensor('L', 'p1', 'g2', 'si'), make_tensor('L', 'P0', 'G3', 'si')]
    a = Term(list_of_tensors, make_sq('g0, G1', 'H0, h1', 'si'))

    ref = Term([make_tensor('H', 'a0, A0', 'g0, G0', 'si'), make_tensor('t', 'a1, A1', 'h0, H0', 'si'),
                make_tensor('L', 'a1', 'a0', 'si'), make_tensor('L', 'A1', 'A0', 'si')],
               make_sq('g0, G0', 'h0, H0', 'si'), -1)
    assert ref == a.canonicalize()

    list_of_tensors = [make_tensor('H', 'g0, G1', 'g2, G3', 'sa'), make_tensor('t', 'P0, p1', 'H0, h1', 'sa'),
                       make_tensor('L', 'p1', 'g2', 'sa'), make_tensor('L', 'P0', 'G3', 'sa')]
    a = Term(list_of_tensors, make_sq('g0, G1', 'h1, H0', 'sa'))

    ref = Term([make_tensor('H', 'a0, A0', 'g0, G0', 'sa'), make_tensor('t', 'a1, A1', 'h0, H0', 'sa'),
                make_tensor('L', 'a1', 'a0', 'sa'), make_tensor('L', 'A1', 'A0', 'sa')],
               make_sq('g0, G0', 'h0, H0', 'sa'))
    assert ref == a.canonicalize()


def test_canonicalize_4():
    list_of_tensors = [make_tensor('Hamiltonian', "g0,g1,c0", "g2,p0,v0"),
                       make_tensor('cluster_amplitude', "p0,p1,g3", "a0,h1,a1"),
                       make_tensor('Kronecker', "v0", "p1"),
                       make_tensor('cumulant', "h1", "c0"), make_tensor('cumulant', "a1", "g3"),
                       make_tensor('cumulant', "g2,a0", "g0,g1")]
    a = Term(list_of_tensors, SecondQuantizedOperator.make_empty())

    ref = Term([make_tensor('H', "c0,a1,a2", "p0,v0,a0"), make_tensor('t', "c0,a4,a5", "p0,v0,a3"),
                make_tensor('L', "a4", "a3"), make_tensor('L', "a1,a2", "a0,a5")],
               SecondQuantizedOperator.make_empty())
    assert a.canonicalize() == ref


def test_generate_spin_cases_1():
    a = Term([make_tensor('H', 'g0,g1', 'g2,g3')], make_sq('g2,g3', 'g0,g1'))
    ref = {Term([make_tensor('H', 'g2,g3', 'g0,g1', 'si')], make_sq('g2,g3', 'g0,g1', 'si')),
           Term([make_tensor('H', 'g1,G1', 'g0,G0', 'si')], make_sq('g1,G1', 'g0,G0', 'si')),
           Term([make_tensor('H', 'G2,G3', 'G0,G1', 'si')], make_sq('G2,G3', 'G0,G1', 'si'))}
    for i in a.generate_spin_cases_naive():
        assert i in ref


def test_generate_spin_cases_2():
    from collections import defaultdict

    # -0.25 * H^{aw}_{xy} * T^{uv}_{az} * L^{xyz}_{uvw}
    a = Term([make_tensor('H', 'a1,a2', 'p0,a0'), make_tensor('t', 'a4,a5', 'p0,a3'),
              make_tensor('L', 'a1,a2,a3', 'a0,a4,a5')],
             SecondQuantizedOperator.make_empty(), -0.25)

    spin_combined = {}
    spin_coeff = defaultdict(list)
    for term in a.generate_spin_cases_naive():
        name = term.hash_term()
        spin_coeff[name].append(term.coeff)
        spin_combined[name] = term

    for name, term in spin_combined.items():
        term.coeff = sum(spin_coeff[name])
        if abs(term.coeff) < 1.0e-15:
            spin_combined.pop(name)

    ref = {Term([make_tensor('H', 'a1,a2', 'p0,a0', 'si'), make_tensor('t', 'a4,a5', 'p0,a3', 'si'),
                 make_tensor('L', 'a1,a2,a3', 'a0,a4,a5', 'si')], SecondQuantizedOperator.make_empty('si'), -0.25),
           Term([make_tensor('H', 'A1,A2', 'P0,A0', 'si'), make_tensor('t', 'A4,A5', 'P0,A3', 'si'),
                 make_tensor('L', 'A1,A2,A3', 'A0,A4,A5', 'si')], SecondQuantizedOperator.make_empty('si'), -0.25),
           Term([make_tensor('H', 'a1,A2', 'p0,A0', 'si'), make_tensor('t', 'a4,a5', 'p0,a3', 'si'),
                 make_tensor('L', 'a1,A2,a3', 'A0,a4,a5', 'si')], SecondQuantizedOperator.make_empty('si'),
                -0.5).canonicalize(),
           Term([make_tensor('H', 'a1,a2', 'p0,a0', 'si'), make_tensor('t', 'a4,A5', 'p0,A3', 'si'),
                 make_tensor('L', 'a1,a2,A3', 'a0,a4,A5', 'si')], SecondQuantizedOperator.make_empty('si'),
                -0.5).canonicalize(),
           Term([make_tensor('H', 'A1,a2', 'P0,a0', 'si'), make_tensor('t', 'A4,a5', 'P0,a3', 'si'),
                 make_tensor('L', 'A1,a2,a3', 'a0,A4,a5', 'si')], SecondQuantizedOperator.make_empty('si'),
                -1).canonicalize(),
           Term([make_tensor('H', 'a1,A2', 'P0,a0', 'si'), make_tensor('t', 'A4,A5', 'P0,A3', 'si'),
                 make_tensor('L', 'a1,A2,A3', 'a0,A4,A5', 'si')], SecondQuantizedOperator.make_empty('si'),
                -0.5).canonicalize(),
           Term([make_tensor('H', 'A1,A2', 'P0,A0', 'si'), make_tensor('t', 'A4,a5', 'P0,a3', 'si'),
                 make_tensor('L', 'A1,A2,a3', 'A0,A4,a5', 'si')], SecondQuantizedOperator.make_empty('si'),
                -0.5).canonicalize(),
           Term([make_tensor('H', 'a1,A2', 'p0,A0', 'si'), make_tensor('t', 'a4,A5', 'p0,A3', 'si'),
                 make_tensor('L', 'a1,A2,A3', 'A0,a4,A5', 'si')], SecondQuantizedOperator.make_empty('si'),
                -1).canonicalize()}

    for i in spin_combined.values():
        assert i in ref
    assert len(spin_combined) == len(ref)


def test_generate_spin_cases_3():
    list_of_tensors = [make_tensor('H', "v0,c0", "v1,c1"), make_tensor('t', "v1", "c2"),
                       make_tensor('t', "v2,v3", "c0,c3"), make_tensor('t', "c2,c3", "v0,v3")]
    a = Term(list_of_tensors, make_sq("c1", "v2"))

    ac = Term([make_tensor('H', 'v2,c1', 'v1,c0'), make_tensor('t', 'c2', 'v1'),
               make_tensor('t', 'c1,c3', 'v0,v3'), make_tensor('t', 'c2,c3', 'v2,v3')],
              make_sq("c0", "v0"))
    assert a.canonicalize() == ac

    i_type = 'si'
    ref = [Term([make_tensor('H', 'v2,c1', 'v1,c0', i_type),
                 make_tensor('t', 'c2', 'v1', i_type),
                 make_tensor('t', 'c1,c3', 'v0,v3', i_type),
                 make_tensor('t', 'c2,c3', 'v2,v3', i_type)],
                make_sq("c0", "v0", i_type)),
           Term([make_tensor('H', 'v2,c1', 'v1,c0', i_type),
                 make_tensor('t', 'c2', 'v1', i_type),
                 make_tensor('t', 'c1,C3', 'v0,V3', i_type),
                 make_tensor('t', 'c2,C3', 'v2,V3', i_type)],
                make_sq("c0", "v0", i_type)).canonicalize(),
           Term([make_tensor('H', 'v2,C1', 'V1,c0', i_type),
                 make_tensor('t', 'C2', 'V1', i_type),
                 make_tensor('t', 'C1,c3', 'v0,V3', i_type),
                 make_tensor('t', 'C2,c3', 'v2,V3', i_type)],
                make_sq("c0", "v0", i_type)).canonicalize(),
           Term([make_tensor('H', 'V2,c1', 'V1,c0', i_type),
                 make_tensor('t', 'C2', 'V1', i_type),
                 make_tensor('t', 'c1,c3', 'v0,v3', i_type),
                 make_tensor('t', 'C2,c3', 'V2,v3', i_type)],
                make_sq("c0", "v0", i_type)).canonicalize(),
           Term([make_tensor('H', 'V2,c1', 'V1,c0', i_type),
                 make_tensor('t', 'C2', 'V1', i_type),
                 make_tensor('t', 'c1,C3', 'v0,V3', i_type),
                 make_tensor('t', 'C2,C3', 'V2,V3', i_type)],
                make_sq("c0", "v0", i_type)).canonicalize(),
           Term([make_tensor('H', 'V2,C1', 'V1,C0', i_type),
                 make_tensor('t', 'C2', 'V1', i_type),
                 make_tensor('t', 'C1,C3', 'V0,V3', i_type),
                 make_tensor('t', 'C2,C3', 'V2,V3', i_type)],
                make_sq("C0", "V0", i_type)),
           Term([make_tensor('H', 'V2,C1', 'V1,C0', i_type),
                 make_tensor('t', 'C2', 'V1', i_type),
                 make_tensor('t', 'C1,c3', 'V0,v3', i_type),
                 make_tensor('t', 'C2,c3', 'V2,v3', i_type)],
                make_sq("C0", "V0", i_type)).canonicalize(),
           Term([make_tensor('H', 'V2,c1', 'v1,C0', i_type),
                 make_tensor('t', 'c2', 'v1', i_type),
                 make_tensor('t', 'c1,C3', 'V0,v3', i_type),
                 make_tensor('t', 'c2,C3', 'V2,v3', i_type)],
                make_sq("C0", "V0", i_type)).canonicalize(),
           Term([make_tensor('H', 'v2,C1', 'v1,C0', i_type),
                 make_tensor('t', 'c2', 'v1', i_type),
                 make_tensor('t', 'C1,c3', 'V0,v3', i_type),
                 make_tensor('t', 'c2,c3', 'v2,v3', i_type)],
                make_sq("C0", "V0", i_type)).canonicalize(),
           Term([make_tensor('H', 'v2,C1', 'v1,C0', i_type),
                 make_tensor('t', 'c2', 'v1', i_type),
                 make_tensor('t', 'C1,C3', 'V0,V3', i_type),
                 make_tensor('t', 'c2,C3', 'v2,V3', i_type)],
                make_sq("C0", "V0", i_type)).canonicalize()
           ]

    count = 0
    for i in ac.generate_spin_cases_naive():
        assert i in ref
        count += 1
    assert count == len(ref)


def test_generate_spin_cases_4():
    # a = Term([make_tensor('H', 'v2,c0', 'v0,c2'), make_tensor('t', 'c1,c2', 'v1,v2')],
    #          make_sq('v0,v1', 'c0,c1'))
    # print(a)
    # print(a.latex())
    # print(a.ambit(), '\n')
    #
    # for i in a.generate_spin_cases_naive():
    #     print(i.latex())
    # print()

    a = Term([make_tensor('H', 'v2,v3', 'v0,v1'), make_tensor('t', 'c0,c1', 'v2,v3')],
             make_sq('v0,v1', 'c0,c1'), 0.125)
    print(a)
    print(a.latex(), '\n')

    for i in a.generate_spin_cases_naive():
        print(i.latex())
        print(i.ambit())

    a = Term([make_tensor('H', 'p0,g0', 'c0,g1'), make_tensor('t', 'c0,h0', 'p0,p1')],
             make_sq('g0,h0', 'g1,p1'))
    print(a)
    print(a.latex(), '\n')

    for i in a.generate_spin_cases_naive():
        print(i.latex())
        print(i.ambit())

    # a = Term([make_tensor('H', 'c0,c1', 'v0,c3'), make_tensor('t', 'c2,c3', 'v1,v2')],
    #          make_sq('c0,c1,c2', 'v0,v1,v2'), 1.0/4)
    # print(a)
    # print(a.latex(), '\n')
    #
    # from dsrg_generator.phys_op_contraction import combine_terms, print_terms_ambit
    # b = list(a.generate_spin_cases_naive())
    # print_terms_ambit(b)
    # c = combine_terms(b)
    # print_terms_ambit(c)
    # for i in :
    #     print(i.latex())
    #     print(i.ambit())


def test_make_ddca():
    pass


def test_make_excitation():
    pass


def test_make_one_body():
    pass


def test_contraction_paths():
    t = Term([make_tensor('H', 'c0,v0', 'v1,v2', 'so'),
              make_tensor('t', 'c1,c2', 'v2,v3', 'so'),
              make_tensor('t', 'c3,c1,c2', 'v1,v4,v5', 'so'),
              make_tensor('t', 'c3,c5,c6', 'v6,v4,v5', 'so')],
             make_sq("v0,v3,v6", "c0,c5,c6", 'so')).canonicalize()
    for i in t.contraction_paths():
        print(i)
    # print(t.optimal_contraction_cost())
    #
    # t = Term([make_tensor('H', 'v3,v4', 'v0,c3', 'so'),
    #           make_tensor('t', 'c4,c5', 'v5,v6', 'so'),
    #           make_tensor('t', 'c0,c1,c2', 'v4,v5,v6', 'so'),
    #           make_tensor('t', 'c3,c4,c5', 'v1,v2,v3', 'so')],
    #          make_sq("v0,v1,v2", "c0,c1,c2", 'so')).canonicalize()
    # print(t.optimal_contraction_cost())

    t = Term([make_tensor('H', 'v4,v5', 'v2,v3', 'so'),
              make_tensor('t', 'c1,c2', 'v0,v3', 'so'),
              make_tensor('t', 'c0,c3', 'v1,v6', 'so'),
              make_tensor('t', 'c1,c2,c3', 'v4,v5,v6', 'so')],
             make_sq("v0,v1", "c0,v2", 'so'))
    print()
    for i in t.contraction_paths():
        print(i)

    t = Term([make_tensor('H', 'c4,c5', 'v3,c3', 'so'),
              make_tensor('t', 'c1,c2', 'v3,v5', 'so'),
              make_tensor('t', 'c0,c6', 'v0,v4', 'so'),
              make_tensor('t', 'c4,c5,c7', 'v1,v2,v6', 'so'),
              make_tensor('t', 'c3,c6,c7', 'v4,v5,v6', 'so')],
             make_sq("v0,v1,v2", "c0,c1,c2", 'so'))
    print()
    print(t.optimal_contraction_cost())
    # for i in t.contraction_paths():
    #     print(i)

    # simply reverse the h-h line to v-v line
    t = Term([make_tensor('H', 'c4,v9', 'v3,v8', 'so'),
              make_tensor('t', 'c1,c8', 'v3,v2', 'so'),
              make_tensor('t', 'c0,c6', 'v0,v4', 'so'),
              make_tensor('t', 'c4,c2,c7', 'v1,v9,v6', 'so'),
              make_tensor('t', 'c8,c6,c7', 'v4,v8,v6', 'so')],
             make_sq("v0,v1,v2", "c0,c1,c2", 'so'))
    print()
    print(t.optimal_contraction_cost())

    t = Term([make_tensor('H', 'c4,c5', 'v3,c3', 'so')],
             make_sq('c4,c5', 'v3,c3', 'so'))
    print()
    print(t.optimal_contraction_cost())

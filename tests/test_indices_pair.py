import pytest
from sympy.combinatorics import Permutation
from sympy.combinatorics.tensor_can import riemann_bsgs

from dsrg_generator.Indices import Indices
from dsrg_generator.IndicesPair import IndicesPair


def test_init():
    with pytest.raises(TypeError):
        IndicesPair(Indices.make_indices("c0", 'spin-adapted'),
                    Indices.make_indices("V0, h2", 'spin-integrated'))

    a = IndicesPair("A0, a1", "p0, G8", 'spin-integrated')
    assert a.upper_indices == Indices.make_indices("A0, a1", 'spin-integrated')
    assert a.lower_indices == Indices.make_indices("p0, G8", 'spin-integrated')
    assert a.n_upper == 2
    assert a.n_lower == 2
    assert not a.is_empty()


def test_eq():
    a = IndicesPair("h1,g0,v4", "c0")
    assert a == IndicesPair(Indices.make_indices("h1, g0, v4", 'so'),
                            Indices.make_indices("c0", 'so'))
    with pytest.raises(TypeError):
        assert a == IndicesPair(Indices.make_indices("h1,g0,v4", 'spin-integrated'),
                                Indices.make_indices("c0", 'spin-integrated'))


def test_ne():
    a = IndicesPair("v1,G9", "h2,a2", 'spin-adapted')
    assert a != IndicesPair(Indices.make_indices("h1,g0,v4", 'sa'),
                            Indices.make_indices("c0", 'sa'))


def test_lt():
    a = IndicesPair("G3", "g0, v2", 'spin-integrated')
    assert a < IndicesPair(Indices.make_indices("A0", 'spin-integrated'),
                           Indices.make_indices("c0", 'spin-integrated'))


def test_le():
    a = IndicesPair("G3", "g0, v2", 'spin-adapted')
    assert a <= IndicesPair(Indices.make_indices("A0", 'sa'),
                            Indices.make_indices("c0", 'sa'))
    assert a <= IndicesPair(Indices.make_indices("G3", 'spin-adapted'),
                            Indices.make_indices("g0,v2", 'spin-adapted'))


def test_gt():
    a = IndicesPair("g3, p0", "h2")
    assert a > IndicesPair("g3, p0", "g4", 'spin-orbital')


def test_ge():
    a = IndicesPair("p0, p1", "g0, h2")
    assert a >= IndicesPair("p0, p1", "g0, h2", 'so')
    assert a >= IndicesPair("p0, p1", "g0, h1", 'so')
    assert a >= IndicesPair("p0, p1", "a8", 'so')


def test_latex():
    a = IndicesPair("p0, p1", "g0, h2", 'spin-orbital')
    assert a.latex() == "^{ p_{0} p_{1} }_{ g_{0} h_{2} }"


def test_ambit():
    a = IndicesPair("p0, p1", "g0, h2", 'spin-orbital')
    assert a.ambit() == '["p0,p1,g0,h2"]'


def test_diagonal_indices():
    a = IndicesPair("p0, p1", "g0, h2", 'spin-orbital')
    assert a.diagonal_indices() == set()

    a = IndicesPair("g0, p1", "g0, p1", 'spin-orbital')
    assert a.diagonal_indices() == a.upper_indices.indices_set


def test_canonicalize():
    a = IndicesPair("G2, p0, p1", "g0, A0, h2", 'spin-integrated')
    c, sign = a.canonicalize()
    assert c == IndicesPair("p0, p1, G2", "g0, h2, A0", 'spin-integrated')
    assert sign == -1


def test_generate_spin_cases():
    so, si = 'spin-orbital', 'spin-integrated'
    a = IndicesPair("g0, g1, p0", "h0, a1, g2", so)
    ref = [IndicesPair("g0, g1, p0", "h0, a1, g2", si),
           IndicesPair("G0, G1, P0", "H0, A1, G2", si),
           IndicesPair("G0, g1, p0", "H0, a1, g2", si),
           IndicesPair("G0, g1, p0", "h0, A1, g2", si),
           IndicesPair("G0, g1, p0", "h0, a1, G2", si),
           IndicesPair("g0, G1, p0", "H0, a1, g2", si),
           IndicesPair("g0, G1, p0", "h0, A1, g2", si),
           IndicesPair("g0, G1, p0", "h0, a1, G2", si),
           IndicesPair("g0, g1, P0", "H0, a1, g2", si),
           IndicesPair("g0, g1, P0", "h0, A1, g2", si),
           IndicesPair("g0, g1, P0", "h0, a1, G2", si),
           IndicesPair("G0, G1, p0", "H0, A1, g2", si),
           IndicesPair("G0, G1, p0", "H0, a1, G2", si),
           IndicesPair("G0, G1, p0", "h0, A1, G2", si),
           IndicesPair("g0, G1, P0", "H0, A1, g2", si),
           IndicesPair("g0, G1, P0", "H0, a1, G2", si),
           IndicesPair("g0, G1, P0", "h0, A1, G2", si),
           IndicesPair("G0, g1, P0", "H0, A1, g2", si),
           IndicesPair("G0, g1, P0", "H0, a1, G2", si),
           IndicesPair("G0, g1, P0", "h0, A1, G2", si)
           ]
    count = 0
    for indices_pair in a.generate_spin_cases():
        count += 1
        assert indices_pair in ref
    assert count == len(ref)

    a = IndicesPair("a0, g0, h2", "v1", so)
    ref = [IndicesPair("a0, g0, h2", "v1", si), IndicesPair("a0, g0, h2", "V1", si),
           IndicesPair("A0, g0, h2", "v1", si), IndicesPair("A0, g0, h2", "V1", si),
           IndicesPair("a0, G0, h2", "v1", si), IndicesPair("a0, G0, h2", "V1", si),
           IndicesPair("a0, g0, H2", "v1", si), IndicesPair("a0, g0, H2", "V1", si),
           IndicesPair("A0, G0, h2", "v1", si), IndicesPair("A0, G0, h2", "V1", si),
           IndicesPair("A0, g0, H2", "v1", si), IndicesPair("A0, g0, H2", "V1", si),
           IndicesPair("a0, G0, H2", "v1", si), IndicesPair("a0, G0, H2", "V1", si),
           IndicesPair("A0, G0, H2", "v1", si), IndicesPair("A0, G0, H2", "V1", si)]
    count = 0
    for indices_pair in a.generate_spin_cases(particle_conserving=False):
        count += 1
        assert indices_pair in ref
    assert count == len(ref)


def test_base_strong_generating_set_1():
    assert IndicesPair("p0", "g0").asym_bsgs(False) == IndicesPair("p0", "g0", "sa").sym_bsgs(False)
    assert IndicesPair("p0", "g0").asym_bsgs(True) == IndicesPair("p0", "g0", "sa").sym_bsgs(True)


def test_base_strong_generating_set_2():
    a = IndicesPair("p0, p1", "g0", 'sa')
    with pytest.raises(ValueError):
        a.base_strong_generating_set(False)

    a = IndicesPair("g0, p1", "g0, h2", 'so')
    assert a.base_strong_generating_set(True) == riemann_bsgs


def test_base_strong_generating_set_3():
    a = IndicesPair("p0, p1", "g0, h2", 'sa')
    sym2 = ([0], [Permutation(5)(0, 1)(2, 3)])
    assert a.base_strong_generating_set(False) == sym2
    sym2 = ([0], [Permutation(5)(0, 1)(2, 3), Permutation(5)(0, 2)(1, 3)])
    assert a.base_strong_generating_set(True) == sym2


def test_base_strong_generating_set_4():
    a = IndicesPair("p0, p1", "g0, h2", 'so')
    asym2 = ([0, 2], [Permutation(0, 1)(4, 5), Permutation(2, 3)(4, 5)])
    assert a.base_strong_generating_set(False) == asym2
    asym2 = ([0, 2], [Permutation(0, 1)(4, 5), Permutation(2, 3)(4, 5), Permutation(5)(0, 2)(1, 3)])
    assert a.base_strong_generating_set(True) == asym2 == riemann_bsgs


def test_base_strong_generating_set_5():
    a = IndicesPair("p0, p1, v0", "g0, h2, c1", 'sa')
    sym3 = ([0, 1], [Permutation(7)(0, 1)(3, 4), Permutation(7)(1, 2)(4, 5)])
    assert a.base_strong_generating_set(False) == sym3
    sym3 = ([0, 1], [Permutation(7)(0, 1)(3, 4), Permutation(7)(1, 2)(4, 5),
                     Permutation(7)(0, 3)(1, 4)(2, 5)])
    assert a.base_strong_generating_set(True) == sym3

    a = IndicesPair("p0, p1, v0", "g0, h2, c1", 'si')
    asym3 = ([0, 1, 3, 4], [Permutation(0, 1)(6, 7), Permutation(1, 2)(6, 7),
                            Permutation(3, 4)(6, 7), Permutation(4, 5)(6, 7)])
    assert a.base_strong_generating_set(False) == asym3
    asym3 = ([0, 1, 3, 4], [Permutation(0, 1)(6, 7), Permutation(1, 2)(6, 7), Permutation(3, 4)(6, 7),
                            Permutation(4, 5)(6, 7), Permutation(7)(0, 3)(1, 4)(2, 5)])
    assert a.base_strong_generating_set(True) == asym3


def test_singlet_adaptation():
    from dsrg_generator.Index import Index
    a = IndicesPair("P0,P1", "G0,P2", 'si')
    replacement = {Index("P0"): Index("p0"),
                   Index("P1"): Index("p2"),
                   Index("G0"): Index("g9"),
                   Index("P2"): Index("p8")}
    b = list(a.generate_singlet_adaptation(replacement))
    print(b)

    a = IndicesPair("h0,P0,P1", "a1,G0,P2", 'si')
    replacement = {Index("P0"): Index("p0"),
                   Index("P1"): Index("p2"),
                   Index("G0"): Index("g9"),
                   Index("P2"): Index("p8"),
                   Index("h0"): Index("h0"),
                   Index("a1"): Index("a1")}
    b = list(a.generate_singlet_adaptation(replacement))
    print(b)

    a = IndicesPair("", "", 'si')
    b = list(a.generate_singlet_adaptation({}))
    print(b)


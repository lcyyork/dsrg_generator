import pytest
from Indices import Indices
from IndicesPair import init_indices_pair, IndicesPair


def test_init():
    with pytest.raises(TypeError):
        IndicesPair(Indices.make_indices("c0", 'spin-adapted'),
                    Indices.make_indices("V0, h2", 'spin-integrated'))

    a = init_indices_pair("A0, a1", "p0, G8", 'spin-integrated')
    assert a.upper_indices == Indices.make_indices("A0, a1", 'spin-integrated')
    assert a.lower_indices == Indices.make_indices("p0, G8", 'spin-integrated')
    assert a.n_upper == 2
    assert a.n_lower == 2
    assert not a.is_empty


def test_eq():
    id_type = 'spin-orbital'
    a = init_indices_pair("h1,g0,v4", "c0", id_type)
    assert a == IndicesPair(Indices.make_indices("h1, g0, v4", id_type),
                            Indices.make_indices("c0", id_type))
    with pytest.raises(TypeError):
        assert a == IndicesPair(Indices.make_indices("h1,g0,v4", 'spin-integrated'),
                                Indices.make_indices("c0", 'spin-integrated'))


def test_ne():
    a = init_indices_pair("v1,G9", "h2,a2", 'spin-adapted')
    assert a != IndicesPair(Indices.make_indices("h1,g0,v4", 'spin-adapted'),
                            Indices.make_indices("c0", 'spin-adapted'))


def test_lt():
    a = init_indices_pair("G3", "g0, v2", 'spin-integrated')
    assert a < IndicesPair(Indices.make_indices("A0", 'spin-integrated'),
                           Indices.make_indices("c0", 'spin-integrated'))


def test_le():
    a = init_indices_pair("G3", "g0, v2", 'spin-adapted')
    assert a <= IndicesPair(Indices.make_indices("A0", 'spin-adapted'),
                           Indices.make_indices("c0", 'spin-adapted'))
    assert a <= IndicesPair(Indices.make_indices("G3", 'spin-adapted'),
                            Indices.make_indices("g0,v2", 'spin-adapted'))


def test_gt():
    a = init_indices_pair("g3, p0", "h2", 'spin-orbital')
    assert a > init_indices_pair("g3, p0", "g4", 'spin-orbital')


def test_ge():
    a = init_indices_pair("p0, p1", "g0, h2", 'spin-orbital')
    assert a >= init_indices_pair("p0, p1", "g0, h2", 'spin-orbital')
    assert a >= init_indices_pair("p0, p1", "g0, h1", 'spin-orbital')
    assert a >= init_indices_pair("p0, p1", "a8", 'spin-orbital')


def test_latex():
    a = init_indices_pair("p0, p1", "g0, h2", 'spin-orbital')
    assert a.latex() == "^{ p_{0} p_{1} }_{ g_{0} h_{2} }"


def test_ambit():
    a = init_indices_pair("p0, p1", "g0, h2", 'spin-orbital')
    assert a.ambit() == '["p0,p1,g0,h2"]'


def test_canonicalize():
    a = init_indices_pair("G2, p0, p1", "g0, A0, h2", 'spin-integrated')
    c, sign = a.canonicalize()
    assert c == init_indices_pair("p0, p1, G2", "g0, h2, A0", 'spin-integrated')
    assert sign == -1


def test_generate_spin_cases():
    so, si = 'spin-orbital', 'spin-integrated'
    a = init_indices_pair("g0, g1, p0", "h0, a1, g2", so)
    ref = [init_indices_pair("g0, g1, p0", "h0, a1, g2", si),
           init_indices_pair("G0, G1, P0", "H0, A1, G2", si),
           init_indices_pair("G0, g1, p0", "H0, a1, g2", si),
           init_indices_pair("G0, g1, p0", "h0, A1, g2", si),
           init_indices_pair("G0, g1, p0", "h0, a1, G2", si),
           init_indices_pair("g0, G1, p0", "H0, a1, g2", si),
           init_indices_pair("g0, G1, p0", "h0, A1, g2", si),
           init_indices_pair("g0, G1, p0", "h0, a1, G2", si),
           init_indices_pair("g0, g1, P0", "H0, a1, g2", si),
           init_indices_pair("g0, g1, P0", "h0, A1, g2", si),
           init_indices_pair("g0, g1, P0", "h0, a1, G2", si),
           init_indices_pair("G0, G1, p0", "H0, A1, g2", si),
           init_indices_pair("G0, G1, p0", "H0, a1, G2", si),
           init_indices_pair("G0, G1, p0", "h0, A1, G2", si),
           init_indices_pair("g0, G1, P0", "H0, A1, g2", si),
           init_indices_pair("g0, G1, P0", "H0, a1, G2", si),
           init_indices_pair("g0, G1, P0", "h0, A1, G2", si),
           init_indices_pair("G0, g1, P0", "H0, A1, g2", si),
           init_indices_pair("G0, g1, P0", "H0, a1, G2", si),
           init_indices_pair("G0, g1, P0", "h0, A1, G2", si)
           ]
    count = 0
    for indices_pair in a.generate_spin_cases():
        count += 1
        assert indices_pair in ref
    assert count == len(ref)

    a = init_indices_pair("a0, g0, h2", "v1", so)
    ref = [init_indices_pair("a0, g0, h2", "v1", si), init_indices_pair("a0, g0, h2", "V1", si),
           init_indices_pair("A0, g0, h2", "v1", si), init_indices_pair("A0, g0, h2", "V1", si),
           init_indices_pair("a0, G0, h2", "v1", si), init_indices_pair("a0, G0, h2", "V1", si),
           init_indices_pair("a0, g0, H2", "v1", si), init_indices_pair("a0, g0, H2", "V1", si),
           init_indices_pair("A0, G0, h2", "v1", si), init_indices_pair("A0, G0, h2", "V1", si),
           init_indices_pair("A0, g0, H2", "v1", si), init_indices_pair("A0, g0, H2", "V1", si),
           init_indices_pair("a0, G0, H2", "v1", si), init_indices_pair("a0, G0, H2", "V1", si),
           init_indices_pair("A0, G0, H2", "v1", si), init_indices_pair("A0, G0, H2", "V1", si)]
    count = 0
    for indices_pair in a.generate_spin_cases(particle_conserving=False):
        count += 1
        assert indices_pair in ref
    assert count == len(ref)

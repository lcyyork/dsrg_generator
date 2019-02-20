import pytest
from Indices import Indices
from IndicesPair import make_indices_pair
from SQOperator import make_sqop, SecondQuantizedOperator


def test_init():
    id_type = 'spin-orbital'
    assert make_sqop([], [], id_type).is_empty()
    a = make_sqop("g0,g1,g2", "p0,p1,p2", id_type)
    assert a.indices_pair == make_indices_pair("g0,g1,g2", "p0,p1,p2", id_type)
    assert a.cre_ops == Indices.make_indices("g0,g1,g2", id_type)
    assert a.ann_ops == Indices.make_indices("p0,p1,p2", id_type)
    assert a.n_ann == 3
    assert a.n_cre == 3


def test_eq():
    a = make_sqop("g0,g1,g2", "p0,p1,p2", 'spin-orbital')
    assert a == SecondQuantizedOperator(make_indices_pair("g0,g1,g2", "p0,p1,p2", 'spin-orbital'))
    with pytest.raises(TypeError):
        assert a == make_sqop("g0,g1,g2", "p0,p1,p2", 'spin-integrated')


def test_ne():
    a = make_sqop("g0,g1,g2", "p0,p1,p2", 'spin-orbital')
    assert a != make_sqop(["g0", "v1", "p2"], ["p0", "a1", "p2"], 'spin-orbital')


def test_lt():
    a = make_sqop("g0,g1,g2", "p0,P1,p2", 'spin-integrated')
    assert a < make_sqop("g0,g1,g2", "p0,P1,p2,p3", 'spin-integrated')
    assert a < make_sqop("g0,g1,g2", "p0,P2,p2", 'spin-integrated')


def test_ge():
    a = make_sqop("G0,G1,g2", "p0,P1,P2", 'spin-integrated')
    assert a >= make_sqop("g0,g1,g2", "p0,P1,p2,p3", 'spin-integrated')
    with pytest.raises(TypeError):
        assert a >= 1


def test_exist_permute_format():
    a = make_sqop("g0,g1,g2", "p0,p1,p2", 'spin-orbital')
    assert not a.exist_permute_format()

    a = make_sqop("g0,g1,g2", "p0,p1,p2", 'spin-integrated')
    assert not a.exist_permute_format()

    a = make_sqop("g0,g1,g2", "p0,P1,p2", 'spin-integrated')
    assert not a.exist_permute_format()

    a = make_sqop("g0,v1,p2", "p0,a1,c2", 'spin-orbital')
    assert a.exist_permute_format()

    a = make_sqop("g0,v1,p2", "p0,a1,c2", 'spin-integrated')
    assert a.exist_permute_format()


def test_n_multiset_permutation():
    a = make_sqop("g0,g1,g2", "p0,p1,p2", 'spin-orbital')
    assert a.n_multiset_permutation() == 1

    a = make_sqop("g0,g1,c2", "p0,p1,p2", 'spin-integrated')
    assert a.n_multiset_permutation() == 3

    a = make_sqop("g0,g1,g2", "p0,P1,p2", 'spin-integrated')
    assert a.n_multiset_permutation() == 1

    a = make_sqop("g0,v1,p2", "p0,a1,p2", 'spin-orbital')
    assert a.n_multiset_permutation() == 18

    a = make_sqop("g0,v1,p2", "p0,a1,p2", 'spin-integrated')
    assert a.n_multiset_permutation() == 18


def test_latex_permute_format():
    n_perm, perm, latex_str = make_sqop("g0,g1,g2", "p0,p1,p2", 'spin-orbital').latex_permute_format()
    assert n_perm == 1
    assert perm == ''
    assert latex_str == 'a^{ g_{0} g_{1} g_{2} }_{ p_{0} p_{1} p_{2} }'

    n_perm, perm, latex_str = make_sqop("g0,v1,p2", "p0,a1,p2", 'spin-integrated').latex_permute_format()
    assert n_perm == 18
    assert perm == '{\\cal P}(g_{0} / p_{2} / v_{1}) {\\cal P}(p_{0} p_{2} / a_{1})'
    assert latex_str == 'a^{ g_{0} v_{1} p_{2} }_{ p_{0} a_{1} p_{2} }'

    n_perm, perm, latex_str = make_sqop("g0,v1,p2", "p0,a1,P2", 'spin-integrated').latex_permute_format()
    assert n_perm == 6
    assert perm == '{\\cal P}(g_{0} / p_{2} / v_{1})'
    assert latex_str == 'a^{ g_{0} v_{1} p_{2} }_{ p_{0} a_{1} P_{2} }'

    n_perm, perm, latex_str = make_sqop("g0,V1,p2", "p0,a1,P2", 'spin-integrated').latex_permute_format()
    assert n_perm == 1
    assert perm == ''
    assert latex_str == 'a^{ g_{0} V_{1} p_{2} }_{ p_{0} a_{1} P_{2} }'


def test_ambit_permute_format():
    a = make_sqop([], [], 'spin-orbital')
    for pair in a.ambit_permute_format():
        assert pair == (1, '')

    a = make_sqop("g0,g1,c2", "p0,p1,p2", 'spin-integrated')
    ref = {(1, '["p0,p1,p2,g0,g1,c2"]'), (-1, '["p0,p1,p2,g0,c2,g1"]'), (1, '["p0,p1,p2,c2,g0,g1"]')}
    for pair in a.ambit_permute_format(cre_first=False):
        assert pair in ref
        ref.remove(pair)
    assert len(ref) == 0

    a = make_sqop("g0,G1,c2", "p0,p1,P2", 'spin-integrated')
    for pair in a.ambit_permute_format(cre_first=False):
        assert pair == (1, '["p0,p1,P2,g0,G1,c2"]')

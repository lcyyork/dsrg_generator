import pytest
from dsrg_generator.Index import Index
from dsrg_generator.Indices import Indices
from dsrg_generator.IndicesPair import IndicesPair
from dsrg_generator.SQOperator import SecondQuantizedOperator


def test_init():
    id_type = 'spin-orbital'
    assert SecondQuantizedOperator([], []).is_empty()
    a = SecondQuantizedOperator("g0,g1,g2", "p0,p1,p2", id_type)
    assert a.cre_ops == Indices.make_indices("g0,g1,g2", id_type)
    assert a.ann_ops == Indices.make_indices("p0,p1,p2", id_type)
    assert a.n_ann == 3
    assert a.n_cre == 3


def test_eq():
    a = SecondQuantizedOperator("g0,g1,g2", "p0,p1,p2", 'spin-orbital')
    b = SecondQuantizedOperator("g0,g1,g2", "p0,p1,p2", 'so')
    assert a is not b
    assert a == b

    with pytest.raises(TypeError):
        assert a == IndicesPair("g0,g1,g2", "p0,p1,p2", 'spin-orbital')
    with pytest.raises(TypeError):
        assert a == SecondQuantizedOperator("g0,g1,g2", "p0,p1,p2", 'spin-integrated')


def test_ne():
    a = SecondQuantizedOperator("g0,g1,g2", "p0,p1,p2", 'so')
    assert a != SecondQuantizedOperator(["g0", "v1", "p2"], ["p0", "a1", "p2"])


def test_lt():
    a = SecondQuantizedOperator("g0,g1,g2", "p0,P1,p2", 'spin-integrated')
    assert a < SecondQuantizedOperator("g0,g1,g2", "p0,P1,p2,p3", 'si')
    assert a < SecondQuantizedOperator("g0,g1,g2", "p0,P2,p2", 'si')


def test_ge():
    a = SecondQuantizedOperator("G0,G1,g2", "p0,P1,P2", 'spin-integrated')
    assert a >= SecondQuantizedOperator("g0,g1,g2", "p0,P1,p2,p3", 'si')
    with pytest.raises(TypeError):
        assert a >= 1


def test_is_particle_conserving():
    assert SecondQuantizedOperator("G0,G1,g2", "p0,P1,P2", 'spin-integrated').is_particle_conserving()
    assert SecondQuantizedOperator("p0", "p0", 'so').is_particle_conserving()
    assert not SecondQuantizedOperator("p0", "p0, p1", 'so').is_particle_conserving()


def test_is_spin_conserving():
    assert SecondQuantizedOperator("p0", "p0", 'si').is_spin_conserving()
    assert not SecondQuantizedOperator("p0", "H0", 'si').is_spin_conserving()
    with pytest.raises(ValueError):
        assert SecondQuantizedOperator("p0", "p0, p1", 'so').is_spin_conserving()


def test_multiset_permutation_1():
    a = SecondQuantizedOperator("g0,g1,g2", "p0,p1,p2", 'spin-orbital')
    p_cre = [[Index(i) for i in "g0,g1,g2".split(',')]]
    p_ann = [[Index(i) for i in "p0,p1,p2".split(',')]]
    assert not a.exist_permute_format(p_cre, p_ann)
    assert a.n_multiset_permutation(p_cre, p_ann) == 1


def test_multiset_permutation_2():
    a = SecondQuantizedOperator("g0,g1,g2", "p0,p1,p2", 'spin-integrated')
    p_cre = [[Index(i) for i in "g0,g1,g2".split(',')]]
    p_ann = [[Index('p0')], [Index('p1')], [Index('p2')]]
    assert a.exist_permute_format(p_cre, p_ann)
    assert a.n_multiset_permutation(p_cre, p_ann) == 6

    p_ann = [[Index('p0'), Index('p1')], [Index('p2')]]
    assert a.n_multiset_permutation(p_cre, p_ann) == 3


def test_multiset_permutation_3():
    a = SecondQuantizedOperator("g0,v1,p2", "p0,a1,p2", 'spin-integrated')
    p_cre = [[Index('g0')], [Index('v1')], [Index('p2')]]
    p_ann = [[Index('p0'), Index('p2')], [Index('a1')]]
    assert a.exist_permute_format(p_cre, p_ann)
    assert a.n_multiset_permutation(p_cre, p_ann) == 18


def test_multiset_permutation_4():
    a = SecondQuantizedOperator("g0,v1,p2", "p0,a1,c2", 'spin-orbital')
    p_cre = [[Index('g0'), Index('v1'), Index('p2')]]
    p_ann = [[Index('p0'), Index('a1'), Index('c2')]]
    assert not a.exist_permute_format(p_cre, p_ann)
    assert a.n_multiset_permutation(p_cre, p_ann) == 1


def test_multiset_permutation_5():
    # n_multiset_permutation for spin-adapted operators is always one
    a = SecondQuantizedOperator("g0,v1,p2", "p0,a1,c2", 'spin-adapted')
    p_cre = [[Index('g0'), Index('v1'), Index('p2')]]
    p_ann = [[Index('p0')], [Index('a1')], [Index('c2')]]
    assert not a.exist_permute_format(p_cre, p_ann)
    assert a.n_multiset_permutation(p_cre, p_ann) == 1


def test_latex_permute_format_1():
    a = SecondQuantizedOperator("g0,g1,g2", "p0,p1,p2", 'spin-orbital')
    p_cre = [[Index(i) for i in "g0,g1,g2".split(',')]]
    p_ann = [[Index(i) for i in "p0,p1,p2".split(',')]]
    n_perm, perm, latex_str = a.latex_permute_format(p_cre, p_ann)
    assert n_perm == 1
    assert perm == ''
    assert latex_str == 'a^{ g_{0} g_{1} g_{2} }_{ p_{0} p_{1} p_{2} }'


def test_latex_permute_format_2():
    a = SecondQuantizedOperator("g0,v1,p3", "p0,a1,p2", 'spin-integrated')
    p_cre = [[Index('g0')], [Index('p3')], [Index('v1')]]
    p_ann = [[Index('p0'), Index('p2')], [Index('a1')]]
    n_perm, perm, latex_str = a.latex_permute_format(p_cre, p_ann)
    assert n_perm == 18
    assert perm == '{\\cal P}(g_{0} / p_{3} / v_{1}) {\\cal P}(p_{0} p_{2} / a_{1})'
    assert latex_str == 'a^{ g_{0} v_{1} p_{3} }_{ p_{0} a_{1} p_{2} }'


def test_latex_permute_format_3():
    # ignore multiset permutations for mixed spin indices
    a = SecondQuantizedOperator("g0,v1,p2", "p0,a1,P2", 'spin-integrated')
    p_cre = [[Index('g0')], [Index('p2')], [Index('v1')]]
    p_ann = [[Index('p0'), Index('P2')], [Index('a1')]]
    n_perm, perm, latex_str = a.latex_permute_format(p_cre, p_ann)
    assert n_perm == 6
    assert perm == '{\\cal P}(g_{0} / p_{2} / v_{1})'
    assert latex_str == 'a^{ g_{0} v_{1} p_{2} }_{ p_{0} a_{1} P_{2} }'


def test_ambit_permute_format_1():
    # empty sq_op ignores all input partitions
    a = SecondQuantizedOperator([], [], 'spin-orbital')
    for pair in a.ambit_permute_format([], []):
        assert pair == (1, '')


def test_ambit_permute_format_2():
    a = SecondQuantizedOperator("g0,g1,c2", "p0,p1,p2", 'spin-integrated')
    p_cre = [[Index('g0'), Index('g1')], [Index('c2')]]
    p_ann = [[Index('p0'), Index('p1'), Index('p2')]]
    ref = {(1, '["p0,p1,p2,g0,g1,c2"]'), (-1, '["p0,p1,p2,g0,c2,g1"]'), (1, '["p0,p1,p2,c2,g0,g1"]')}
    for pair in a.ambit_permute_format(p_cre, p_ann, cre_first=False):
        assert pair in ref
        ref.remove(pair)
    assert len(ref) == 0


def test_ambit_permute_format_3():
    # ignore all input partitions for mixed spin indices
    a = SecondQuantizedOperator("g0,G1,c2", "p0,p1,P2", 'spin-integrated')
    for pair in a.ambit_permute_format([], [], cre_first=True):
        assert pair == (1, '["g0,G1,c2,p0,p1,P2"]')


def test_canonicalize():
    a = SecondQuantizedOperator("G2, p0, p1", "g0, A0, h2, A2", 'si')
    c, sign = a.canonicalize()
    assert c == SecondQuantizedOperator("p0, p1, G2", "g0, h2, A0, A2", 'si')
    assert sign == -1


def test_void():
    a = SecondQuantizedOperator("G2, p0, p1", "g0, A0, h2", 'si')
    b = a.void()
    assert a.indices_type == b.indices_type
    assert b.size == 0
    assert b is not SecondQuantizedOperator.make_empty('si')


def test_base_strong_generating_set():
    with pytest.raises(ValueError):
        SecondQuantizedOperator("", "", 'si').base_strong_generating_set()


def test_possible_excitation():
    assert SecondQuantizedOperator("g0, g1", "g2, p0").is_possible_excitation()
    assert SecondQuantizedOperator("p0, p1", "c0, h0").is_possible_excitation()
    assert not SecondQuantizedOperator("c0, h0", "p0, p1").is_possible_excitation()
    assert not SecondQuantizedOperator("a0, A0", "a1, A1", 'si').is_possible_excitation()

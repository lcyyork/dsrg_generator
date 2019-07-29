import pytest
from src.Index import Index
from src.Indices import Indices, IndicesSpinOrbital, IndicesSpinIntegrated


def test_indices_init():
    with pytest.raises(ValueError):
        Indices("a0 c1, P2, V8")
    with pytest.raises(ValueError):
        Indices(["a0"] * 3)
    a = Indices("a0, P2")
    assert a.indices == [Index("a0"), Index("P2")]
    assert a.size == 2
    assert a.indices_set == {Index("a0"), Index("P2")}

    a = Indices([])
    assert a == Indices("")
    assert a.size == 0


def test_indices_str():
    assert str(Indices(["p0", "p1", "a0", "h0", "g4", "g1", "c1", "v2"])) == "p0, p1, a0, h0, g4, g1, c1, v2"


def test_indices_latex():
    assert Indices("g1, g4, p0, h1, v2, c1").latex(True) == "${ g_{1} g_{4} p_{0} h_{1} v_{2} c_{1} }$"


def test_indices_ambit():
    assert Indices(["p0", "p1", "a0", "h0", "g4", "g1", "c1", "v2"]).ambit() == "p0,p1,a0,h0,g4,g1,c1,v2"


def test_indices_eq():
    assert Indices("g0, c1, V3") == Indices(["g0", 'c1', "V3"])


def test_indices_ne():
    assert not Indices("g0, c1, V3") != Indices(["g0", 'c1', "V3"])


def test_indices_lt():
    assert Indices("A3") < Indices("g0, v2")
    assert Indices("g0, c1, V3") < Indices("c1, V3, g0")


def test_indices_le():
    assert Indices("A3") <= Indices("g0, v2")
    assert Indices("g0, c1, V3") <= Indices("c1, V3, g0")
    assert Indices("g0, c1, V3") <= Indices("g0, c1, V3")


def test_indices_gt():
    assert Indices("g0, C2, H8") > Indices("v2")
    assert Indices("g0, C2, h8") > Indices("g0, c2, h8")


def test_indices_ge():
    assert Indices("g0, C2, H8") >= Indices("v2")
    assert Indices("g0, C2, h8") >= Indices("g0, c2, h8")
    assert Indices("g0, c1, V3") >= Indices("g0, c1, V3")


def test_indices_get():
    with pytest.raises(IndexError):
        assert Indices([])[0]
    assert Indices("p0, p1, a0, h0, g4, g1, c1, v2")[2] == Index("a0")


def test_indices_add():
    a = Indices("a0")
    assert a + Indices("a1") == Indices("a0, a1")
    a += Indices("a1")
    assert a == Indices("a0, a1")


def test_indices_perm():
    a = Indices("p0, p1, g2, A4")
    assert a.is_permutation(Indices("g2, p1, p0, A4"))


def test_indices_count_space():
    assert Indices("p0, P1, a2, h0, g4, G1, C1, v2").count_index_space(['p', 'v', 'P', 'V']) == 3


def test_any_overlap():
    a = Indices("p0, p1, g2, A4")
    b = Indices("p3, p1, c2, a4")
    assert a.any_overlap(b)
    b = Indices("p3, h1, c2, a4")
    assert not a.any_overlap(b)


def test_clone():
    a = Indices("p0, c1, g2, A4")
    b = a.clone()
    assert a == b
    assert a is not b


def test_indices_so_init():
    with pytest.raises(ValueError):
        Indices.make_indices("p0, V2, A2", 'spin-orbital')


def test_indices_so_canonical():
    a = Indices.make_indices("p0, p1, a0, h0, g4, g1, c1, v2", 'so')
    c, sign = a.canonicalize()
    assert sign == -1
    assert c == IndicesSpinOrbital("g1, g4, p0, p1, h0, v2, c1, a0")
    with pytest.raises(TypeError):
        assert c == Indices("g1, g4, p0, p1, h0, v2, c1, a0")


def test_indices_so_ambit_perm():
    a = IndicesSpinOrbital(["p0", "p1", "v2", "a3"])
    part = [[Index("p0"), Index("p1")], [Index("v2")], [Index("a3")]]
    for sign, indices_str in a.ambit_permute_format(part):
        assert sign == (-1) ** a.count_permutations(Indices(indices_str))


def test_indices_so_latex_perm():
    part = [[Index("g2")], [Index("p0"), Index("p1")], [Index("a4")]]
    n_perm, perm = IndicesSpinOrbital("p0, p1, g2, a4").latex_permute_format(part)
    assert n_perm == 12
    assert perm == "{\\cal P}(g_{2} / p_{0} p_{1} / a_{4})"


def test_indices_so_spin():
    ref = ["c0, c1, g0", "c0, c1, G0", "c0, C1, g0", "C0, c1, g0",
           "c0, C1, G0", "C0, c1, G0", "C0, C1, g0", "C0, C1, G0"]
    ref = set(map(IndicesSpinIntegrated, ref))
    a = IndicesSpinOrbital("c0, c1, g0")
    for i in a.generate_spin_cases():
        assert i in ref
        ref.remove(i)
    assert len(ref) == 0

    ref = ["c0, C1, G0", "C0, c1, G0", "C0, C1, g0"]
    ref = set(map(IndicesSpinIntegrated, ref))
    for i in a.generate_spin_cases(2):
        assert i in ref
        ref.remove(i)
    assert len(ref) == 0

    with pytest.raises(TypeError):
        for _ in a.generate_spin_cases('a'):
            pass

    with pytest.raises(ValueError):
        for _ in a.generate_spin_cases(9):
            pass


def test_indices_si_canonical():
    a = IndicesSpinIntegrated("p0, p1, a0, h0, G4, g1, C1, v2")
    c, sign = a.canonicalize()
    assert sign == -1
    assert c == IndicesSpinIntegrated("g1, p0, p1, h0, v2, a0, G4, C1")


def test_indices_si_ambit_perm():
    a = IndicesSpinIntegrated(["P0", "P1", "V2", "A3"])
    part = [["V2"], ["P0", "P1"], ["A3"]]
    ref = {"P0,P1,V2,A3",
           "P0,P1,A3,V2",
           "P0,A3,P1,V2",
           "A3,P0,P1,V2",
           "P0,V2,P1,A3",
           "V2,P0,P1,A3",
           "P0,A3,V2,P1",
           "P0,V2,A3,P1",
           "V2,P0,A3,P1",
           "A3,P0,V2,P1",
           "A3,V2,P0,P1",
           "V2,A3,P0,P1"}
    for sign, indices_str in a.ambit_permute_format(part):
        assert sign == (-1) ** a.count_permutations(Indices(indices_str))
        assert indices_str in ref
        ref.remove(indices_str)
    assert len(ref) == 0

    a = IndicesSpinIntegrated(["P0", "P1", "c2", "A3"])
    for sign, indices_str in a.ambit_permute_format(part):
        assert sign == 1
        assert indices_str == ",".join(map(str, a.indices))


def test_indices_si_spin_pure():
    assert IndicesSpinIntegrated(["P0", "P1", "V2", "A3"]).spin_pure
    assert not IndicesSpinIntegrated(["P0", "a1", "a2", "g3"]).spin_pure


def test_indices_si_alpha_beta():
    a = IndicesSpinIntegrated(["P0", "P1", "V2", "A3"])
    assert a.n_beta() == 4 and a.n_alpha() == 0
    a = IndicesSpinIntegrated(["P0", "P1", "v2", "A3", "h2"])
    assert a.n_beta() == 3 and a.n_alpha() == 2


def test_indices_si_latex_permute_format():
    a = IndicesSpinIntegrated(["P0", "P1", "P2", "A3"])
    part = [["P2"], ["P0", "P1"], ["A3"]]
    with pytest.raises(AttributeError):
        a.latex_permute_format(part)

    part = [[Index("P2")], [Index("P0"), Index("P1")], [Index("A3")]]
    n_perm, perm_str = a.latex_permute_format(part)
    assert (n_perm, perm_str) == (12, '{\\cal P}(P_{2} / P_{0} P_{1} / A_{3})')

    a = IndicesSpinIntegrated(["p0", "P1", "V2", "A3"])
    assert a.latex_permute_format(part) == (1, '')

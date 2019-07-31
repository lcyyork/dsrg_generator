import pytest
from src.Indices import IndicesSpinAdapted, IndicesSpinIntegrated
from src.IndicesPair import IndicesPair
from src.Tensor import Tensor, HoleDensity, Cumulant, Kronecker, ClusterAmplitude, HamiltonianTensor


def test_init_1():
    # name is not string
    with pytest.raises(TypeError):
        Tensor("H0, C2", "P1, V4", 'sa', 1, 10)

    # priority is not an integer
    with pytest.raises(TypeError):
        Tensor("H0, C2", "P1, V4", 'spin-adapted', "temp", 'x')


def test_init_2():
    a = Tensor("H0, C2", "P1, V4", 'spin-adapted', 'temp', 10)
    pair = IndicesPair("H0, C2", "P1, V4", 'spin-adapted')
    assert a.name == 'temp'
    assert a.priority == 10
    assert a.upper_indices == pair.upper_indices
    assert a.lower_indices == pair.lower_indices
    assert a.n_lower == a.n_upper == 2
    assert a.size == 4
    assert a.indices_type is IndicesSpinAdapted


def test_init_3():
    # lower indices contains both hole and particle indices
    with pytest.raises(ValueError):
        assert Tensor.make_tensor('cluster_amplitude', "a1,a2", "c3,p0", 'spin-orbital')


def test_eq():
    a = Tensor("a0,a1", "c2,v6", 'spin-orbital', 'temp', 8)
    b = a.clone()
    assert a == b
    assert a is not b


def test_ne():
    a = Tensor.make_tensor('cumulant', "a0,a1,a2", "a3,a4,a5", 'so')
    assert a != Tensor.make_tensor('cluster_amplitude', "a0,a1,a2", "a3,a4,a5", 'so')
    assert a != Tensor.make_tensor('cumulant', "a0", "a1", 'so')

    with pytest.raises(TypeError):
        assert a != Tensor.make_tensor('cumulant', "a0", "a1", 'spin-integrated')
    with pytest.raises(TypeError):
        assert a != Tensor.make_tensor('cumulant', "a0,a1,a2", "a3,a4,a5", 'spin-integrated')


def test_lt():
    indices_pair = ("C1", "P2", 'spin-integrated')
    assert Kronecker(*indices_pair) < HamiltonianTensor(*indices_pair) < ClusterAmplitude(*indices_pair) \
        < HoleDensity(*indices_pair) < Cumulant(*indices_pair)
    assert ClusterAmplitude(*indices_pair) < Tensor.make_tensor('T', "c1,c2", "v2,v5", 'si')
    a = Tensor.make_tensor('cumulant', "a0,A2", "a3, A1", 'spin-integrated')
    assert a < Tensor.make_tensor('cumulant', "a0, A2", "A1, a3", 'spin-integrated')


def test_le():
    a = Tensor.make_tensor('cumulant', "a0,A2", "a3, A1", 'spin-integrated')
    assert a <= Tensor.make_tensor('cumulant', "a0, A2", "A1, a3", 'spin-integrated')
    assert a <= Cumulant("a0,A2", "a3, A1", 'spin-integrated')


def test_gt():
    a = Tensor.make_tensor('cumulant', "a0,A2", "a3, A1", 'spin-integrated')
    assert a > Tensor.make_tensor('cumulant', "a0,A2", "a1, A3", 'spin-integrated')


def test_ge():
    a = Tensor.make_tensor('cumulant', "a0,A2", "a3, A1", 'spin-integrated')
    assert a >= Tensor.make_tensor('cumulant', "a0,A2", "a1, A3", 'spin-integrated')
    assert a >= Tensor.make_tensor('Hamiltonian', "a0", "a3", 'spin-integrated')


def test_latex():
    a = Tensor.make_tensor('cluster_amplitude', "a1,a2", "v3,p0", 'spin-orbital')
    assert a.latex() == "T^{ a_{1} a_{2} }_{ v_{3} p_{0} }"

    a = Tensor.make_tensor('cluster_amplitude', "v3,p0", "c1,a2", 'spin-orbital')
    assert a.latex() == "T^{ v_{3} p_{0} }_{ c_{1} a_{2} }"


def test_ambit():
    a = Tensor.make_tensor('cluster_amplitude', "a1,a2", "v3,p0", 'spin-orbital')
    assert a.ambit() == 'T2["a1,a2,v3,p0"]'

    a = Tensor.make_tensor('cluster_amplitude', "p0, a3", "a1,a2", 'spin-orbital')
    assert a.ambit() == 'T2["a1,a2,p0,a3"]'


def test_any_overlapped_indices():
    a = Tensor.make_tensor('cumulant', "c0,A2", "g3,A1", 'spin-integrated')
    assert a.any_overlapped_indices(Tensor.make_tensor('cumulant', "g0,P2", "A1,v1", 'spin-integrated'))
    assert not a.any_overlapped_indices(Tensor.make_tensor('cumulant', "a2,v2", "a3,h1", 'spin-integrated'))


def test_canonicalize():
    a = Tensor.make_tensor('cluster_amplitude', "a1,a2", "v3,p0", 'spin-integrated')
    c, sign = a.canonicalize()
    assert c == Tensor.make_tensor('cluster_amplitude', "a1,a2", "p0,v3", 'spin-integrated')
    assert sign == -1
    assert type(c) is ClusterAmplitude
    assert c.indices_type is IndicesSpinIntegrated


def test_generate_spin_cases():
    a = Tensor.make_tensor('cluster_amplitude', "a1,a2", "p0,p1", 'spin-orbital')
    ref = {Tensor.make_tensor('cluster_amplitude', "a1,a2", "p0,p1", 'spin-integrated'),
           Tensor.make_tensor('cluster_amplitude', "A1,a2", "P0,p1", 'spin-integrated'),
           Tensor.make_tensor('cluster_amplitude', "A1,a2", "p0,P1", 'spin-integrated'),
           Tensor.make_tensor('cluster_amplitude', "a1,A2", "P0,p1", 'spin-integrated'),
           Tensor.make_tensor('cluster_amplitude', "a1,A2", "p0,P1", 'spin-integrated'),
           Tensor.make_tensor('cluster_amplitude', "A1,A2", "P0,P1", 'spin-integrated')}
    count = 0
    for tensor in a.generate_spin_cases():
        count += 1
        assert tensor in ref
        assert type(tensor) is ClusterAmplitude
        assert tensor.indices_type is IndicesSpinIntegrated
    assert count == len(ref)


def test_expand_hole_density():
    a = Tensor.make_tensor('hole_density', "a4", "p3", 'si')
    k, c = a.expand()
    assert k == Tensor.make_tensor('delta', "a4", "p3", 'si')
    assert c == Tensor.make_tensor('lambda', "a4", "p3", 'si')


def test_downgrade_indices():
    for tensor_type in ['cluster_amplitude', 'Hamiltonian']:
        a = Tensor.make_tensor(tensor_type, 'P1', 'H3', 'spin-adapted')
        with pytest.raises(NotImplementedError):
            a.downgrade_indices()

    a = Tensor.make_tensor('Kronecker', 'P1', 'H3', 'spin-integrated')
    assert a.downgrade_indices() == 'A'

    a = Tensor.make_tensor('Kronecker', 'V1', 'C3', 'spin-integrated')
    assert a.downgrade_indices() == ''

    a = Tensor.make_tensor('Kronecker', 'G1', 'H3', 'spin-integrated')
    assert a.downgrade_indices() == 'H'

    a = Tensor.make_tensor('cumulant', 'G0,c2,p2', 'v2,H4,a1', 'spin-integrated')
    with pytest.raises(NotImplementedError):
        a.downgrade_indices()

    space_pair = {('c', 'c'): 'c',
                  ('c', 'a'): '',
                  ('c', 'v'): '',
                  ('c', 'h'): 'c',
                  ('c', 'p'): '',
                  ('c', 'g'): 'c',
                  ('a', 'a'): 'a',
                  ('a', 'v'): '',
                  ('a', 'h'): 'a',
                  ('a', 'p'): 'a',
                  ('a', 'g'): 'a',
                  ('v', 'v'): '',
                  ('v', 'h'): '',
                  ('v', 'p'): '',
                  ('v', 'g'): '',
                  ('h', 'h'): 'h',
                  ('h', 'p'): 'a',
                  ('h', 'g'): 'h',
                  ('p', 'p'): 'a',
                  ('p', 'g'): 'a',
                  ('g', 'g'): 'h'}
    for spaces, value in space_pair.items():
        index0, index1 = (f"{i}{j}" for i, j in zip(spaces, range(2)))
        a = Tensor.make_tensor('cumulant', index0, index1, 'spin-orbital')
        assert a.downgrade_indices() == value

    space_pair = {('C', 'C'): '',
                  ('C', 'A'): '',
                  ('C', 'V'): '',
                  ('C', 'H'): '',
                  ('C', 'P'): '',
                  ('C', 'G'): '',
                  ('A', 'A'): 'A',
                  ('A', 'V'): '',
                  ('A', 'H'): 'A',
                  ('A', 'P'): 'A',
                  ('A', 'G'): 'A',
                  ('V', 'V'): 'V',
                  ('V', 'H'): '',
                  ('V', 'P'): 'V',
                  ('V', 'G'): 'V',
                  ('H', 'H'): 'A',
                  ('H', 'P'): 'A',
                  ('H', 'G'): 'A',
                  ('P', 'P'): 'P',
                  ('P', 'G'): 'P',
                  ('G', 'G'): 'P'}
    for spaces, value in space_pair.items():
        index0, index1 = (f"{i}{j}" for i, j in zip(spaces, range(2)))
        a = Tensor.make_tensor('hole_density', index0, index1, 'spin-integrated')
        assert a.downgrade_indices() == value


def test_is_all_active():
    a = Tensor.make_tensor('cluster_amplitude', "a1,a2", "p0,v1", 'spin-orbital')
    assert not a.is_all_active()

    a = Tensor.make_tensor('cluster_amplitude', "a1,a2", "a3,a4", 'spin-orbital')
    assert a.is_all_active()

    a = Tensor.make_tensor('cluster_amplitude', "a1,a2,A0", "a3,a4,A1", 'spin-integrated')
    assert a.is_all_active()

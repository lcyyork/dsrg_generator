import pytest
from Tensor import make_tensor_preset, make_tensor
from Tensor import Tensor, HoleDensity, Cumulant, Kronecker, ClusterAmplitude, Hamiltonian
from IndicesPair import make_indices_pair
from Indices import IndicesSpinAdapted, IndicesSpinIntegrated
from Index import Index
from sqop_contraction import expand_hole_densities


def test_init():
    indices_pair = make_indices_pair("H0, C2", "P1, V4", 'spin-adapted')
    with pytest.raises(TypeError):
        Tensor(indices_pair, 1, 10)  # name is not string
    with pytest.raises(TypeError):
        Tensor("h0", "temp", 10)  # indices is not of IndicesPair
    with pytest.raises(TypeError):
        Tensor(indices_pair, "temp", 'x')  # priority is not an integer

    a = Tensor(indices_pair, 'temp', 10)
    assert a.name == 'temp'
    assert a.indices_pair == make_indices_pair("H0, C2", "P1, V4", 'spin-adapted')
    assert a.priority == 10
    assert a.upper_indices == indices_pair.upper_indices
    assert a.lower_indices == indices_pair.lower_indices
    assert a.n_lower == 2
    assert a.n_upper == 2
    assert a.size == 4
    assert a.type_of_indices is IndicesSpinAdapted


def test_eq():
    a = make_tensor('temp', "a0,a1", "c2,v6", 'spin-orbital', 8)
    assert a == Tensor(make_indices_pair("a0,a1", "c2,v6", 'spin-orbital'), 'temp', 8)


def test_ne():
    a = make_tensor_preset('cumulant', "a0,a1,a2", "a3,a4,a5", 'spin-orbital')
    assert a != make_tensor_preset('cluster_amplitude', "a0,a1,a2", "a3,a4,a5", 'spin-orbital')
    assert a != make_tensor_preset('cumulant', "a0", "a1", 'spin-orbital')

    # note comparision sequence: priority, name, size, indices_pair
    # thus, the first of the following will not raise an error, while the second will.
    assert a != make_tensor_preset('cumulant', "a0", "a1", 'spin-integrated')
    with pytest.raises(TypeError):
        assert a != make_tensor_preset('cumulant', "a0,a1,a2", "a3,a4,a5", 'spin-integrated')


def test_lt():
    indices_pair = make_indices_pair("C1", "P2", 'spin-integrated')
    assert Kronecker(indices_pair) < Hamiltonian(indices_pair) < ClusterAmplitude(indices_pair) \
        < HoleDensity(indices_pair) < Cumulant(indices_pair)
    assert ClusterAmplitude(indices_pair) < make_tensor_preset('cluster_amplitude', "c1,c2", "v2,v5",
                                                               'spin-integrated')
    a = make_tensor_preset('cumulant', "a0,A2", "a3, A1", 'spin-integrated')
    assert a < make_tensor_preset('cumulant', "a0, A2", "A1, a3", 'spin-integrated')


def test_le():
    a = make_tensor_preset('cumulant', "a0,A2", "a3, A1", 'spin-integrated')
    assert a <= make_tensor_preset('cumulant', "a0, A2", "A1, a3", 'spin-integrated')
    assert a <= Cumulant(make_indices_pair("a0,A2", "a3, A1", 'spin-integrated'))


def test_gt():
    a = make_tensor_preset('cumulant', "a0,A2", "a3, A1", 'spin-integrated')
    assert a > make_tensor_preset('cumulant', "a0,A2", "a1, A3", 'spin-integrated')


def test_ge():
    a = make_tensor_preset('cumulant', "a0,A2", "a3, A1", 'spin-integrated')
    assert a >= make_tensor_preset('cumulant', "a0,A2", "a1, A3", 'spin-integrated')
    assert a >= make_tensor_preset('Hamiltonian', "a0", "a3", 'spin-integrated')


def test_latex():
    a = make_tensor_preset('cluster_amplitude', "a1,a2", "c3,p0", 'spin-orbital')
    assert a.latex() == "T^{ a_{1} a_{2} }_{ c_{3} p_{0} }"


def test_ambit():
    a = make_tensor_preset('cluster_amplitude', "a1,a2", "c3,p0", 'spin-orbital')
    assert a.ambit() == 'T_2_2["a1,a2,c3,p0"]'


def test_is_permutation():
    a = make_tensor_preset('cluster_amplitude', "a1,a2", "c3,p0", 'spin-orbital')
    assert not a.is_permutation(make_tensor_preset('cluster_amplitude', "a1", "c3", 'spin-orbital'))
    assert not a.is_permutation(make_tensor_preset('cumulant', "a1,a2", "c3,p0", 'spin-orbital'))
    assert a.is_permutation(make_tensor_preset('cluster_amplitude', "a1,a2", "p0,c3", 'spin-orbital'))


def test_any_overlapped_indices():
    a = make_tensor_preset('cumulant', "c0,A2", "g3,A1", 'spin-integrated')
    assert a.any_overlapped_indices(make_tensor_preset('cumulant', "g0,P2", "A1,v1", 'spin-integrated'))
    assert not a.any_overlapped_indices(make_tensor_preset('cumulant', "a2,v2", "a3,h1", 'spin-integrated'))


def test_canonicalize():
    a = make_tensor_preset('cluster_amplitude', "a1,a2", "c3,p0", 'spin-integrated')
    c, sign = a.canonicalize()
    assert c == make_tensor_preset('cluster_amplitude', "a1,a2", "p0,c3", 'spin-integrated')
    assert sign == -1
    assert type(c) is ClusterAmplitude
    assert c.type_of_indices is IndicesSpinIntegrated


def test_generate_spin_cases():
    a = make_tensor_preset('cluster_amplitude', "a1,a2", "p0,p1", 'spin-orbital')
    ref = [make_tensor_preset('cluster_amplitude', "a1,a2", "p0,p1", 'spin-integrated'),
           make_tensor_preset('cluster_amplitude', "A1,a2", "P0,p1", 'spin-integrated'),
           make_tensor_preset('cluster_amplitude', "A1,a2", "p0,P1", 'spin-integrated'),
           make_tensor_preset('cluster_amplitude', "a1,A2", "P0,p1", 'spin-integrated'),
           make_tensor_preset('cluster_amplitude', "a1,A2", "p0,P1", 'spin-integrated'),
           make_tensor_preset('cluster_amplitude', "A1,A2", "P0,P1", 'spin-integrated')]
    count = 0
    for tensor in a.generate_spin_cases():
        count += 1
        assert tensor in ref
        assert type(tensor) is ClusterAmplitude
        assert tensor.type_of_indices is IndicesSpinIntegrated
    assert count == len(ref)


def test_expand_hole():
    tensors = [make_tensor_preset('Hamiltonian', "g0", "h1", 'spin-integrated'),
               make_tensor_preset('hole_density', "a4", "p3", 'spin-orbital'),
               make_tensor_preset('cluster_amplitude', "a1,a2", "p0,p1", 'spin-orbital'),
               make_tensor_preset('hole_density', "v0", "p6", 'spin-orbital'),
               make_tensor_preset('hole_density', "h0", "p4", 'spin-orbital')]
    ref = [(1, [make_tensor_preset('Hamiltonian', "g0", "h1", 'spin-integrated'),
                make_tensor_preset('cluster_amplitude', "a1,a2", "p0,p1", 'spin-orbital'),
                make_tensor_preset('Kronecker', "a4", "p3", 'spin-orbital'),
                make_tensor_preset('Kronecker', "v0", "p6", 'spin-orbital'),
                make_tensor_preset('Kronecker', "h0", "p4", 'spin-orbital')]),
           (-1, [make_tensor_preset('Hamiltonian', "g0", "h1", 'spin-integrated'),
                 make_tensor_preset('cluster_amplitude', "a1,a2", "p0,p1", 'spin-orbital'),
                 make_tensor_preset('Kronecker', "a4", "p3", 'spin-orbital'),
                 make_tensor_preset('Kronecker', "v0", "p6", 'spin-orbital'),
                 make_tensor_preset('cumulant', "h0", "p4", 'spin-orbital')]),
           (-1, [make_tensor_preset('Hamiltonian', "g0", "h1", 'spin-integrated'),
                 make_tensor_preset('cluster_amplitude', "a1,a2", "p0,p1", 'spin-orbital'),
                 make_tensor_preset('cumulant', "a4", "p3", 'spin-orbital'),
                 make_tensor_preset('Kronecker', "v0", "p6", 'spin-orbital'),
                 make_tensor_preset('Kronecker', "h0", "p4", 'spin-orbital')]),
           (1, [make_tensor_preset('Hamiltonian', "g0", "h1", 'spin-integrated'),
                make_tensor_preset('cluster_amplitude', "a1,a2", "p0,p1", 'spin-orbital'),
                make_tensor_preset('cumulant', "a4", "p3", 'spin-orbital'),
                make_tensor_preset('Kronecker', "v0", "p6", 'spin-orbital'),
                make_tensor_preset('cumulant', "h0", "p4", 'spin-orbital')])]

    count = 0
    for sign_tensors in expand_hole_densities(tensors):
        assert sign_tensors in ref
        count += 1
    assert count == 4


def test_downgrade_indices():
    for tensor_type in ['cluster_amplitude', 'Hamiltonian']:
        a = make_tensor_preset(tensor_type, 'P1', 'H3', 'spin-adapted')
        with pytest.raises(NotImplementedError):
            a.downgrade_indices()

    a = make_tensor_preset('Kronecker', 'P1', 'H3', 'spin-integrated')
    assert a.downgrade_indices() == 'A'

    a = make_tensor_preset('Kronecker', 'V1', 'C3', 'spin-integrated')
    assert a.downgrade_indices() == ''

    a = make_tensor_preset('Kronecker', 'G1', 'H3', 'spin-integrated')
    assert a.downgrade_indices() == 'H'

    a = make_tensor_preset('cumulant', 'G0,c2,p2', 'v2,H4,a1', 'spin-integrated')
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
        a = make_tensor_preset('cumulant', index0, index1, 'spin-orbital')
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
        a = make_tensor_preset('hole_density', index0, index1, 'spin-integrated')
        assert a.downgrade_indices() == value

def test_is_all_active():
    a = make_tensor_preset('cluster_amplitude', "a1,a2", "p0,v1", 'spin-orbital')
    assert not a.is_all_active()

    a = make_tensor_preset('cluster_amplitude', "a1,a2", "a3,a4", 'spin-orbital')
    assert a.is_all_active()

    a = make_tensor_preset('cluster_amplitude', "a1,a2,A0", "a3,a4,A1", 'spin-integrated')
    assert a.is_all_active()

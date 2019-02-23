import pytest
from Tensor import make_tensor_preset, make_tensor
from Tensor import Tensor, HoleDensity, Cumulant, Kronecker, ClusterAmplitude, Hamiltonian
from IndicesPair import make_indices_pair
from Indices import IndicesSpinAdapted, IndicesSpinIntegrated
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
    a = make_tensor_preset("a0,a1,a2", "a3,a4,a5", 'spin-orbital', 'cumulant')
    assert a != make_tensor_preset("a0,a1,a2", "a3,a4,a5", 'spin-orbital', 'cluster_amplitude')
    assert a != make_tensor_preset("a0", "a1", 'spin-orbital', 'cumulant')

    # note comparision sequence: priority, name, size, indices_pair
    # thus, the first of the following will not raise an error, while the second will.
    assert a != make_tensor_preset("a0", "a1", 'spin-integrated', 'cumulant')
    with pytest.raises(TypeError):
        assert a != make_tensor_preset("a0,a1,a2", "a3,a4,a5", 'spin-integrated', 'cumulant')


def test_lt():
    indices_pair = make_indices_pair("C1", "P2", 'spin-integrated')
    assert Kronecker(indices_pair) < Hamiltonian(indices_pair) < ClusterAmplitude(indices_pair) \
        < HoleDensity(indices_pair) < Cumulant(indices_pair)
    assert ClusterAmplitude(indices_pair) < make_tensor_preset("c1,c2", "v2,v5",
                                                               'spin-integrated', 'cluster_amplitude')
    a = make_tensor_preset("a0,A2", "a3, A1", 'spin-integrated', 'cumulant')
    assert a < make_tensor_preset("a0, A2", "A1, a3", 'spin-integrated', 'cumulant')


def test_le():
    a = make_tensor_preset("a0,A2", "a3, A1", 'spin-integrated', 'cumulant')
    assert a <= make_tensor_preset("a0, A2", "A1, a3", 'spin-integrated', 'cumulant')
    assert a <= Cumulant(make_indices_pair("a0,A2", "a3, A1", 'spin-integrated'))


def test_gt():
    a = make_tensor_preset("a0,A2", "a3, A1", 'spin-integrated', 'cumulant')
    assert a > make_tensor_preset("a0,A2", "a1, A3", 'spin-integrated', 'cumulant')


def test_ge():
    a = make_tensor_preset("a0,A2", "a3, A1", 'spin-integrated', 'cumulant')
    assert a >= make_tensor_preset("a0,A2", "a1, A3", 'spin-integrated', 'cumulant')
    assert a >= make_tensor_preset("a0", "a3", 'spin-integrated', 'Hamiltonian')


def test_latex():
    a = make_tensor_preset("a1,a2", "c3,p0", 'spin-orbital', 'cluster_amplitude')
    assert a.latex() == "T^{ a_{1} a_{2} }_{ c_{3} p_{0} }"


def test_ambit():
    a = make_tensor_preset("a1,a2", "c3,p0", 'spin-orbital', 'cluster_amplitude')
    assert a.ambit() == 'T_2_2["a1,a2,c3,p0"]'


def test_is_permutation():
    a = make_tensor_preset("a1,a2", "c3,p0", 'spin-orbital', 'cluster_amplitude')
    assert not a.is_permutation(make_tensor_preset("a1", "c3", 'spin-orbital', 'cluster_amplitude'))
    assert not a.is_permutation(make_tensor_preset("a1,a2", "c3,p0", 'spin-orbital', 'cumulant'))
    assert a.is_permutation(make_tensor_preset("a1,a2", "p0,c3", 'spin-orbital', 'cluster_amplitude'))


def test_any_overlapped_indices():
    a = make_tensor_preset("c0,A2", "g3,A1", 'spin-integrated', 'cumulant')
    assert a.any_overlapped_indices(make_tensor_preset("g0,P2", "A1,v1", 'spin-integrated', 'cumulant'))
    assert not a.any_overlapped_indices(make_tensor_preset("a2,v2", "a3,h1", 'spin-integrated', 'cumulant'))


def test_canonicalize():
    a = make_tensor_preset("a1,a2", "c3,p0", 'spin-integrated', 'cluster_amplitude')
    c, sign = a.canonicalize()
    assert c == make_tensor_preset("a1,a2", "p0,c3", 'spin-integrated', 'cluster_amplitude')
    assert sign == -1
    assert type(c) is ClusterAmplitude
    assert c.type_of_indices is IndicesSpinIntegrated


def test_generate_spin_cases():
    a = make_tensor_preset("a1,a2", "p0,p1", 'spin-orbital', 'cluster_amplitude')
    ref = [make_tensor_preset("a1,a2", "p0,p1", 'spin-integrated', 'cluster_amplitude'),
           make_tensor_preset("A1,a2", "P0,p1", 'spin-integrated', 'cluster_amplitude'),
           make_tensor_preset("A1,a2", "p0,P1", 'spin-integrated', 'cluster_amplitude'),
           make_tensor_preset("a1,A2", "P0,p1", 'spin-integrated', 'cluster_amplitude'),
           make_tensor_preset("a1,A2", "p0,P1", 'spin-integrated', 'cluster_amplitude'),
           make_tensor_preset("A1,A2", "P0,P1", 'spin-integrated', 'cluster_amplitude')]
    count = 0
    for tensor in a.generate_spin_cases():
        count += 1
        assert tensor in ref
        assert type(tensor) is ClusterAmplitude
        assert tensor.type_of_indices is IndicesSpinIntegrated
    assert count == len(ref)


def test_expand_hole():
    tensors = [make_tensor_preset("g0", "h1", 'spin-integrated', 'Hamiltonian'),
               make_tensor_preset("a4", "p3", 'spin-orbital', 'hole_density'),
               make_tensor_preset("a1,a2", "p0,p1", 'spin-orbital', 'cluster_amplitude'),
               make_tensor_preset("v0", "p6", 'spin-orbital', 'hole_density'),
               make_tensor_preset("h0", "p4", 'spin-orbital', 'hole_density')]
    ref = [(1, [make_tensor_preset("g0", "h1", 'spin-integrated', 'Hamiltonian'),
                make_tensor_preset("a1,a2", "p0,p1", 'spin-orbital', 'cluster_amplitude'),
                make_tensor_preset("a4", "p3", 'spin-orbital', 'Kronecker'),
                make_tensor_preset("v0", "p6", 'spin-orbital', 'Kronecker'),
                make_tensor_preset("h0", "p4", 'spin-orbital', 'Kronecker')]),
           (-1, [make_tensor_preset("g0", "h1", 'spin-integrated', 'Hamiltonian'),
                 make_tensor_preset("a1,a2", "p0,p1", 'spin-orbital', 'cluster_amplitude'),
                 make_tensor_preset("a4", "p3", 'spin-orbital', 'Kronecker'),
                 make_tensor_preset("v0", "p6", 'spin-orbital', 'Kronecker'),
                 make_tensor_preset("h0", "p4", 'spin-orbital', 'cumulant')]),
           (-1, [make_tensor_preset("g0", "h1", 'spin-integrated', 'Hamiltonian'),
                 make_tensor_preset("a1,a2", "p0,p1", 'spin-orbital', 'cluster_amplitude'),
                 make_tensor_preset("a4", "p3", 'spin-orbital', 'cumulant'),
                 make_tensor_preset("v0", "p6", 'spin-orbital', 'Kronecker'),
                 make_tensor_preset("h0", "p4", 'spin-orbital', 'Kronecker')]),
           (1, [make_tensor_preset("g0", "h1", 'spin-integrated', 'Hamiltonian'),
                make_tensor_preset("a1,a2", "p0,p1", 'spin-orbital', 'cluster_amplitude'),
                make_tensor_preset("a4", "p3", 'spin-orbital', 'cumulant'),
                make_tensor_preset("v0", "p6", 'spin-orbital', 'Kronecker'),
                make_tensor_preset("h0", "p4", 'spin-orbital', 'cumulant')])]

    count = 0
    for sign_tensors in expand_hole_densities(tensors):
        assert sign_tensors in ref
        count += 1
    assert count == 4

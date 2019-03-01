import pytest
from Tensor import make_tensor_preset
from SpaceCounter import SpaceCounter


def test_init():
    tensor1 = make_tensor_preset('Hamiltonian', "g0,g1", "g2,g3", "spin-orbital")
    tensor2 = make_tensor_preset('cluster_amplitude', "p0,p1", "h0,h1", "spin-orbital")
    sc = SpaceCounter(tensor1, tensor2)
    assert sc.upper == [0] * 6
    assert sc.lower == [0] * 6
    assert sc.size == 0

    tensor2 = make_tensor_preset('cumulant', "g2,a1", "h0,h1", "spin-orbital")
    sc = SpaceCounter(tensor1, tensor2)
    assert sc.upper == [0] * 6
    assert sc.lower == [1, 0, 0, 0, 0, 0]
    assert sc.n_upper == 0
    assert sc.n_lower == 1

    tensor1 = make_tensor_preset('Hamiltonian', "p0,P1", "h0,H1", "spin-integrated")
    tensor2 = make_tensor_preset("cluster_amplitude", "h0, H1", "p0,A0", "spin-integrated")
    sc = SpaceCounter(tensor1, tensor2)
    assert sc.upper == [0, 1, 0, 0, 0, 0] + [0] * 6
    assert sc.lower == [0, 0, 1, 0, 0, 0] + [0, 0, 1, 0, 0, 0]
    assert sc.n_upper == 1
    assert sc.n_lower == 2


def test_str():
    tensor1 = make_tensor_preset('Hamiltonian', "p0,P1", "h0,H1", "spin-integrated")
    tensor2 = make_tensor_preset("cluster_amplitude", "h0, H1", "p0,A0", "spin-integrated")
    sc = SpaceCounter(tensor1, tensor2)
    assert str(sc) == 'SpaceCounter (0,1,0,0,0,0,0,0,0,0,0,0; 0,0,1,0,0,0,0,0,1,0,0,0)'


def test_eq():
    a = SpaceCounter(make_tensor_preset('Hamiltonian', "g0,g1", "g2,g3", "spin-orbital"),
                     make_tensor_preset('cluster_amplitude', "g2,a1", "h0,h1", "spin-orbital"))
    b = SpaceCounter(make_tensor_preset('cumulant', "g0,g1", "g2,g3", "spin-orbital"),
                     make_tensor_preset('cumulant', "g2,a1", "h0,h1", "spin-orbital"))
    assert a == b

    b = SpaceCounter(make_tensor_preset('cumulant', "g0,g1", "g2,g3", "spin-integrated"),
                     make_tensor_preset('cumulant', "g2,a1", "h0,h1", "spin-integrated"))
    with pytest.raises(ValueError):
        assert a == b

    a = SpaceCounter(make_tensor_preset('Hamiltonian', "g0,g1", "g2,g3", "spin-adapted"),
                     make_tensor_preset('cumulant', "g2,a1", "h0,h1", "spin-adapted"))
    assert a == b


def test_lt():
    t1 = make_tensor_preset('Hamiltonian', "g0,p0,p1", "c0,h1,h2", 'spin-orbital')
    t2 = make_tensor_preset('cumulant', "h1,h2", "p0, p1", 'spin-orbital')
    t3 = make_tensor_preset('cumulant', "c0,h1,h2", "v0,v1,v2", 'spin-orbital')
    assert SpaceCounter(t1, t3) < SpaceCounter(t1, t2)

    t3 = make_tensor_preset('cumulant', "c0,h2", "p0,p1", 'spin-orbital')
    assert SpaceCounter(t1, t3) < SpaceCounter(t1, t2)


def test_le():
    t1 = make_tensor_preset('Hamiltonian', "g0,p0,p1", "c0,h1,h2", 'spin-orbital')
    t2 = make_tensor_preset('cumulant', "h1,h2", "p0, p1", 'spin-orbital')
    t3 = make_tensor_preset('cumulant', "h1,h2", "p0, p1", 'spin-orbital')
    assert SpaceCounter(t1, t3) <= SpaceCounter(t1, t2)

    t3 = make_tensor_preset('cumulant', "c0,h2", "p0,p1", 'spin-orbital')
    assert SpaceCounter(t1, t3) <= SpaceCounter(t1, t2)


def test_transpose():
    tensor1 = make_tensor_preset('Hamiltonian', "p0,P1", "h0,H1", "spin-integrated")
    tensor2 = make_tensor_preset("cluster_amplitude", "h0, H1", "p0,A0", "spin-integrated")
    sc = SpaceCounter(tensor1, tensor2)
    assert sc.transpose() == SpaceCounter(tensor2, tensor1)

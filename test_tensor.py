from Tensor import *


def test_tensor_class():
    tensor = Tensor("T", ['a1', 'a2'], ['c3', 'p0'])
    assert str(tensor) == "T^{ a_{1} a_{2} }_{ c_{3} p_{0} }", "Tensor format failed."
    assert tensor.ambit() == 'T2["a1,a2,c3,p0"]', "Tensor format ambit failed."
    tensor_c, tensor_sign = tensor.canonicalize_copy()
    assert tensor_c == Tensor("T", ['a1', 'a2'], ['p0', 'c3']), "Tensor canonicalize failed."
    assert tensor_sign == -1, "Tensor canonicalize sign failed."
    assert tensor_c <= tensor, "Tensor comparison <= failed."
    print("Tensor tests passed.")
import pytest
from src.Index import Index


def test_index_init():
    with pytest.raises(ValueError):
        Index("x0")
    with pytest.raises(ValueError):
        Index("ax1")
    with pytest.raises(ValueError):
        Index("a1.25")
    with pytest.raises(TypeError):
        Index(123)


def test_index_str():
    assert str(Index("p0")) == "p0"


def test_index_latex():
    assert Index("a3").latex() == "a_{3}"


def test_index_eq():
    assert Index("p0") == Index("p0")


def test_index_ne_1():
    assert Index("a0") != Index("P0")


def test_index_ne_2():
    assert Index("a0") != Index("a20")


def test_index_lt():
    assert Index("h0") < Index("G2")


def test_index_le():
    assert Index("p0") <= Index("p1")


def test_index_gt():
    assert Index("H0") > Index("a4")


def test_index_ge():
    assert Index("P1") >= Index("P1")


def test_index_spin():
    assert Index("G1").is_beta()
    assert not Index("g1").is_beta()
    assert Index("a0").to_beta() == Index("A0")
    assert Index("p0").to_alpha() == Index("p0")

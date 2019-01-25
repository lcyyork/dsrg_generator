import pytest
from Index import Index


def test_index_init_1():
    with pytest.raises(ValueError):
        Index("x0")


def test_index_init_2():
    with pytest.raises(ValueError):
        Index("ax1")


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

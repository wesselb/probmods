import pytest

from probmods.test import unequal


def test_unequal():
    with pytest.raises(RuntimeError):
        unequal(1, 2)


def test_unequal_atol():
    unequal(1, 2, atol=0.5)
    with pytest.raises(AssertionError):
        # Must be strictly more than `atol`.
        unequal(1, 1.5, atol=0.5)
    with pytest.raises(AssertionError):
        unequal(1, 1.25, atol=0.5)


def test_unequal_rtol():
    unequal(10, 11, rtol=0.05)
    with pytest.raises(AssertionError):
        # Must be strictly more than `rtol`.
        unequal(10, 10.5, rtol=0.05)
    with pytest.raises(AssertionError):
        unequal(10, 10.25, rtol=0.05)

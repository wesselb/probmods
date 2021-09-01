import lab as B
from plum import Dispatcher
from numpy.testing import assert_allclose

__all__ = ["approx", "unequal", "Regularisation"]

_dispatch = Dispatcher()


@_dispatch
def approx(x, y, atol=1e-10, rtol=1e-8, assert_dtype=False):
    """Assert that two numerical inputs are equal.

    Args:
        x (tensor): First input.
        y (tensor): Second input.
        atol (float, optional): Absolute tolerance. Defaults to `1e-10`.
        rtol (float, optional): Relative tolerance. Defaults to `1e-8`.
        assert_dtype (bool, optional): Assert that `x` and `y` have the same data type.
    """
    assert_allclose(
        B.to_numpy(B.dense(x)),
        B.to_numpy(B.dense(y)),
        atol=atol,
        rtol=rtol,
    )
    if assert_dtype:
        assert B.dtype(x) == B.dtype(x)


@_dispatch
def approx(x: tuple, y: tuple, **kw_args):
    assert len(x) == len(y)
    for xi, yi in zip(x, y):
        approx(xi, yi, **kw_args)


def unequal(x, y, atol=None, rtol=None):
    """Assert that two numerical inputs are unequal.

    Args:
        x (tensor): First input.
        y (tensor): Second input.
        atol (float, optional): Lower bound on the mean absolute difference.
        rtol (float, optional): Lower bound on the mean absolute difference relative to
            the mean absolute values.
    """
    if not atol and not rtol:
        raise RuntimeError("Must specify either `atol` or `rtol`.")
    x, y = B.to_numpy(x, y)
    diff = B.mean(B.abs(x - y))
    if atol:
        assert diff > atol
    if rtol:
        assert diff / B.maximum(B.mean(B.abs(x)), B.mean(B.abs(y))) > rtol


class Regularisation:
    """A context manager which temporarily changes `B.epsilon`.

    Args:
        epsilon (float): New value.
    """

    def __init__(self, epsilon):
        self.epsilon = epsilon
        self.old_epsilon = None

    def __enter__(self):
        self.old_epsilon = B.epsilon
        B.epsilon = self.epsilon

    def __exit__(self, exc_type, exc_val, exc_tb):
        B.epsilon = self.old_epsilon

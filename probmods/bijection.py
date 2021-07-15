import inspect
import re

import lab as B
from matrix import AbstractMatrix, TiledBlocks
from plum import Union, Dispatcher, Tuple

__all__ = [
    "Bijection",
    "Composition",
    "Normaliser",
    "Log",
    "Squishing",
    "parse",
]

_dispatch = Dispatcher()

_Numeric = Union[B.Numeric, AbstractMatrix]


class Bijection:
    """A bijection."""

    def transform(self, x, *args):
        """Transform `x`. You can also run `transform` by simply calling the object.

        Args:
            x (object): Objects to transform.
            i (int, optional): Index of `x` to transform.

        Returns:
            object: Transformed version of `x`.
        """
        return transform(self, x, *args)

    def __call__(self, *args):
        return self.transform(*args)

    def untransform(self, y, *args):
        """Untransform `y`.

        Args:
            y (object): Object to untransform.
            i (int, optional): Index of `y` to untransform.

        Returns:
            object: Original version of `y` before the transform.
        """
        return untransform(self, y, *args)

    def logdet(self, x, *args):
        """Compute the log-determinant of the Jacobian of the transform.

        Args:
            x (object): Object to transform.
            i (int, optional): Index of `x` to transform.

        Returns:
            scalar: Log-determinant of the Jocobian at `x`.
        """
        return logdet(self, x, *args)


class Composition(Bijection):
    """Composition of bijections.

    Args:
        *bijections (:class:`.Bijection`)
    """

    def __init__(self, *bijections):
        self.bijections = bijections


@_dispatch
def transform(b: Composition, x, *args):
    for bi in reversed(b.bijections):
        x = transform(bi, x, *args)
    return x


@_dispatch
def untransform(b: Composition, y, *args):
    for bi in b.bijections:
        y = untransform(bi, y, *args)
    return y


@_dispatch
def logdet(b: Composition, y, *args):
    logdet_sum = 0
    for bi in b.bijections[:-1]:
        logdet_sum = logdet_sum + logdet(bi, y, *args)
        y = untransform(bi, y, *args)
    return logdet_sum + logdet(b.bijections[-1], y, *args)


def _extend_i(f):
    signature = inspect.signature(f)
    T1, T2 = [signature.parameters[p].annotation for p in list(signature.parameters)]

    def extension(b: T1, x: T2, i: B.Int):
        return f(b, x)

    extension.__name__ = f.__name__
    extension.__qualname__ = f.__qualname__  # Important for dispatch!
    _dispatch(extension)
    return f


class Identity(Bijection):
    """Identity bijection."""


@_dispatch
@_extend_i
def transform(b: Identity, x):
    return x


@_dispatch
@_extend_i
def untransform(b: Identity, y):
    return y


@_dispatch
@_extend_i
def logdet(b: Identity, y):
    return B.zero(y)


class Normaliser(Bijection):
    """Create a data normaliser."""

    def __init__(self):
        self._fit = False
        self._mean = None
        self._scale = None

    def _get_mean_scale(self, x, i=None, fit=False):
        if self._fit:
            if i is None:
                return self._mean, self._scale
            else:
                return self._mean[0, i], self._scale[0, i]
        else:
            if fit:
                self._mean = B.nanmean(x, axis=0, squeeze=False)
                self._scale = B.nanstd(x, axis=0, squeeze=False)
                self._fit = True
                return self._get_mean_scale(x, i)
            else:
                if i is None:
                    dtype = B.dtype(x)
                    n = B.shape(x, 0)
                    return B.zeros(dtype, n, 1), B.ones(dtype, n, 1)
                else:
                    return B.zero(x), B.one(x)


@_dispatch
def transform(b: Normaliser, x: _Numeric):
    mean, scale = b._get_mean_scale(x, fit=True)
    return (x - mean) / scale


@_dispatch
def transform(b: Normaliser, x: _Numeric, i: B.Int):
    mean, scale = b._get_mean_scale(x, i)
    return (x - mean) / scale


@_dispatch
def transform(b: Normaliser, x: TiledBlocks):
    # `x` is a variance.
    mean, scale = b._get_mean_scale(x)
    return TiledBlocks(
        *[
            (block / scale / B.transpose(scale), rep)
            for block, rep in zip(x.blocks, x.reps)
        ],
        axis=x.axis,
    )


@_dispatch
def untransform(b: Normaliser, y: _Numeric):
    mean, scale = b._get_mean_scale(y)
    return y * scale + mean


@_dispatch
def untransform(b: Normaliser, y: _Numeric, i: B.Int):
    mean, scale = b._get_mean_scale(y, i)
    return y * scale + mean


@_dispatch
def untransform(b: Normaliser, y: Tuple[_Numeric, _Numeric]):
    mean, scale = b._get_mean_scale(y)
    dist_mean, dist_var = y
    if B.rank(dist_var) == 2:
        return (
            dist_mean * scale + mean,
            dist_var * scale ** 2,
        )
    elif B.rank(dist_var) == 3:
        return (
            dist_mean * scale + mean,
            dist_var * scale[:, :, None] * scale[:, None, :],
        )
    else:
        raise ValueError(f"Invalid rank {B.rank(dist_var)} of the variance.")


@_dispatch
def logdet(b: Normaliser, y: _Numeric):
    _, scale = b._get_mean_scale(y)
    return -B.shape(y, 0) * B.sum(B.log(scale))


@_dispatch
def logdet(b: Normaliser, y: _Numeric, i: B.Int):
    _, scale = b._get_mean_scale(y, i)
    return -B.shape(y, 0) * B.log(scale)


class Log(Bijection):
    """Log-transform for positive data."""


@_dispatch
@_extend_i
def transform(b: Log, x: _Numeric):
    return B.log(x)


@_dispatch
@_extend_i
def untransform(b: Log, y: _Numeric):
    return B.exp(y)


@_dispatch
def untransform(b: Log, y: Tuple[_Numeric, _Numeric]):
    mean, var = y
    if B.rank(var) == 2:
        pass
    elif B.rank(var) == 3:
        raise NotImplementedError(f"Log-transform not implemented for joint variances.")
    else:
        raise ValueError(f"Invalid rank {B.rank(var)} of the variance.")
    # These are the mean and variance of the log-normal distribution.
    return (
        B.exp(mean + 0.5 * var),
        (B.exp(var) - 1) * B.exp(2 * mean + var),
    )


@_dispatch
@_extend_i
def logdet(b: Log, y: _Numeric):
    return -B.nansum(y)


class Squishing(Bijection):
    """Squishing transform for real-valued, heavy-tailed data."""


@_dispatch
@_extend_i
def transform(b: Squishing, x: _Numeric):
    return B.sign(x) * B.log(1 + B.abs(x))


@_dispatch
@_extend_i
def untransform(b: Squishing, y: _Numeric):
    return B.sign(y) * (B.exp(B.abs(y)) - 1)


@_dispatch
@_extend_i
def logdet(b: Squishing, y: _Numeric):
    return -B.nansum(B.abs(y))


@_dispatch
def parse(spec: str):
    """Build a bijection from a string specification.

    The specifications "normaliser" or "normalise" build a normalising transformation.
    The specifications "log" or "positive" build a log-transform. The specification
    "squishing" builds a squishing transformation. These transformations can be composed
    with a comma or plus, e.g. "normaliser,log" or "normaliser+log". Specifications are
    case insensitive.

    Args:
        spec (str): Specification.

    Returns:
        :class:`.Bijection`: Bijection corresponding to specification.
    """
    if not spec:
        return Identity()

    parts = [part.strip().lower() for part in re.split(r"[,+]", spec)]
    bijections = []
    for part in parts:
        if part in {"normaliser", "normalise"}:
            bijections.append(Normaliser())
        elif part in {"log", "positive"}:
            bijections.append(Log())
        elif part == "squishing":
            bijections.append(Squishing())
        else:
            raise ValueError(f'Unknown bijection "{part}".')
    return Composition(*bijections)


@_dispatch
def parse(bijection: Bijection):
    return bijection


@_dispatch
def parse(bijection: type(None)):
    return Identity()

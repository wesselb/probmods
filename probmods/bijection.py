import inspect
import re

import lab as B
from matrix import AbstractMatrix, TiledBlocks
from plum import Union, Dispatcher, Tuple, parametric
from plum.parametric import CovariantMeta

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


class Bijection(metaclass=CovariantMeta):
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


@parametric
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
def logdet(b: Composition, x, *args):
    logdet_sum = 0
    for bi in reversed(b.bijections[1:]):
        logdet_sum = logdet_sum + logdet(bi, x, *args)
        x = transform(bi, x, *args)
    return logdet_sum + logdet(b.bijections[0], x, *args)


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
                if B.rank(x) < 2:
                    return B.squeeze(self._mean), B.squeeze(self._scale)
                else:
                    return self._mean, self._scale
            else:
                if B.rank(self._mean) == 2 and B.rank(self._scale) == 2:
                    return self._mean[0, i], self._scale[0, i]
                else:
                    # Must be scalar, so safe to just return.
                    return self._mean, self._scale
        else:
            if fit:
                if B.rank(x) == 0:
                    self._mean = x
                    self._scale = 1
                elif B.rank(x) == 1:
                    self._mean = B.nanmean(x)
                    self._scale = B.nanstd(x)
                    if B.jit_to_numpy(self._scale) == 0:
                        self._scale = 1
                elif B.rank(x) == 2:
                    self._mean = B.nanmean(x, axis=0, squeeze=False)
                    self._scale = B.nanstd(x, axis=0, squeeze=False)
                    self._scale = B.where(
                        B.jit_to_numpy(self._scale) == 0,
                        B.ones(self._scale),
                        self._scale,
                    )
                else:
                    raise ValueError(f"Invalid rank of `x` {B.rank(x)}.")
                self._fit = True
                return self._get_mean_scale(x, i)
            else:
                if B.rank(x) in {0, 1}:
                    return B.zero(x), B.one(x)
                elif B.rank(x) == 2:
                    if i is None:
                        dtype = B.dtype(x)
                        n = B.shape(x, 0)
                        return B.zeros(dtype, n, 1), B.ones(dtype, n, 1)
                    else:
                        return B.zero(x), B.one(x)
                else:
                    raise ValueError(f"Invalid rank of `x` {B.rank(x)}.")


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
def transform(b: Normaliser, y: Tuple[_Numeric, _Numeric]):
    dist_mean, dist_var = y
    mean, scale = b._get_mean_scale(dist_mean)
    if B.rank(dist_var) in {0, 1, 2}:
        return (
            (dist_mean - mean) / scale,
            dist_var / scale ** 2,
        )
    elif B.rank(dist_var) == 3:
        scale = B.uprank(scale, rank=2)
        return (
            (dist_mean - mean) / scale,
            dist_var / scale[:, :, None] / scale[:, None, :],
        )
    else:
        raise ValueError(f"Invalid rank {B.rank(dist_var)} of the variance.")


@_dispatch
def untransform(b: Normaliser, y: _Numeric):
    mean, scale = b._get_mean_scale(y)
    return y * scale + mean


@_dispatch
def untransform(b: Normaliser, y: _Numeric, i: B.Int):
    mean, scale = b._get_mean_scale(y, i)
    return y * scale + mean


@_dispatch
def untransform(b: Normaliser, x: TiledBlocks):
    # `x` is a variance.
    mean, scale = b._get_mean_scale(x)
    return TiledBlocks(
        *[
            (block * scale * B.transpose(scale), rep)
            for block, rep in zip(x.blocks, x.reps)
        ],
        axis=x.axis,
    )


@_dispatch
def untransform(b: Normaliser, y: Tuple[_Numeric, _Numeric]):
    dist_mean, dist_var = y
    mean, scale = b._get_mean_scale(dist_mean)
    if B.rank(dist_var) in {0, 1, 2}:
        return (
            dist_mean * scale + mean,
            dist_var * scale ** 2,
        )
    elif B.rank(dist_var) == 3:
        scale = B.uprank(scale, rank=2)
        return (
            dist_mean * scale + mean,
            dist_var * scale[:, :, None] * scale[:, None, :],
        )
    else:
        raise ValueError(f"Invalid rank {B.rank(dist_var)} of the variance.")


@_dispatch
def logdet(b: Normaliser, x: _Numeric):
    _, scale = b._get_mean_scale(x)
    return -B.sum(B.ones(x) * B.log(scale))


@_dispatch
def logdet(b: Normaliser, x: _Numeric, i: B.Int):
    _, scale = b._get_mean_scale(x, i)
    return -B.sum(B.ones(x) * B.log(scale))


class Log(Bijection):
    """Log-transform for positive data."""


@_dispatch
@_extend_i
def transform(b: Log, x: _Numeric):
    return B.log(x)


@_dispatch
def transform(b: Log, y: Tuple[_Numeric, _Numeric]):
    log_mean, log_var = y
    if B.rank(log_var) in {0, 1, 2}:
        pass
    elif B.rank(log_var) == 3:
        raise NotImplementedError(f"Log-transform not implemented for joint variances.")
    else:
        raise ValueError(f"Invalid rank {B.rank(log_var)} of the variance.")
    # These are the mean and variance of the normal distribution.
    var = B.log(log_var / log_mean ** 2 + 1)
    return (B.log(log_mean) - 0.5 * var, var)


@_dispatch
@_extend_i
def untransform(b: Log, x: _Numeric):
    return B.exp(x)


@_dispatch
def untransform(b: Log, y: Tuple[_Numeric, _Numeric]):
    mean, var = y
    if B.rank(var) in {0, 1, 2}:
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
def logdet(b: Log, x: _Numeric):
    return -B.nansum(B.log(x))


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
def logdet(b: Squishing, x: _Numeric):
    return -B.nansum(B.log(1 + B.abs(x)))


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
def parse(bijection: None):
    return Identity()

import inspect
import re

import lab as B
from matrix import AbstractMatrix, TiledBlocks
from plum import (
    Union,
    Dispatcher,
    Tuple,
    parametric,
    Signature,
    type_of,
    NotFoundLookupError,
)
from plum.parametric import CovariantMeta
from varz import Vars, Struct

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

_NoParameters = None
_Parameters = Union[Vars, Struct]
_MaybeParameters = Union[_NoParameters, _Parameters]


class Bijection(metaclass=CovariantMeta):
    """A bijection."""

    def transform(self, *args):
        """Transform `x`. You can also run `transform` by simply calling the object.

        Args:
            ps (object, optional): Parameter struct to take parameters from.
            x (object): Objects to transform.
            i (int, optional): Index of `x` to transform.

        Returns:
            object: Transformed version of `x`.
        """
        return transform(self, *args)

    def __call__(self, *args):
        return self.transform(*args)

    def untransform(self, *args):
        """Untransform `y`.

        Args:
            ps (object, optional): Parameter struct to take parameters from.
            y (object): Object to untransform.
            i (int, optional): Index of `y` to untransform.

        Returns:
            object: Original version of `y` before the transform.
        """
        return untransform(self, *args)

    def logdet(self, *args):
        """Compute the log-determinant of the Jacobian of the transform.

        Args:
            ps (object, optional): Parameter struct to take parameters from.
            x (object): Object to transform.
            i (int, optional): Index of `x` to transform.

        Returns:
            scalar: Log-determinant of the Jocobian at `x`.
        """
        return logdet(self, *args)


# Default to no-parameter methods by prepending `None`. However, be sure to prevent
# recursions!


@_dispatch
def transform(b: Bijection, x, *args):
    # Prevent recursion.
    if x is None:
        sig = Signature(type_of(b), *(type_of(y) for y in (x,) + args))
        raise NotFoundLookupError(
            f"No bijection `transform` implementation for signature {sig}."
        )
    return transform(b, None, x, *args)


@_dispatch
def untransform(b: Bijection, y, *args):
    # Prevent recursion.
    if y is None:
        sig = Signature(type_of(b), *(type_of(x) for x in (y,) + args))
        raise NotFoundLookupError(
            f"No bijection `untransform` implementation for signature {sig}."
        )
    return untransform(b, None, y, *args)


@_dispatch
def logdet(b: Bijection, x, *args):
    # Prevent recursion.
    if x is None:
        sig = Signature(type_of(b), *(type_of(y) for y in (x,) + args))
        raise NotFoundLookupError(
            f"No bijection `logdet` implementation for signature {sig}."
        )
    return logdet(b, None, x, *args)


@parametric
class Composition(Bijection):
    """Composition of bijections.

    Args:
        *bijections (:class:`.Bijection`)
    """

    def __init__(self, *bijections):
        self.bijections = bijections


@_dispatch
def _ensure_maybe_parameters_list(ps: None, n: B.Int):
    return [None] * n


@_dispatch
def _ensure_maybe_parameters_list(ps: Vars, n: B.Int):
    return _ensure_maybe_parameters_list(ps.struct, n)


@_dispatch
def _ensure_maybe_parameters_list(ps: Struct, n: B.Int):
    return [ps[i] for i in range(n)]


@_dispatch
def transform(b: Composition, ps: _MaybeParameters, x, *args):
    ps = _ensure_maybe_parameters_list(ps, len(b.bijections))
    for psi, bi in reversed(list(zip(ps, b.bijections))):
        x = transform(bi, psi, x, *args)
    return x


@_dispatch
def untransform(b: Composition, ps: _MaybeParameters, y, *args):
    ps = _ensure_maybe_parameters_list(ps, len(b.bijections))
    for psi, bi in zip(ps, b.bijections):
        y = untransform(bi, psi, y, *args)
    return y


@_dispatch
def logdet(b: Composition, ps: _MaybeParameters, x, *args):
    ps = _ensure_maybe_parameters_list(ps, len(b.bijections))
    logdet_sum = 0
    for psi, bi in reversed(list(zip(ps[1:], b.bijections[1:]))):
        logdet_sum = logdet_sum + logdet(bi, psi, x, *args)
        x = transform(bi, psi, x, *args)
    return logdet_sum + logdet(b.bijections[0], ps[0], x, *args)


def _extend_i(f):
    sig = inspect.signature(f)
    T1, T2, T3 = [sig.parameters[p].annotation for p in list(sig.parameters)]

    def extension(b: T1, ps: T2, x: T3, i: B.Int):
        return f(b, ps, x)

    extension.__name__ = f.__name__
    extension.__qualname__ = f.__qualname__  # Important for dispatch!
    _dispatch(extension)
    return f


class Identity(Bijection):
    """Identity bijection."""


@_dispatch
@_extend_i
def transform(b: Identity, ps: _MaybeParameters, x):
    return x


@_dispatch
@_extend_i
def untransform(b: Identity, ps: _MaybeParameters, y):
    return y


@_dispatch
@_extend_i
def logdet(b: Identity, ps: _MaybeParameters, y):
    return B.zero(y)


class Normaliser(Bijection):
    """Create a data normaliser."""

    def __init__(self):
        self._fit = False
        self._mean = None
        self._scale = None

    def _get_mean_scale(self, ps, x, i=None, fit=False):
        if self._fit:
            # Eventually process the tensors as `f(tensor[index])`.
            f = lambda x: x
            index = None

            if i is None:
                if B.rank(x) < 2:
                    # Squeeze away possible extra dimensions.
                    f = B.squeeze
                else:
                    # Nothing to do.
                    # NOTE: For some reason, CI thinks this branch is not tested, which
                    #   is absolutely false. Not sure what's going on.
                    pass  # pragma: no cover
            else:
                if B.rank(self._mean) == 2 and B.rank(self._scale) == 2:
                    # Get the right element.
                    index = (0, i)
                else:
                    # Must be scalar, so safe to just return.
                    pass

            # Perform parametrisation and transformation.
            mean, scale = self._mean, self._scale
            if ps is not None:
                mean = ps.mean.unbounded(mean)
                scale = ps.scale.positive(scale)
            if index:
                mean = mean[index]
                scale = scale[index]
            return f(mean), f(scale)

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
                return self._get_mean_scale(ps, x, i)

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


def _safe(op, x, y):
    """Perform a binary operation in a way that safely handles NaNs.

    Args:
        op (function): Binary operation.
        x (tensor): Tensor possibly containing NaNs.
        y (tensor): Tensor without NaNs.

    Returns:
        tensor: `op(x, y)` but with NaNs excluded from the computation.
    """
    shape = B.shape_broadcast(x, y)
    x = B.broadcast_to(x, *shape)
    y = B.broadcast_to(y, *shape)
    available = B.jit_to_numpy(~B.isnan(x))
    x_safe = B.where(available, x, B.zero(x))
    return B.where(available, op(x_safe, y), x)


@_dispatch
def transform(b: Normaliser, ps: _MaybeParameters, x: _Numeric):
    mean, scale = b._get_mean_scale(ps, x, fit=True)
    return _safe(B.divide, _safe(B.subtract, x, mean), scale)


@_dispatch
def transform(b: Normaliser, ps: _MaybeParameters, x: _Numeric, i: B.Int):
    mean, scale = b._get_mean_scale(ps, x, i)
    return _safe(B.divide, _safe(B.subtract, x, mean), scale)


@_dispatch
def transform(b: Normaliser, ps: _MaybeParameters, x: TiledBlocks):
    # `x` is a variance.
    mean, scale = b._get_mean_scale(ps, x)
    return TiledBlocks(
        *[
            (block / scale / B.transpose(scale), rep)
            for block, rep in zip(x.blocks, x.reps)
        ],
        axis=x.axis,
    )


@_dispatch
def transform(b: Normaliser, ps: _MaybeParameters, y: Tuple[_Numeric, _Numeric]):
    dist_mean, dist_var = y
    mean, scale = b._get_mean_scale(ps, dist_mean)
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
def untransform(b: Normaliser, ps: _MaybeParameters, y: _Numeric):
    mean, scale = b._get_mean_scale(ps, y)
    return y * scale + mean


@_dispatch
def untransform(b: Normaliser, ps: _MaybeParameters, y: _Numeric, i: B.Int):
    mean, scale = b._get_mean_scale(ps, y, i)
    return y * scale + mean


@_dispatch
def untransform(b: Normaliser, ps: _MaybeParameters, x: TiledBlocks):
    # `x` is a variance.
    mean, scale = b._get_mean_scale(ps, x)
    return TiledBlocks(
        *[
            (block * scale * B.transpose(scale), rep)
            for block, rep in zip(x.blocks, x.reps)
        ],
        axis=x.axis,
    )


@_dispatch
def untransform(b: Normaliser, ps: _MaybeParameters, y: Tuple[_Numeric, _Numeric]):
    dist_mean, dist_var = y
    mean, scale = b._get_mean_scale(ps, dist_mean)
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
def logdet(b: Normaliser, ps: _MaybeParameters, x: _Numeric):
    _, scale = b._get_mean_scale(ps, x)
    return -B.sum(B.ones(x) * B.log(scale))


@_dispatch
def logdet(b: Normaliser, ps: _MaybeParameters, x: _Numeric, i: B.Int):
    _, scale = b._get_mean_scale(ps, x, i)
    return -B.sum(B.ones(x) * B.log(scale))


class Log(Bijection):
    """Log-transform for positive data."""


@_dispatch
@_extend_i
def transform(b: Log, ps: _MaybeParameters, x: _Numeric):
    return B.log(x)


@_dispatch
def transform(b: Log, ps: _MaybeParameters, y: Tuple[_Numeric, _Numeric]):
    log_mean, log_var = y
    if B.rank(log_var) in {0, 1, 2}:
        pass
    elif B.rank(log_var) == 3:
        raise NotImplementedError(f"Log-transform not implemented for joint variances.")
    else:
        raise ValueError(f"Invalid rank {B.rank(log_var)} of the variance.")
    # These are the mean and variance of the log-normal distribution.
    var = B.log(log_var / log_mean ** 2 + 1)
    return (B.log(log_mean) - 0.5 * var, var)


@_dispatch
@_extend_i
def untransform(b: Log, ps: _MaybeParameters, x: _Numeric):
    return B.exp(x)


@_dispatch
def untransform(b: Log, ps: _MaybeParameters, y: Tuple[_Numeric, _Numeric]):
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
def logdet(b: Log, ps: _MaybeParameters, x: _Numeric):
    return -B.nansum(B.log(x))


class Squishing(Bijection):
    """Squishing transform for real-valued, heavy-tailed data."""


@_dispatch
@_extend_i
def transform(b: Squishing, ps: _MaybeParameters, x: _Numeric):
    return B.sign(x) * B.log(1 + B.abs(x))


@_dispatch
@_extend_i
def untransform(b: Squishing, ps: _MaybeParameters, y: _Numeric):
    return B.sign(y) * (B.exp(B.abs(y)) - 1)


@_dispatch
@_extend_i
def logdet(b: Squishing, ps: _MaybeParameters, x: _Numeric):
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

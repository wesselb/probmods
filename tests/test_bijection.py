import jax
import jax.numpy as jnp
import lab.jax as B
import tensorflow as tf
import numpy as np
import pytest
from matrix import TiledBlocks
from plum import NotFoundLookupError
from stheno import Normal
from varz import Vars

from probmods.bijection import (
    Bijection,
    Identity,
    Log,
    Squishing,
    _ensure_maybe_parameters_list,
    Composition,
    _safe,
    Normaliser,
    parse,
)
from .util import approx


def check_bijection(bijection, x, *args):
    y = bijection(x, *args)

    # Check that transforming and untransforming preserves the shape.
    assert B.shape(bijection(x, *args)) == B.shape(x)
    assert B.shape(bijection.transform(x, *args)) == B.shape(x)
    assert B.shape(bijection.untransform(y, *args)) == B.shape(x)

    # Check that the transforms are implemented correctly.
    approx(bijection.untransform(y, *args), x)

    # Check that the log-determinant is implemented correctly.
    def transform(x_):
        x_ = B.reshape(x_, *B.shape(x))
        y_ = bijection.transform(x_, *args)
        assert B.shape(y_) == B.shape(y)
        return B.flatten(y_)

    approx(
        bijection.logdet(x, *args),
        B.log(B.det(jax.jacfwd(transform)(B.flatten(x)))),
        rtol=1e-6,
        atol=1e-6,
    )

    # Check that the log-determinant can handle NaNs.
    if B.length(x) > 1:
        # We can modify it because we won't use it afterwards.
        x[0] = B.nan
        assert ~B.isnan(bijection.logdet(x, *args))


def check_bijection_simple(bijection, x):
    y = bijection(x)
    assert isinstance(y, type(x))
    assert isinstance(bijection.untransform(y), type(x))
    approx(bijection.untransform(y), x)


def test_fallback_methods():
    class MyBijection(Bijection):
        pass

    b = MyBijection()

    with pytest.raises(NotFoundLookupError) as e:
        b(1)
    assert "No bijection `transform` implementation" in str(e.value)

    with pytest.raises(NotFoundLookupError) as e:
        b.untransform(1)
    assert "No bijection `untransform` implementation" in str(e.value)

    with pytest.raises(NotFoundLookupError) as e:
        b.logdet(1)
    assert "No bijection `logdet` implementation" in str(e.value)


shape_args = [
    ((), ()),
    ((), (0,)),
    ((1,), ()),
    ((1,), (0,)),
    ((5,), ()),
    ((5,), (0,)),
    ((5, 1), ()),
    ((5, 1), (0,)),
    ((1, 5), ()),
    ((5, 5), ()),
]


@pytest.mark.parametrize("shape, args", shape_args)
def test_identity(shape, args):
    check_bijection(Identity(), B.randn(*shape), *args)


def test_safe():
    def f_safe(x, y):
        return B.nanmean(_safe(B.multiply, x, y))

    def f_unsafe(x, y):
        return B.nanmean(x * y)

    x = tf.constant([1.0, np.nan, 2.0])
    y = tf.constant([3.0])

    with tf.GradientTape() as tape:
        tape.watch(y)
        grad_unsafe = tape.gradient(f_unsafe(x, y), y)[0]

    with tf.GradientTape() as tape:
        tape.watch(y)
        grad_safe = tape.gradient(f_safe(x, y), y)[0]

    assert B.isnan(grad_unsafe)
    assert grad_safe == 1.5


def _zeros_to_ones(x):
    if B.is_scalar(x):
        if x == 0:
            return 1
        else:
            return x
    else:
        return B.where(x == 0, B.ones(x), x)


@pytest.mark.parametrize("shape, args", shape_args)
def test_normaliser(shape, args):
    bijection = Normaliser()
    x = B.randn(*shape)

    # Before training, untransforming should do nothing.
    approx(bijection.untransform(x, *args), x)
    assert B.shape(bijection.untransform(x, *args)) == B.shape(x)

    # Train the bijection.
    assert not bijection._fit
    bijection(x)  # Fit the bijection. Don't provide `args`!
    y = bijection(x, *args)
    approx(y, bijection(x, *args))  # Output should be consistent.
    assert bijection._fit
    assert B.shape(y) == B.shape(x)

    # After training, transforming should make it mean zero and variance one.
    if B.rank(x) == 0:
        approx(y, 0)
    elif B.rank(x) == 1:
        approx(B.mean(y), 0, atol=1e-8)
        approx(_zeros_to_ones(B.std(y)), 1, rtol=1e-8)
    elif B.rank(x) == 2:
        approx(B.mean(y, axis=0), 0, atol=1e-8)
        approx(_zeros_to_ones(B.std(y, axis=0)), 1, rtol=1e-8)
    else:
        raise RuntimeError(f"Cannot test `x` with rank {B.rank(x)}.")

    check_bijection(bijection, B.randn(*shape), *args)


def test_normaliser_correctness():
    n = 10
    p = 5

    # Construct a test distribution.
    chol = B.randn(p, p) + B.eye(p)
    dist = Normal(B.randn(p, 1), chol @ chol.T)

    # Generate some random data. We would like to fit `dist` to a normalised version
    # of the data.
    x = B.randn(n, p)

    # Create and train a normaliser.
    bijection = Normaliser()
    bijection(x)
    assert bijection._fit

    # Construct the transformed distribution.
    b_mean = B.uprank(B.squeeze(bijection._mean), rank=1)
    b_scale = B.uprank(B.squeeze(bijection._scale), rank=1)
    dist_untransformed = Normal(
        dist.mean * b_scale[:, None] + b_mean[:, None],
        dist.var * b_scale[:, None] * b_scale[None, :],
    )

    # Check correctness.
    x = B.randn(n, p)
    y = bijection(x)
    approx(
        B.sum(dist_untransformed.logpdf(x.T)),
        B.sum(dist.logpdf(y.T)) + bijection.logdet(x),
    )


def test_normaliser_checks():
    with pytest.raises(ValueError):
        Normaliser().transform(B.randn(5, 5, 5))
    with pytest.raises(ValueError):
        Normaliser().untransform(B.randn(5, 5, 5))


def test_normaliser_tiledblocks():
    bijection = Normaliser()
    bijection(B.randn(10, 5))
    x = TiledBlocks(B.randn(5, 5), 5)
    check_bijection_simple(bijection, x)


shape_train_shape_eval = [
    ((5,), ()),
    ((5,), (1,)),
    ((5,), (10,)),
    ((5,), (10, 1)),
    ((5,), (1, 5)),
    ((5,), (10, 5)),
    ((10, 5), (10, 5)),
]


@pytest.mark.parametrize("shape_train, shape_eval", shape_train_shape_eval)
def test_normaliser_tuple(shape_train, shape_eval):
    bijection = Normaliser()
    bijection(B.randn(*shape_train))

    # Marginal variances:
    x = (B.randn(*shape_eval), B.rand(*shape_eval))
    check_bijection_simple(bijection, x)

    # Batches of covariance matrices:
    if len(shape_eval) == 2:
        chol = B.randn(*shape_eval, shape_eval[-1])
        x = (B.randn(*shape_eval), chol @ B.transpose(chol))
        check_bijection_simple(bijection, x)

    # Variance rank check:
    for method in [bijection.transform, bijection.untransform]:
        with pytest.raises(ValueError):
            method((B.randn(2, 2), B.randn(2, 2, 2, 2)))


@pytest.mark.parametrize("shape, args", shape_args)
def test_log(shape, args):
    check_bijection(Log(), B.rand(*shape), *args)


@pytest.mark.parametrize("shape_train, shape_eval", shape_train_shape_eval)
def test_log_tuple(shape_train, shape_eval):
    bijection = Log()

    # Marginal variances:
    check_bijection_simple(bijection, (B.rand(*shape_eval), B.rand(*shape_eval)))

    # Variance check:
    for method in [bijection.transform, bijection.untransform]:
        with pytest.raises(NotImplementedError):
            method((B.randn(2, 2), B.randn(2, 2, 2)))
        with pytest.raises(ValueError):
            method((B.randn(2, 2), B.randn(2, 2, 2, 2)))


@pytest.mark.parametrize("shape, args", shape_args)
def test_squishing(shape, args):
    check_bijection(Squishing(), B.randn(*shape), *args)


def test_ensure_maybe_parameters_list():
    def path_equal(x, y):
        return [xi._path for xi in x] == [yi._path for yi in y]

    vs = Vars(np.float64)

    assert _ensure_maybe_parameters_list(None, 3) == [None, None, None]
    assert path_equal(
        _ensure_maybe_parameters_list(vs, 2),
        [vs.struct[0], vs.struct[1]],
    )
    assert path_equal(
        _ensure_maybe_parameters_list(vs.struct, 2),
        [vs.struct[0], vs.struct[1]],
    )


@pytest.mark.parametrize("shape, args", shape_args)
def test_composition(shape, args):
    check_bijection(
        Composition(Normaliser(), Squishing(), Log()), B.rand(*shape), *args
    )


def test_parse():
    assert type(parse(None)) == Identity
    assert type(parse("")) == Identity
    assert type(parse("positive")) == Composition[Log]
    assert type(parse("normalise")) == Composition[Normaliser]
    assert type(parse("squishing")) == Composition[Squishing]
    assert type(parse("normalise, positive")) == Composition[Normaliser, Log]
    assert type(parse("normalise + positive")) == Composition[Normaliser, Log]

    bijection = Log()
    assert parse(bijection) is bijection

    with pytest.raises(ValueError):
        parse("invalid")

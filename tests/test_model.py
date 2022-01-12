import lab as B
import lab.tensorflow  # noqa
import lab.jax  # noqa
import numpy as np
import jax.numpy as jnp
import pytest
import tensorflow as tf
from stheno import EQ, GP
from varz import Vars

from probmods import (
    Model,
    instancemethod,
    priormethod,
    posteriormethod,
    cast,
    Transformed,
)
from probmods.model import _same_framework, _cast, _to_np, _safe_dtype, format_class_of
from probmods.test import check_model


def test_same_framework():
    assert not _same_framework(tf.float64, 2)
    # Check non-numeric inputs.
    assert not _same_framework(tf.float64, "2")
    # Check different data types.
    assert _same_framework(tf.float64, B.ones(tf.float32, 3))
    assert _same_framework(tf.float32, B.ones(tf.float32, 3))
    # Check structured objects.
    assert _same_framework(tf.float64, (B.ones(tf.float32, 3), 2))
    assert _same_framework(tf.float64, (B.ones(tf.float32, 3), "2"))
    assert not _same_framework(tf.float64, (2, 2))


def test_internal_cast(monkeypatch):
    # Test conversion of floating-point tensors.
    assert B.dtype(_cast(tf.float32, B.ones(1))) == tf.float32
    # Check that scalars are ignored.
    assert _cast(tf.float32, 1) is 1
    assert _cast(tf.float32, 1.0) is 1.0
    # Check that non-numeric objects are ignored.
    assert _cast(tf.float32, "1") is "1"
    # Check that the tensor is moved to the active device.
    monkeypatch.setattr(B, "to_active_device", lambda x: "on_active_device")
    assert _cast(tf.float32, B.ones(1)) == "on_active_device"


def test_to_np():
    assert isinstance(_to_np(B.ones(tf.float32, 1)), B.NPNumeric)
    assert _to_np("1") is "1"
    assert isinstance(_to_np([B.ones(tf.float32, 1)])[0], B.NPNumeric)
    assert isinstance(_to_np((B.ones(tf.float32, 1),))[0], B.NPNumeric)
    assert isinstance(_to_np({"key": B.ones(tf.float32, 1)})["key"], B.NPNumeric)


def test_safe_dtype():
    assert _safe_dtype(1) == int
    assert _safe_dtype("a") == np.bool  # Should return a small data type.
    assert _safe_dtype(1, 1.0) == np.float64
    assert _safe_dtype(1, np.float32(1.0)) == np.float64
    assert _safe_dtype(np.int16(1), np.float32(1.0)) == np.float32
    assert _safe_dtype(()) == np.bool  # Should return a small data type.
    assert _safe_dtype({"key": np.float32(1.0)}) == np.float32


def test_cast():
    class A:
        dtype = tf.float32

    self = A()

    @cast
    def f(self, x, y=None):
        assert isinstance(x, B.TF)
        assert isinstance(y, (type(None), B.TF))
        return x

    assert isinstance(f(self, B.ones(5)), B.NP)
    assert isinstance(f(self, B.ones(5), y=B.ones(5)), B.NP)
    assert isinstance(f(self, B.ones(tf.float32, 5), y=B.ones(5)), B.TF)
    assert isinstance(f(self, B.ones(tf.float64, 5), y=B.ones(5)), B.TF)
    assert isinstance(f(self, B.ones(5), y=B.ones(tf.float32, 5)), B.TF)
    assert isinstance(f(self, B.ones(tf.float32, 5), y=B.ones(tf.float32, 5)), B.TF)


def test_instancemethod():
    class MyModel(Model):
        def __prior__(self):
            pass

        @instancemethod
        def f(self):
            assert self.instantiated

        def g(self):
            assert not self.instantiated

    model = MyModel()
    model.f()  # This only succeeds if `@instancemethod` automatically instantiates.
    model.g()


def test_priormethod_posteriormethod():
    class MyModel(Model):
        def __prior__(self):
            pass

        def __condition__(self):
            pass

        @priormethod
        def f(self):
            return 1

        @posteriormethod
        def f(self):
            return 2

        @priormethod
        def g(self):
            return 3

        @posteriormethod
        def h(self):
            return 4

    m = MyModel()

    assert m.f() == 1
    assert m.g() == 3
    with pytest.raises(RuntimeError):
        m.h()

    m = m.condition()

    assert m.f() == 2
    with pytest.raises(RuntimeError):
        m.g()
    assert m.h() == 4


def test_format_class_of():
    class A:
        pass

    assert format_class_of(A()) == "tests.test_model.test_format_class_of.<locals>.A"


def test_model_default_implementations():
    class MyModel(Model):
        pass

    model = MyModel()

    with pytest.raises(NotImplementedError):
        model.__prior__()
    with pytest.raises(NotImplementedError):
        model.__condition__(None, None)
    with pytest.raises(NotImplementedError):
        model.__noiseless__()
    with pytest.raises(NotImplementedError):
        model.logpdf(None, None)
    with pytest.raises(NotImplementedError):
        model.sample(None)

    class MyModel2(Model):
        def __prior__(self):
            pass

        def sample(self, x):
            return 1

    model = MyModel2()
    assert model.predict(None, num_samples=100) == (1, 0)


def test_model_instantiation():
    class MyModel(Model):
        def __prior__(self, *args, **kw_args):
            pass

    vs = Vars(np.float64)
    ps = Vars(np.float32).struct

    # Test `instantiated`.
    model = MyModel()
    assert not model.instantiated
    assert model().instantiated

    # Pass a variable container.
    model = MyModel()
    with pytest.raises(AttributeError):
        model.vs
    with pytest.raises(AttributeError):
        model.ps
    with pytest.raises(AttributeError):
        model.dtype
    with pytest.raises(AttributeError):
        model(vs).vs
    assert model(vs).ps._vs is vs
    assert model(vs).dtype is vs.dtype

    # Pass a parameter struct.
    model = MyModel()
    # Already did tests for uninstantiated model.
    with pytest.raises(AttributeError):
        model(ps).vs
    assert model(ps).ps is ps
    assert model(ps).dtype is ps._vs.dtype

    # Set a variable container as a default.
    model = MyModel()
    model.vs = vs
    assert model.vs is vs
    with pytest.raises(AttributeError):
        model.ps
    assert model.dtype is vs.dtype
    assert model().vs is vs
    assert model().ps._vs is vs
    assert model().dtype is vs.dtype

    # This cannot happen: model not instantiated, but parameter struct assigned. Test
    # for it anyway.
    model = MyModel()
    model._set_ps(ps)
    with pytest.raises(AttributeError):
        model.ps

    # Test explicitly setting the data type.
    model = MyModel()
    model.dtype = np.float16
    assert model.dtype == np.float16
    assert model(vs).dtype == vs.dtype
    assert model(ps).dtype == ps._vs.dtype
    # Cannot set `dtype` if either `vs` or `ps` is set.
    with pytest.raises(AttributeError):
        model(vs).dtype = np.float16
    model.vs = vs
    with pytest.raises(AttributeError):
        model.dtype = np.float16

    def check_instance(*args, dtype, **kw_args):
        model = MyModel()
        instance = model(*args, **kw_args)

        with pytest.raises(AttributeError):
            instance.vs
        with pytest.raises(AttributeError):
            instance.ps
        assert instance.dtype == dtype

    # Check the base case.
    check_instance(dtype=np.float16)

    # Check that non-numeric arguments can be passed.
    check_instance("a", ("b",), keyword=["c"], dtype=np.float16)

    # Check no framework and promotion to floating type.
    check_instance(B.ones(5), keyword=B.ones(5), dtype=np.float64)
    check_instance(B.ones(int, 5), keyword=B.ones(int, 5), dtype=np.float64)

    # Check framework and promotion to floating type.
    check_instance(B.ones(np.float32, 5), keyword=B.ones(tf.int16, 5), dtype=np.float32)
    check_instance(B.ones(tf.int16, 5), keyword=B.ones(np.int16, 5), dtype=np.float32)

    # Check that frameworks cannot be mixed.
    with pytest.raises(ValueError):
        model = MyModel()
        model(B.ones(tf.float32, 2), keyword=B.ones(jnp.float32, 2))


def test_copy():
    class MyModel(Model):
        def __init__(self):
            self.x = 1

        def __prior__(self):
            pass

        def __noiseless__(self):
            pass

        def __condition__(self):
            pass

    model = MyModel()

    # Check that instances are copies.
    instance = model()
    instance.x = 2
    assert model.x == 1

    # Check that a noiseless version is a copy.
    noiseless = model.noiseless
    noiseless.x = 2
    assert model.x == 1

    # Check that a posterior is a copy.
    posterior = model.condition()
    posterior.x = 2
    assert model.x == 1


def test_prior_posterior():
    class MyModel(Model):
        def __init__(self):
            self.x = 1

        def __prior__(self):
            pass

        def __condition__(self):
            pass

    model = MyModel()
    assert model.prior
    assert not model.posterior

    instance = model()
    assert instance.prior
    assert not instance.posterior

    instance = instance.condition()
    assert not instance.prior
    assert instance.posterior


def test_num_outputs():
    class MyModel(Model):
        pass

    model = MyModel()

    with pytest.raises(AttributeError):
        model.num_outputs
    model.num_outputs = 5
    assert model.num_outputs == 5


def test_additional():
    # Check mutating operations on model.
    class MyModel(Model):
        def __prior__(self):
            self.noise = 10
            self.conditioned = False

        def __noiseless__(self):
            self.noise = 0

        def __condition__(self):
            self.conditioned = True

    model = MyModel()
    instance = model()

    assert instance.noise == 10
    assert not instance.conditioned

    res = instance.noiseless
    assert res is instance

    assert instance.noise == 0
    assert not instance.conditioned

    res = instance.condition()
    assert res is instance

    assert instance.noise == 0
    assert instance.conditioned


def test_transformed():
    class MyModel(Model):
        pass

    # Check that we can feed a data type or a variable container.
    vs = Vars(np.float32)
    assert isinstance(Transformed(vs.dtype, MyModel()).vs, Vars)
    assert isinstance(Transformed(vs, MyModel()).vs, Vars)

    # Check the attribute `num_outputs`.
    model = Transformed(np.float32, MyModel())
    with pytest.raises(AttributeError):
        model.num_outputs
    model.model.num_outputs = 5
    assert model.num_outputs == 5


class GPModel(Model):
    def __init__(self, init_variance, init_length_scale, init_noise):
        self.init_variance = init_variance
        self.init_length_scale = init_length_scale
        self.init_noise = init_noise

    def __prior__(self):
        """Construct the prior of the model."""
        variance = self.ps.variance.positive(self.init_variance)
        length_scale = self.ps.length_scale.positive(self.init_length_scale)
        self.f = GP(variance * EQ().stretch(length_scale))
        self.noise = self.ps.noise.positive(self.init_noise)

    def __noiseless__(self):
        """Transform the model into a noiseless one."""
        self.noise = 0

    @cast
    def __condition__(self, x, y):
        """Condition the model on data."""
        self.f = self.f | (self.f(x, self.noise), y)

    @instancemethod
    @cast
    def logpdf(self, x, y):
        """Compute the log-pdf."""
        return self.f(x).logpdf(y)

    @instancemethod
    @cast
    def predict(self, x):
        """Make predictions at new input locations."""
        return self.f(x, self.noise).marginals()

    @instancemethod
    @cast
    def sample(self, x):
        """Sample at new input locations."""
        return self.f(x, self.noise).sample()


@pytest.mark.parametrize("learn_transform", [True, False])
def test_transformed_gp_model(learn_transform):
    check_model(
        Transformed(
            tf.float64,
            GPModel(1, 1, 1e-2),
            transform="normalise",
            learn_transform=learn_transform,
        ),
        tf.float64,
    )

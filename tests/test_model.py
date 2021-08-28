# noinspection PyUnresolvedReferences
import lab.tensorflow as B
import numpy as np
import pytest
import tensorflow as tf
from probmods import Model, instancemethod, priormethod, posteriormethod, convert
from probmods.test import check_model
from stheno import EQ, GP
from varz import Vars


class GPModel(Model):
    def __init__(self, init_variance, init_length_scale, init_noise):
        self.variance = init_variance
        self.length_scale = init_length_scale
        self.noise = init_noise

    def __prior__(self):
        """Construct the prior of the model."""
        variance = self.ps.variance.positive(self.variance)
        length_scale = self.ps.length_scale.positive(self.length_scale)
        self.f = GP(variance * EQ().stretch(length_scale))
        self.noise = self.ps.noise.positive(self.noise)

    def __noiseless__(self):
        """Transform the model into a noiseless one."""
        self.noise = 0

    @convert
    def __condition__(self, x, y):
        """Condition the model on data."""
        self.f = self.f | (self.f(x, self.noise), y)

    @instancemethod
    @convert
    def logpdf(self, x, y):
        """Compute the log-pdf."""
        return self.f(x).logpdf(y)

    @instancemethod
    @convert
    def predict(self, x):
        """Make predictions at new input locations."""
        return self.f(x, self.noise).marginals()

    @instancemethod
    @convert
    def sample(self, x):
        """Sample at new input locations."""
        return self.f(x, self.noise).sample()


def test_gp_model():
    check_model(GPModel(1, 1, 1e-2), tf.float64)


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

    with pytest.raises(AttributeError):
        m.f()
    with pytest.raises(AttributeError):
        m.g()
    with pytest.raises(AttributeError):
        m.h()

    m.vs = Vars(np.float64)

    assert m.f() == 1
    assert m.g() == 3
    with pytest.raises(RuntimeError):
        m.h()

    m = m.condition()

    assert m.f() == 2
    with pytest.raises(RuntimeError):
        m.g()
    assert m.h() == 4

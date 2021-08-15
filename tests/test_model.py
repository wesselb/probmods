# noinspection PyUnresolvedReferences
import lab.tensorflow as B
import tensorflow as tf
from stheno import EQ, GP

from probmods import Model, instancemethod, convert
from probmods.test import check_model


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

from lab import B
from numpy.testing import assert_allclose as approx
from plum import Dispatcher
from probmods import Model
from varz import Vars

__all__ = ["check_model"]

_dispatch = Dispatcher()


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


def check_model(
    model,
    dtype=None,
    epsilon=1e-12,
    atol=1e-4,
    rtol=1e-4,
    test_sampling_random=True,
    test_sampling_noiseless_posterior=True,
    test_predict_noiseless_posterior=True,
    test_predict_noisy_posterior=True,
    test_logpdf_computes=True,
    test_fit=True,
):
    """Perform some basic assertions on a model.

    Args:
        model (:class:`.model.Model`): Model to test.
        dtype (dtype, optional): Data type of variables. to instantiate the model with.
        epsilon (float, optional): Amount of extra regularisation. Defaults to `1e-12`.
        atol (float, optional): Absolute tolerance on recovery tests. Defaults to
            `1e-4`.
        rtol (float, optional): Relative tolerance on recovery tests. Defaults to
            `1e-4`.
        test_sampling_random (bool, optional): Test that sampling is random. Defaults
            to `True`.
        test_sampling_noiseless_posterior (bool, optional): Test that sampling from a
            noiseless model conditioned on data recovers the data. Defaults to `True`.
        test_predict_noiseless_posterior (bool, optional): Test that predicting with a
            noiseless model conditioned on data recovers the data. Defaults to `True`.
        test_predict_noisy_posterior (bool, optional): Test that predicting with a noisy
            model conditioned on data gives random samples. Default to `True`.
        test_logpdf_computes (bool, optional): Test that the log-pdf computes, also
            after conditioning. Defaults to `True`.
        test_fit (bool, optional): Test that `model.fit(x, y)` runs. Defaults to `True`.
    """
    with Regularisation(epsilon):
        if dtype is not None:
            model.vs = Vars(dtype)

        # Generate data by sampling from the prior.
        x = B.linspace(0, 5, 5)
        # Sample from a noiseless version of the model to ensure that it is in
        # the support of the model.
        y = model.noiseless.sample(x)

        # Check that sampling is random.
        if test_sampling_random:
            for m in [
                model,
                model.noiseless,
                model.condition(x, y),
                model.condition(x, y).noiseless,
            ]:
                unequal(m.sample(x), m.sample(x), rtol=1e-2)

        # Check that sampling after conditioning without noise matches the data.
        if test_sampling_noiseless_posterior:
            approx(
                B.squeeze(model.noiseless.condition(x, y).sample(x)),
                B.squeeze(y),
                rtol=rtol,
            )

        # Check that predictions after conditioning without noise matches the data.
        if test_predict_noiseless_posterior:
            mean, var = model.noiseless.condition(x, y).predict(x)
            approx(B.squeeze(mean), B.squeeze(y), rtol=rtol)
            approx(var, 0, atol=atol)

        # Check that predictions after conditioning noise gives predictive variance.
        if test_predict_noisy_posterior:
            mean, var = model.condition(x, y).predict(x)
            assert B.all(var >= 0)  # Variance must be non-negative.
            unequal(var, 0, rtol=1e-2)

        # Check that the log-pdf computes. Note that the log-pdf may be stochastic.
        if test_logpdf_computes:
            assert ~B.isnan(model.logpdf(x, y))
            assert ~B.isnan(model.condition(x, y).logpdf(x, y))

        # Check that the model can be fit.
        if test_fit:
            # Add some noise: the data was noiseless.
            model.fit(x, y + B.randn(*B.shape(y)), iters=10)

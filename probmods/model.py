import abc
import contextlib
from functools import wraps
from types import FunctionType
from typing import Union

from lab import B
from plum import Dispatcher
from varz import Vars

__all__ = ["ProbabilisticModel", "MultiOutputModel"]

_dispatch = Dispatcher()


@_dispatch
def _same_framework(dtype, a: B.Numeric):
    a_dtype = B.dtype(a)
    # Only check for TensorFlow, PyTorch, and JAX.
    return any(
        [
            isinstance(dtype, DType) and isinstance(a_dtype, DType)
            for DType in [B.TFDType, B.TorchDType, B.JAXDType]
        ]
    )


@_dispatch
def _same_framework(dtype, a):
    # Do nothing if `a` is not numeric.
    return False


@_dispatch
def _convert_input(dtype, rank, a: B.Numeric):
    return B.to_active_device(B.uprank(B.cast(dtype, a), rank=rank))


@_dispatch
def _convert_input(dtype, rank, a: Union[bool, int, float]):
    # Do not convert certain builtins.
    return a


@_dispatch
def _convert_input(dtype, rank, a):
    # Do nothing if `a` is not numeric.
    return a


@_dispatch
def _on_gpu_if_possible(dtype: B.NPDType):
    return contextlib.suppress()


@_dispatch
def _on_gpu_if_possible(dtype: B.TorchDType):
    import torch

    if torch.cuda.is_available():
        return B.on_device("cuda")
    else:
        return contextlib.suppress()


@_dispatch
def _on_gpu_if_possible(dtype: B.TFDType):
    import tensorflow as tf

    if len(tf.config.list_physical_devices("gpu")) >= 1:
        return B.on_device("gpu")
    else:
        return contextlib.suppress()


@_dispatch
def _on_gpu_if_possible(dtype: B.JAXDType):
    import jax

    gpus = [d for d in jax.devices() if "gpu" in str(d).lower()]
    if len(gpus) >= 1:
        return B.on_device(gpus[0])
    else:
        return contextlib.suppress()


@_dispatch
def _convert_output_to_np(x: B.Numeric):
    return B.to_numpy(x)


@_dispatch
def _convert_output_to_np(xs: list):
    return [_convert_output_to_np(x) for x in xs]


@_dispatch
def _convert_output_to_np(xs: tuple):
    return tuple(_convert_output_to_np(x) for x in xs)


@_dispatch
def _convert_output_to_np(d: dict):
    return {k: _convert_output_to_np(v) for k, v in d.items()}


@_dispatch
def _convert_output_to_np(x):
    # Default to not converting.
    return x


def create_argument_converter(rank=2, autogpu=True):
    """Create a decorator which automatically converts argument to the right framework
    and the right data type, converts the tensors to at least a given rank, and
    automatically converts back the output to NumPy if none of the arguments were of the
    same right framework. It also automatically runs the method on the GPU if one is
    available.

    Args:
        rank (int, optional): Minimum rank of arguments. Default to `2`.
        autogpu (bool, optional): Automatically run the method on the GPU if one is
            available.

    Returns:
        function: Decorator.
    """

    def decorator(f):
        @wraps(f)
        def f_wrapped(self, *args, **kw_args):
            # Convert the result to NumPy if none of the arguments or keyword arguments
            # are in the same framewor as `dtype`.
            res_to_np = not (
                any(_same_framework(self.dtype, arg) for arg in args)
                or any(_same_framework(self.dtype, v) for v in kw_args.values())
            )

            def convert_and_run():
                # Perform conversion and run wrapped function.
                converted_args = tuple(
                    _convert_input(self.dtype, rank, arg) for arg in args
                )
                convert_kw_args = {
                    k: _convert_input(self.dtype, rank, v) for k, v in kw_args.items()
                }
                return f(self, *converted_args, **convert_kw_args)

            if autogpu:
                # Run on the GPU, if possible.
                with _on_gpu_if_possible(self.dtype):
                    res = convert_and_run()
            else:
                res = convert_and_run()

            if res_to_np:
                return _convert_output_to_np(res)
            else:
                return res

        return f_wrapped

    return decorator


convert_arguments = create_argument_converter(rank=2, autogpu=True)
"""Decorator: Common instance of the argument converter."""


class ProbabilisticModelMeta(abc.ABCMeta):
    """Metaclass for probabilistic models."""

    def __new__(cls, name, bases, dict_):
        instance = abc.ABCMeta.__new__(cls, name, bases, dict_)
        skip_methods = instance.__abstractmethods__
        for k, v in dict_.items():
            is_private_method = k.startswith("_")
            if (
                isinstance(v, FunctionType)
                and k not in skip_methods
                and not is_private_method
            ):
                setattr(instance, k, convert_arguments(v))
        return instance


class ProbabilisticModel(metaclass=ProbabilisticModelMeta):
    """A probabilistic model.

    Args:
        dtype (dtype): Data type.
    """

    def __init__(self, dtype):
        self.dtype = dtype
        self._vs_source = _Source(Vars(dtype))

    @property
    def noiseless(self):
        """:class:`.ProbabilisticModel`: Noiseless version of the model."""
        raise NotImplementedError(
            f"A noiseless version of "
            f"`{self.__class__.__name__}.{self.__class__.__qualname__}` is not "
            f"available."
        )

    @abc.abstractmethod
    def logpdf(self, x, y, noise=None):
        """Compute the logpdf of observations.

        Args:
            x (tensor): Inputs of observations.
            y (tensor): Outputs of observations.
            noise (tensor, optional): Additional noise for the observations.

        Returns:
            scalar: The logpdf.
        """

    @abc.abstractmethod
    def condition(self, x, y, noise=None):
        """Condition the model on observations.

        Args:
            x (tensor): Inputs of observations.
            y (tensor): Outputs of observations.
            noise (tensor, optional): Additional noise for the observations.

        Returns:
            :class:`.ProbabilisticModel`: A posterior version of the model.
        """

    @abc.abstractmethod
    def fit(self, x, y, noise=None, **kw_args):
        """Fit the model.

        Takes in further keyword arguments which will be passed to an optimiser.

        Args:
            x (tensor): Inputs of observations.
            y (tensor): Outputs of observations.
            noise (tensor, optional): Additional noise for the observations.
        """

    @abc.abstractmethod
    def sample(self, x):
        """Sample from the model.

        Args:
            x (tensor): Inputs to sample at.

        Returns:
            tensor: Samples from the model.
        """

    @abc.abstractmethod
    def predict(self, x):
        """Make predictions.

        Args:
            x (tensor): Inputs to predict at.

        Returns:
            object: Predictions.
        """

    @property
    def vs(self):
        """:class:`varz.Vars`: Variable container."""
        return self._vs_source.vs

    @vs.setter
    @_dispatch
    def vs(self, source: Vars):
        self._vs_source.vs = source

    @property
    def params(self):
        """:class:`varz.spec.Struct`: Parameter struct."""
        return self._vs_source.params

    @_dispatch
    def use_vs(self, vs: Vars):
        """Use different variables.

        Args:
            vs (:class:`varz.Vars`): Variable container.

        Returns:
            context: Context in which `self` uses different variables.
        """
        return _SetVS(vs, self)

    @_dispatch
    def bind(self, model: "ProbabilisticModel"):
        self._vs_source = _Reference(model)
        return self

    @_dispatch
    def bind(self, model: "ProbabilisticModel", attr: str):
        self._vs_source = _Reference(model, attr)
        return self

    @_dispatch
    def __setattr__(self, key, value):
        object.__setattr__(self, key, value)

    @_dispatch
    def __setattr__(self, key, model: "ProbabilisticModel"):
        model.bind(self, key)
        object.__setattr__(self, key, model)


class _AbstractSource:
    pass


class _Source(_AbstractSource):
    def __init__(self, vs):
        self._vs = vs

    @property
    def vs(self):
        return self._vs

    @vs.setter
    def vs(self, vs):
        self._vs = vs

    @property
    def params(self):
        return self.vs.struct


class _Reference(_AbstractSource):
    def __init__(self, source, attr=None):
        self._source = source
        self._attr = attr

    @property
    def vs(self):
        return self._source.vs

    @vs.setter
    def vs(self, vs):
        self._source.vs = vs

    @property
    def params(self):
        params = self._source.params
        if self._attr:
            return getattr(params, self._attr)
        else:
            return params


class _SetVS:
    def __init__(self, vs, model):
        self.vs = vs
        self.old_vs = None
        self.model = model

    def __enter__(self):
        self.old_vs = self.model.vs
        self.model.vs = self.vs

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.model.vs = self.old_vs


class MultiOutputModel(ProbabilisticModel):
    """A probabilistic model with multiple outputs."""

    @property
    def num_outputs(self):
        """int: Number of outputs."""
        if hasattr(self, "_num_outputs") and self._num_outputs is not None:
            return self._num_outputs
        else:
            raise RuntimeError("Number of outputs is requested, but not (yet) defined.")

    @num_outputs.setter
    def num_outputs(self, num):
        self._num_outputs = num

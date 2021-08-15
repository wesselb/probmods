import contextlib
import copy
from functools import wraps
from types import FunctionType

from lab import B
from plum import Dispatcher, parametric, Union
from plum.parametric import CovariantMeta
from varz import Vars, minimise_l_bfgs_b
from varz.spec import Struct

from .bijection import parse as parse_transform

__all__ = ["convert", "instancemethod", "Model", "Transformed"]

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
def _same_framework(dtype, tup: tuple):
    return any(_same_framework(dtype, a) for a in tup)


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


def convert(f=None, rank=2, gpu=True):
    """Create a decorator which automatically converts argument to the right framework
    and the right data type, converts the tensors to at least a given rank, and
    automatically converts back the output to NumPy if none of the arguments were of the
    same right framework. It also automatically runs the method on the GPU if one is
    available.

    Args:
        rank (int, optional): Minimum rank of arguments. Default to `2`.
        gpu (bool, optional): Automatically run the method on the GPU if one is
            available.

    Returns:
        function: Decorator.
    """

    def decorator(f):
        @wraps(f)
        def f_wrapped(self, *args, **kw_args):
            # Convert the result to NumPy if none of the arguments or keyword arguments
            # are in the same framework.
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

            if gpu:
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

    if f is None:
        return decorator
    else:
        return decorator(f)


def instancemethod(f):
    """Decorator to indicate that the method is a method of an instance rather than
    a model."""

    @wraps(f)
    def f_wrapped(self, *args, **kw_args):
        if not self.instantiated:
            # Attempt to automatically instantiate.
            self = self(self.vs)
        return f(self, *args, **kw_args)

    return f_wrapped


def format_class_of(x):
    cls = type(x)
    return f"{cls.__module__}.{cls.__qualname__}"


class Model(metaclass=CovariantMeta):
    """A probabilistic model."""

    def __call__(self, ps=None, *args, **kw_args):
        """Instantiate the model.

        Args:
            ps (:class:`varz.Vars` or :class:`varz.Struct`, optional): Parameter to
                instantiate the model with. If no arguments are given, this will default
                to `self.vs`.
            *args (object): Arguments to be passed to `__prior__`.
            **kw_args (object): Keyword arguments to be passed to `__prior__`.
        """
        instance = copy.copy(self)
        instance.instantiated = True
        instance.ps = ps if ps is not None else self.vs.struct
        instance.__prior__(*args, **kw_args)
        return instance.instantiator(instance)

    @property
    def instantiated(self):
        """bool: Boolean indicating whether the model is instantiated or not."""
        return hasattr(self, "_instantiated") and self._instantiated

    @instantiated.setter
    @_dispatch
    def instantiated(self, value: bool):
        self._instantiated = value

    @property
    def instantiator(self):
        if hasattr(self, "_instantiator"):
            return self._instantiator
        else:
            return lambda x: x

    @instantiator.setter
    @_dispatch
    def instantiator(self, instantiator: FunctionType):
        self._instantiator = instantiator

    @property
    def ps(self):
        """:class:`varz.spec.Struct`: Parameter struct."""
        if not hasattr(self, "_ps") or self._ps is None:
            raise AttributeError("Parameter struct not available.")
        if not self.instantiated:
            raise AttributeError(
                "Parameter struct is available, but the model is not yet instantiated."
            )
        return self._ps

    @ps.setter
    @_dispatch
    def ps(self, vs: Vars):
        self._ps = vs.struct

    @ps.setter
    @_dispatch
    def ps(self, ps: Struct):
        self._ps = ps

    @property
    def vs(self):
        """:class:`varz.Vars`: Variable container."""
        if not hasattr(self, "_vs") or self._vs is None:
            raise AttributeError("Variable container not available.")
        return self._vs

    @vs.setter
    @_dispatch
    def vs(self, vs: Vars):
        self._vs = vs

    @property
    def dtype(self):
        """dtype: If the model is instantiated, data type of the model parameters. If
        the model is not instantiated, data type of the attached variable container."""
        if self.instantiated:
            return self.ps._vs.dtype
        else:
            return self.vs.dtype

    @property
    def num_outputs(self):
        """int: Number of outputs."""
        if not hasattr(self, "_num_outputs") or self._num_outputs is None:
            raise RuntimeError("Number of outputs is requested, but not (yet) defined.")
        return self._num_outputs

    @num_outputs.setter
    def num_outputs(self, num):
        self._num_outputs = num

    def __prior__(self):
        """Construct the prior."""
        raise NotImplementedError(
            f'The prior of of "{format_class_of(self)}" is not implemented.'
        )

    def __condition__(self, x, y):
        """Condition the model on observations.

        Args:
            x (tensor): Inputs of observations.
            y (tensor): Outputs of observations.
        """
        raise NotImplementedError(
            f'Conditioning for "{format_class_of(self)}" is not implemented.'
        )

    def __noiseless__(self):
        """Remove noise from the model."""
        raise NotImplementedError(
            f'A noiseless version of "{format_class_of(self)}" is not implemented.'
        )

    @property
    def noiseless(self):
        """:class:`.ProbabilisticModel`: Noiseless version of the model."""
        instance = copy.copy(self)

        def instantiator(*args, **kw_args):
            model = self.instantiator(*args, **kw_args)
            model.__noiseless__()
            return model

        instance.instantiator = instantiator

        return instance

    def logpdf(self, x, y):
        """Compute the logpdf of observations.

        Args:
            x (tensor): Inputs of observations.
            y (tensor): Outputs of observations.

        Returns:
            scalar: The logpdf.
        """
        raise NotImplementedError(
            f'The log-pdf for "{format_class_of(self)}" is not implemented.'
        )

    def condition(self, x, y):
        """Condition the model on observations.

        Args:
            x (tensor): Inputs of observations.
            y (tensor): Outputs of observations.

        Returns:
            :class:`.Model`: A posterior version of the model.
        """
        instance = copy.copy(self)

        def instantiator(*args, **kw_args):
            model = self.instantiator(*args, **kw_args)
            model.__condition__(x, y)
            return model

        instance.instantiator = instantiator

        return instance

    def sample(self, x):
        """Sample from the model.

        Args:
            x (tensor): Inputs to sample at.

        Returns:
            tensor: Samples from the model.
        """
        raise NotImplementedError(
            f'Sampling from "{format_class_of(self)}" is not implemented.'
        )

    @instancemethod
    def predict(self, x, num_samples=100):
        """Make predictions.

        Args:
            x (tensor): Inputs to predict at.
            num_samples (int, optional): Number of samples. Defaults to 100.

        Returns:
            object: Predictions.
        """
        samples = B.stack(*[self.sample(x) for _ in range(num_samples)], axis=0)
        return B.mean(samples), B.std(samples) ** 2

    def fit(self, *args, **kw_args):
        """Fit the model. See :func:`.model.fit`."""
        fit(self, *args, **kw_args)

    @_dispatch
    def _(self):
        pass  # This method is necessary for dispatch to work.


@_dispatch
def _as_vars(x: Union[Vars, Struct]):
    return x


@_dispatch
def _as_vars(dtype: B.DType):
    return Vars(dtype)


@parametric
class Transformed(Model):
    """Transform the outputs of a model.

    Args:
        dtype (dtype): Initialise a variable container with this data type. You
            can also pass a variable container.
        model (model): Model to transform the outputs of.
        data_transform (object): Transform. See :func:`.bijector.parse`.
    """

    def __init__(self, dtype, model, data_transform="normalise"):
        self.vs = _as_vars(dtype)
        self.model = model
        self.data_transform = parse_transform(data_transform)

    @classmethod
    def __infer_type_parameter__(cls, dtype, model, *args, **kw_args):
        return type(model)

    def __prior__(self):
        self.model = self.model(self.ps)

    @convert
    def __condition__(self, x, y):
        self.model.__condition__(x, self.data_transform(y))

    def __noiseless__(self):
        self.model.__noiseless__()

    @instancemethod
    @convert
    def logpdf(self, x, y):
        y_transformed = self.data_transform(y)
        return self.model.logpdf(x, y_transformed) + self.data_transform.logdet(y)

    @instancemethod
    @convert
    def sample(self, x):
        return self.data_transform.untransform(self.model.sample(x))

    @instancemethod
    @convert
    def predict(self, x):
        return self.data_transform.untransform(self.model.predict(x))

    @property
    def num_outputs(self):
        return self.model.num_outputs

    def __getattr__(self, item):
        if item.startswith("_"):
            raise AttributeError(f"Attribute `{item}` not found.")
        else:
            return getattr(self.model, item)


@_dispatch
@convert
def fit(model, x, y, minimiser=minimise_l_bfgs_b, trace=True, **kw_args):
    """Fit the model.

    Takes in further keyword arguments which will be passed to the minimiser.

    Args:
        x (tensor): Inputs of observations.
        y (tensor): Outputs of observations.
        minimiser (function): Minimiser. Defaults to `varz.minimise_l_bfgs_b`.
    """

    def normalised_negative_log_marginal_likelihood(vs):
        n = B.sum(B.cast(B.dtype(y), ~B.isnan(y)))
        return -model(vs).logpdf(x, y) / n

    minimiser(
        normalised_negative_log_marginal_likelihood,
        model.vs,
        trace=trace,
        **kw_args,
    )

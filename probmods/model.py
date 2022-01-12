import copy
import warnings
from functools import wraps, partial
from types import FunctionType

import numpy as np
from lab import B
from plum import Dispatcher, parametric, Union, convert
from plum.parametric import CovariantMeta
from varz import Vars, minimise_l_bfgs_b
from varz.spec import Struct

from .bijection import parse as parse_transform

__all__ = [
    "cast",
    "instancemethod",
    "priormethod",
    "posteriormethod",
    "Model",
    "Transformed",
    "fit",
]

_dispatch = Dispatcher()


_fws_numeric = [B.TFNumeric, B.TorchNumeric, B.JAXNumeric]
"""list[type]: Numerical types of frameworks to check for."""
_fws_dtypes = [B.TFDType, B.TorchDType, B.JAXDType]
"""list[dtype]: Data types corresponding to the `_fws_numeric`."""


@_dispatch
def _same_framework(dtype, a: B.Numeric):
    a_dtype = B.dtype(a)
    # Only check for TensorFlow, PyTorch, and JAX.
    return any(
        [isinstance(dtype, fw) and isinstance(a_dtype, fw) for fw in _fws_dtypes]
    )


@_dispatch
def _same_framework(dtype, tup: tuple):
    return any(_same_framework(dtype, a) for a in tup)


@_dispatch
def _same_framework(dtype, a):
    # Do nothing if `a` is not numeric.
    return False


@_dispatch
def _cast(dtype, a: B.Numeric):
    # Only convert floating-point tensors.
    if not B.issubdtype(B.dtype(a), np.floating):
        return a
    # Only convert arrays.
    if B.is_scalar(a):
        return a
    return B.to_active_device(B.cast(dtype, a))


@_dispatch
def _cast(dtype, a):
    # Do nothing if `a` is not numeric.
    return a


@_dispatch
def _to_np(x: B.Numeric):
    return B.to_numpy(x)


@_dispatch
def _to_np(xs: list):
    return [_to_np(x) for x in xs]


@_dispatch
def _to_np(xs: tuple):
    return tuple(_to_np(x) for x in xs)


@_dispatch
def _to_np(d: dict):
    return {k: _to_np(v) for k, v in d.items()}


@_dispatch
def _to_np(x):
    # Default to not converting.
    return x


@_dispatch
def _safe_dtype(x):
    try:
        return B.dtype(x)
    except AttributeError:
        # Return a very small data type.
        return np.bool


@_dispatch
def _safe_dtype(*xs):
    return _safe_dtype(xs)


@_dispatch
def _safe_dtype(xs: tuple):
    if len(xs) == 0:
        # Return a very small data type.
        return np.bool
    else:
        return B.promote_dtypes(*(_safe_dtype(x) for x in xs))


@_dispatch
def _safe_dtype(d: dict):
    return _safe_dtype(tuple(d.values()))


def cast(f):
    """Decorator which automatically converts argument to the right framework
    and the right data type and automatically converts back the output to NumPy if none
    of the arguments were of the same right framework.
    """

    @wraps(f)
    def f_wrapped(self, *args, **kw_args):
        # Convert the result to NumPy if none of the arguments or keyword arguments
        # are in the same framework.
        res_to_np = not (
            any(_same_framework(self.dtype, arg) for arg in args)
            or any(_same_framework(self.dtype, v) for v in kw_args.values())
        )

        # Perform conversion and run wrapped function.
        args = tuple(_cast(self.dtype, arg) for arg in args)
        kw_args = {k: _cast(self.dtype, v) for k, v in kw_args.items()}
        res = f(self, *args, **kw_args)

        if res_to_np:
            return _to_np(res)
        else:
            return res

    return f_wrapped


def instancemethod(f):
    """Decorator to indicate that the method is a method of an instance rather than
    a model."""

    @wraps(f)
    def f_wrapped(self, *args, **kw_args):
        if not self.instantiated:
            # Attempt to automatically instantiate.
            self = self()
        return f(self, *args, **kw_args)

    return f_wrapped


class _PendingFunction:
    prior_pending = []
    posterior_pending = []

    def __init__(self):
        self.name = None

    def __set_name__(self, owner, name):
        self.name = name

        if not hasattr(owner, "_prior_methods"):
            owner._prior_methods = {}
        for f in _PendingFunction.prior_pending:
            owner._prior_methods[f.__name__] = f
        _PendingFunction.prior_pending.clear()

        if not hasattr(owner, "_posterior_methods"):
            owner._posterior_methods = {}
        for f in _PendingFunction.posterior_pending:
            owner._posterior_methods[f.__name__] = f
        _PendingFunction.posterior_pending.clear()

    def __get__(self, instance, owner):
        # Prior and posterior methods are instance methods.
        if not instance.instantiated:
            # Attempt to automatically instantiate.
            instance = instance()

        # Find the right method.
        if instance.posterior:
            if self.name not in owner._posterior_methods:
                raise RuntimeError(f"There is no posterior method for `{self.name}`.")
            else:
                return partial(owner._posterior_methods[self.name], instance)
        else:
            if self.name not in owner._prior_methods:
                raise RuntimeError(f"There is no prior method for `{self.name}`.")
            else:
                return partial(owner._prior_methods[self.name], instance)


def priormethod(f):
    """Decorator to indicate that the method is a method of an instance which is still
    a prior."""
    _PendingFunction.prior_pending.append(f)
    return _PendingFunction()


def posteriormethod(f):
    """Decorator to indicate that the method is a method of an instance which is
    conditioned on data."""
    _PendingFunction.posterior_pending.append(f)
    return _PendingFunction()


def format_class_of(x):
    cls = type(x)
    return f"{cls.__module__}.{cls.__qualname__}"


class Model(metaclass=CovariantMeta):
    """A probabilistic model."""

    def __call__(self, *args, **kw_args):
        """Instantiate the model.

        The first argument can be a parameter struct of type :class:`varz.Vars` or
        :class:`varz.Struct`, which will populate `self.ps`. If no such first argument
        is given, the parameter struct will be attempt to be extracted from `self.vs`.
        If `self.vs` also is not populated, the right data type will be attempted to
        be extracted from the arguments of the call.

        Args:
            *args (object): Arguments to be passed to `__prior__`.
            **kw_args (object): Keyword arguments to be passed to `__prior__`.
        """
        instance = copy.copy(self)
        instance.instantiated = True
        if len(args) >= 1 and isinstance(args[0], (Vars, Struct)):
            instance._set_ps(args[0])
            args = args[1:]
        else:
            try:
                instance._set_ps(instance.vs)
            except AttributeError:
                # Try to determine data type from arguments.
                fw_hits = [
                    any([isinstance(x, t) for x in args + tuple(kw_args.values())])
                    for t in _fws_numeric
                ]
                # Determine the data type and promote to floats.
                dtype = B.promote_dtypes(_safe_dtype(args, kw_args), np.float16)
                if sum(fw_hits) > 1:
                    raise ValueError(
                        "Instantiated the model with a mixture of frameworks. Cannot "
                        "determine data type from arguments."
                    )
                elif sum(fw_hits) == 1:
                    print(fw_hits)
                    instance.dtype = convert(dtype, _fws_dtypes[fw_hits.index(True)])
                else:
                    # No framework hits. Just use NumPy.
                    instance.dtype = convert(dtype, B.NPDType)
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
    def prior(self):
        """bool: Boolean indicated whether the model is conditioned or not."""
        return not self.posterior

    @property
    def posterior(self):
        """bool: Boolean indicated whether the model is conditioned or not."""
        return hasattr(self, "_posterior") and self._posterior

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

    @_dispatch
    def _set_ps(self, vs: Vars):
        self._ps = vs.struct

    @_dispatch
    def _set_ps(self, ps: Struct):
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
        try:
            if self.instantiated:
                return self.ps._vs.dtype
            else:
                return self.vs.dtype
        except AttributeError as e:
            if hasattr(self, "_dtype"):
                return self._dtype
            else:
                raise e

    @dtype.setter
    def dtype(self, dtype):
        if hasattr(self, "_ps") or hasattr(self, "_vs"):
            raise AttributeError(
                "Cannot set data type: attribute `vs` or `ps` exists, which determines "
                "the data type."
            )
        self._dtype = dtype

    @property
    def num_outputs(self):
        """int: Number of outputs."""
        if not hasattr(self, "_num_outputs") or self._num_outputs is None:
            raise AttributeError(
                "Number of outputs is requested, but not (yet) defined."
            )
        return self._num_outputs

    @num_outputs.setter
    def num_outputs(self, num):
        self._num_outputs = num

    def __prior__(self):
        """Construct the prior."""
        raise NotImplementedError(
            f'The prior of "{format_class_of(self)}" is not implemented.'
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
        if self.instantiated:
            self.__noiseless__()
            return self
        else:

            def instantiator(*args, **kw_args):
                model = self.instantiator(*args, **kw_args)
                model.__noiseless__()
                return model

            instance = copy.copy(self)
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

    def condition(self, *condition_args, **condition_kw_args):
        """Condition the model on observations.

        Args:
            x (tensor): Inputs of observations.
            y (tensor): Outputs of observations.

        Returns:
            :class:`.Model`: A posterior version of the model.
        """
        if self.instantiated:
            self.__condition__(*condition_args, **condition_kw_args)
            self._posterior = True
            return self
        else:

            def instantiator(*args, **kw_args):
                model = self.instantiator(*args, **kw_args)
                model.__condition__(*condition_args, **condition_kw_args)
                model._posterior = True
                return model

            instance = copy.copy(self)
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
    def _(self):  # pragma: no cover
        pass  # This method is necessary for dispatch to work.


@_dispatch
def _as_vars(x: Vars):
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
        transform (object, optional): Transform. See :func:`.bijector.parse`.
            Defaults to normalising the data.
        learn_transform (bool, optional): Learn parameters in the transform. Defaults
            to `False`.
    """

    def __init__(
        self,
        dtype,
        model,
        transform="normalise",
        learn_transform=False,
    ):
        self.vs = _as_vars(dtype)
        self.model = model
        self.transform = parse_transform(transform)
        self.learn_transform = learn_transform

    @classmethod
    def __infer_type_parameter__(cls, dtype, model, *args, **kw_args):
        return type(model)

    def __prior__(self):
        self.model = self.model(self.ps)

    @property
    def ps_transform(self):
        if self.learn_transform:
            return self.ps.transform
        else:
            return None

    @cast
    def __condition__(self, x, y):
        self.model.__condition__(x, self.transform(self.ps_transform, y))

    def __noiseless__(self):
        self.model.__noiseless__()

    @instancemethod
    @cast
    def logpdf(self, x, y):
        y_transformed = self.transform(self.ps_transform, y)
        logpdf = self.model.logpdf(x, y_transformed)
        logdet = self.transform.logdet(self.ps_transform, y)
        return logpdf + logdet

    @instancemethod
    @cast
    def sample(self, x):
        return self.transform.untransform(self.ps_transform, self.model.sample(x))

    @instancemethod
    @cast
    def predict(self, x):
        return self.transform.untransform(self.ps_transform, self.model.predict(x))

    @property
    def num_outputs(self):
        return self.model.num_outputs


@_dispatch
@cast
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

# [ProbMods: Probabilistic Models](http://github.com/wesselb/probmods)

[![CI](https://github.com/wesselb/probmods/workflows/CI/badge.svg)](https://github.com/wesselb/probmods/actions?query=workflow%3ACI)
[![Coverage Status](https://coveralls.io/repos/github/wesselb/probmods/badge.svg?branch=main)](https://coveralls.io/github/wesselb/probmods?branch=main)
[![Latest Docs](https://img.shields.io/badge/docs-latest-blue.svg)](https://wesselb.github.io/probmods)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

An interface to facilitate rapid development and deployment of probabilistic models

Contents:

- [Installation](#installation)
- [Manual](#manual)
    - [What is the Problem?](#what-is-the-problem)
    - [Basic Principles](#basic-principles)
        - [Models and Instances](#models-and-instances)
        - [Convenience: `@convert`](#convenience-convert)
        - [Details of Model Instantiation](#details-of-model-instantiation)
        - [Automatic Model Instantiation: `@instancemethod`](#automatic-model-instantiation-instancemethod)
        - [Description of Models](#description-of-models)
    - [Model Fitting](#model-fitting)
    - [`Transformed`](#transformed)
    - [Automatic Model Tests](#automatic-model-tests)

## Installation

See [the instructions here](https://gist.github.com/wesselb/4b44bf87f3789425f96e26c4308d0adc).
Then simply

```
pip install probmods
```

## Manual

### What is the Problem?

Suppose that we have implemented a probabilistic model, and we would like to
apply it to data.
A typical way to do this would be to write a script roughly along the following
lines:

```python
import lab.tensorflow as B
import tensorflow as tf
from stheno.tensorflow import EQ, GP
from varz import Vars, minimise_l_bfgs_b

# Increase regularisation for `float32`s.
B.epsilon = 1e-6

# Initial model parameters:
init_noise = 1e-2
init_variance = 1
init_length_scale = 1


def prior(vs):
    """Construct the prior of the model."""
    ps = vs.struct  # Dynamic parameter struct
    variance = ps.variance.positive(init_variance)
    length_scale = ps.length_scale.positive(init_length_scale)
    noise = ps.noise.positive(init_noise)
    return GP(variance * EQ().stretch(length_scale)), noise


def sample(vs, x):
    """Sample from the prior."""
    f, noise = prior(vs)
    return f(x, noise).sample()


# Create a variable container.
vs = Vars(tf.float32)

# Generate data by sampling from the prior.
x = B.linspace(tf.float32, 0, 10, 100)
y = sample(vs, x)


def objective(vs):
    f, noise = prior(vs)
    return -f(x, noise).logpdf(y)


# Fit the model.
minimise_l_bfgs_b(objective, vs, trace=True, jit=True)


def posterior(vs):
    """Construct the posterior."""
    f, noise = prior(vs)
    post = f | (f(x, noise), y)
    return post, noise


def predict(vs, x):
    """Make predictions at new input locations."""
    f, noise = posterior(vs)
    return f(x).marginals()


# Make predictions.
mean, var = predict(vs, B.linspace(10, 15, 100))
```

In the example, we sample data from a Gaussian process using
[Stheno](https://github.com/wesselb/stheno), learn hyperparameters for the
Gaussian process, and finally make predictions for new input locations.
Several aspects are not completely satisfactory:

* There is not one model object which you can conveniently pass around.
  Instead, the initial model parameters are defined as global variables, and
  the model is implemented with many functions (`prior`, `sample`,
  `objective`, `posterior`, `predict`, ...) which all depend on
  each other.
  This is not a convenient interface.
  If you wanted to share your model with someone else, so they could apply it
  to their data, they would not be able to just insert your model in existing
  code, but they would have to take your whole script.
  Moreover, they even might have to adjust it  appropriately, for what if they
  wanted to sample from the posterior or fit the posterior to some data?
    
* Since the script uses TensorFlow as a backend, you have to be careful
  to convert everything to TensorFlow tensors of the appropriate data type.
    
* The script is not easily extensible.
  What if you wanted to add in data normalisation by subtracting the mean
  and dividing by the standard deviation?
  Among other things, you would have to keep track of those parameters
  and appropriately transform the predictions back to the original domain.
  What if, in addition, your data was positive, so you would want to also
  employ a log-transform?
  The code now starts to become spaghetti-like.
    
The package `probmods` aims to solve all of these problems.
Below is how the above model could be implemented with `probmods`:

```python
import lab.tensorflow as B
import numpy as np
import tensorflow as tf
from stheno import EQ, GP

from probmods import Model, instancemethod, convert, Transformed

# Increase regularisation for `float32`s.
B.epsilon = 1e-6


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


# This model object can easily be inserted anywhere in existing code and be
# passed around
model = Transformed(
    tf.float32,
    GPModel(1, 1, 1e-1),
    data_transform="normalise+positive"
)

# Generate data by sampling from the prior.
x = np.linspace(0, 10, 100)
y = model.sample(x)

# Fit the model and print the learned parameters.
model.fit(x, y, trace=True, jit=True)
model.vs.print()

# Make predictions.
x_new = np.linspace(10, 15, 100)
posterior = model.condition(x, y)
mean, var = posterior.predict(x_new)

# But we can go on....

# ...to sample from the posterior.
y_new = posterior.sample(x_new)

# Or to train the model parameters by _fitting the posterior_ to new data!
posterior.fit(x_new, y_new, trace=True, jit=True)

# Or to condition on even more data!
posterior = posterior.condition(x_new, y_new)

# Or to make more preditions!
mean, var = posterior.predict(x_new)
```

### Basic Principles

#### Models and Instances
_Models_ are functions from _learnable parameters_ to _instances of models_.
An _instance of a model_, or simply an _instance_, is an object with concrete
values for it's all parameters and which can do things like sample or compute
a log-pdf.
Moreover, models can be parametrised by _non-learnabe parameters_, like 
initial values for learnable parameters or parameters which define the structure
of the model, like other submodels.
In this sense, models can interpreted as _configurations_
or _templates_ for instances of models.

For example, consider

```python
>>> model = GPModel(init_variance=1, init_length_scale=1, init_noise=1e-2)
```

Here `model` is a model parametrised with initial values for learnable
parameters.
We can try to sample from it, but this runs into an error, because, although the
initial values for the parameters of `model` are set, the actual parameters of
`model` and the internals, such as the Gaussian process, are not yet created:

```python
>>> x = np.linspace(0, 10, 10)

>>> model.sample(x)
AttributeError: Variable container not available.
```

**Important assumption:**
It is assumed that models like `model` can safely be copied using `copy.copy`,
which  performs a _shallow copy_.
If a shallow copy is not appropriate, you should implement `model.__copy__`.

The object `model` acts like a function from parameters to instances of model.
To demonstrate this, we first need to create parameters.
`probmods` uses [Varz](https://github.com/varz) manage parameters:

```python
>>> from varz import Vars

>>> parameters = Vars(tf.float32)
```

The object `parameters` will create and keep track of all parameters which
`model` will use.
We can feed `parameters` to `model` to get an instance, which we can then
sample from.

```python
>>> instance = model(parameters)

>>> instance.sample(x)
array([[ 0.58702797],
       [ 0.40569574],
       [ 0.42661083],
       [ 1.1435565 ],
       [ 0.02888119],
       [-1.8267081 ],
       [-0.5065604 ],
       [ 0.7860895 ],
       [-0.32402134],
       [-2.4540234 ]], dtype=float32)
```

You can check whether a model is instantiated or not with the property
`instantiated`:

```python
>>> model.instantiated
False

>>> instance.instantiated
True
```

Representing models as functions from parameters to instances has a number of
benefits:

* *Transparancy:* You can always construct an instance simply by calling it 
    with parameters. As can sometimes be the case with objects, you do not
    need to call particular methods in a particular sequence or worry about
    other side effects.

* *Efficiency:* Functions can be compiled using a JIT, which eliminates Python's
    overhead and can create extremely efficient implementations:
  
    ```python
    @jit
    def pplp(parameters, x1, y1, x2, y2):
        """Compute the log-pdf of `(x1, y1)` given `(x2, y2)`."""
        posterior = model.condition(x2, y2)
        return posterior(parameters).logpdf(x1, y1)
    ```

* *Composability:* Models can easily be used as components in bigger models.

#### Convenience: `@convert`

Although the internal variables of `instance` are TensorFlow tensors,
you can simply feed a NumPy array to `instance.sample`.
Furthermore, the output of `instance.sample(x)` is a NumPy array, rather than a 
TensorFlow tensor:

```python
>>> x = np.linspace(0, 10, 10)

>>> instance.sample(x)
array([[0.58702797],
       [0.40569574],
       [0.42661083],
       [1.1435565],
       [0.02888119],
       [-1.8267081],
       [-0.5065604],
       [0.7860895],
       [-0.32402134],
       [-2.4540234]], dtype=float32)
```

This behaviour is due to the `@convert` decorator, which automatically
converts NumPy arguments  to the right framework (in this case, TensorFlow) and
the right data type (in this case, `tf.float32`).
Moreover, if _only_ NumPy arguments were given, `probmods` then also converts
back to the result to NumPy.
For example, if we were to pass a TensorFlow tensor, we would get a TensorFlow
tensor back:

```python
>>> instance.sample(tf.constant(x, dtype=tf.float32))
<tf.Tensor: shape=(10, 1), dtype=float32, numpy=
array([[ 0.37403315],
       [-1.423271  ],
       [-0.60986364],
       [ 0.94153786],
       [ 2.247231  ],
       [ 2.799852  ],
       [ 2.8013554 ],
       [ 1.9703895 ],
       [ 0.6884947 ],
       [-0.47107112]], dtype=float32)>
```

The rules for `@convert` are as follows:

* Every argument which is a NumPy array is automatically converted to the
  appropriate framework and the approprate data type.
  
* Every array argument is ensured to be a tensor of at least rank 2, which is
  achieved by added empty dimensions. You can specify another minimum rank, like
  0 or 1, by using `@convert(rank=0)`.

* Every argument is automatically moved to the GPU, if one is available. You
  can disable this behaviour by using `@convert(gpu=False)`.
    
* If the arguments contained _only_ NumPy arrays, then any framework types,
  like TensorFlow tensors, are automatically converted back to NumPy arrays.

#### Details of Model Instantiation

When `model` is instantiated by calling it with `parameters`, the following
happens:

1. First of all, the model is _copied_ to safely allow mutation of the copy: 

    ```python
    instance = copy.copy(model)
    ```
   
    This copy is a _shallow copy_.
    If a shallow copy is not appropriate, then you should implement
   `instance.__copy__`.
    
2. The parameters are stored in the internal variable `ps`
   (short for parameters):

   ```python
   instance.ps = parameters
   ```

3. The prior is constructed:

    ```python
    instance.__prior__()
    ```

    Calling `instance.__prior__()` mutates `instance`, but that's fine, because
    `instance` is a copy of the original, so no harm done.
    The implementation of `instance.__prior__` can make use of parameters
    through `instance.ps`.
    

4. For every previous `model = model.condition(x, y)` or
   `model = model.noiseless` call, the corresponding operations are performed
   on `instance` in the same order:
   
    ```python
    instance.__condition__(x, y)
   
    instance.__noiseless__()
    ```
   
    The implementations of `instance.__condition__` and `instance.__noiseless__`
    can also make use of parameters through `instance.ps`.
   
5. We're done! The result `instance` is returned. `instance` is populated with
   parameters, has constructed its prior, and has done any potential
   conditioning, so it is ready to be used e.g. for sampling:
   `instance.sample(x)`.


#### Automatic Model Instantiation: `@instancemethod`

From what we've seen so far, you can create a model and sample from it in the
following way:

```python
# Create a model.
model = GPModel(1, 1, 1e-2)

# Sample from it at inputs `x`.
parameters = Vars(tf.float32)
instance = model(parameters)
sample = instance.sample(x)
```

This pattern is slightly suboptimal in two ways:

1. You need to constantly carry around a variable container `parameters`.

2. You need to not forget to instantiate the model (calling `model(parameters)`)
    before doing an operation like sampling.
   
The decorator `@instancemethod` is designed to help with these issues.
If you decorate a method with `@instancemethod`, then that indicates that
that method can only be called on _instances of models_ rather than _models_.
If you call an `@instancemethod` without instantiating `model`, then
the decorator will automatically instantiate the model with the variable
container `model.vs`, assuming that it is available.
That is, if `model.sample` is an `@instancemethod`, then `model.sample(x)`
automatically translates to `model(model.vs).sample(x)`.
`model.vs` does not automatically contain a variable container: 
you will need to assign it one.

```python
# Create a model.
model = GPModel(1, 1, 1e-2)

# Assign a variable container.
model.vs = Vars(tf.float32)

# Sample from the model.
sample = model.sample(x)
```

#### Description of Models

The `Model` class offers the following properties:

| Property             | Description |
| --                   | -- |
| `model.vs`           |  A variable container which will be used to automatically instantiate the model when an `@instancemethod` is called uninstantiated. You need to explicitly assign a variable container to `model.vs`. |
| `model.ps`           | Once the model is instantiated, `model.ps` (or `self.ps` from within the class) can be used to initialise constrained variables. `model.ps` is not available for uninstantiated models. As an example, after instantiation, `self.ps.parameter_group.matrix.orthogonal(shape=(5, 5))` returns a randomly initialised orthogonal matrix of shape `(5, 5)` named `parameter_group.matrix`. `ps` behaves like a nested struct, dictionary, or list. See [Varz](https://github.com/wesselb/varz#structlike-specification) for more details. |
| `model.instantiated` | `True` if `model` is instantiated and `False` otherwise. |
| `model.dtype`        | If the model is instantiated, this return the data type of `model.ps`. If the model is not instantiated, this attempts to returns the data type of `model.vs` |
| `model.num_outputs`  | A convenience property which can be set to the number of outputs of the model. |

When you subclass `Model`, you should implement the following methods:

| Method                           | Description |
| --                               | -- |
| `__prior__(self)`           | Construct the prior of the model. |
| `__condition__(self, x, y)` | The prior was previously constructed. Update the model by conditioning on `(x, y)`. You may want to use `@convert`. |
| `__noiseless__(self)`       | Remove noise from the current model. |
| `logpdf(self, x, y)`        | Compute the logpdf for `(x, y)`. This needs to be an `@instancemethod` and you may want to use `@convert`. |
| `sample(self, x)`           | Sample at inputs `x`. This needs to be an `@instancemethod` and you may want to use `@convert`. |
| `predict(self)`             | Predict at inputs `x`. The default implementation samples and computes the mean and variance of these samples, but you can override this implementation. This needs to be an `@instancemethod` and you may want to use `@convert`. |

For reference, we again show the implementation of `GPModel` here:

```python
from stheno import EQ, GP

from probmods import Model, instancemethod, convert, Transformed


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
```

### `Transformed`

The package offers an implementation of one model: `Transformed`.
`Transformed` takes an existing model and transforms the output of the model,
e.g., to deal with positive data or to normalise data.

Example:

```python
model = Transformed(
    tf.float32,
    GPModel(1, 1, 1e-2),
    data_transform="normalise+positive"
)
```

The first argument `tf.float32` indicates the data type of the parameters
that you would like to use. 
`Transformed` then automatically creates a variable container and assigns
it to `model.vs`.
The second and third arguments are the model to transform and the specification
of the transform.
The following transformations are possible:

| Transformation | Description |
| :- | :- |
| `"normalise"` | Subtract the mean and divide by the standard deviation. The mean to subtract and the standard deviation to divide by are computed from the data to which the transform is first applied; these values are then remembered. |
| `"positive"` | Perform a log-transform. This is handy for positive data. |
| `"squishing"` | Perform a transform which suppresses tails. This is handy for heavy-tailed data. |

You can combine transforms by joining the strings with a `,` or `+`.
For example, `"normalise+positive"` first applies a log-transform and then
normalises the data.
For a more detailed description of, please see
`probmods.bijection.parse`.

### Model Fitting

To fit a model, you can just call `model.fit(x, y)`.
The default implementation simply maximises `model.logpdf(x, y)`.
See `probmods.model.fit` for a description of the arguments to `fit`.

If you want to provide a custom fitting procedure for your model,
then you can implement a method for `fit`:

```python
from probmods import fit

@fit.dispatch
def fit(model: GPModel, x, y):
    ...  # Custom fitting procedure.
```

Note that this will only apply to `model`s which are of the type `GPModel`.
For example, this will not apply to `Transformed(dtype, GPModel(...))`.
To implement a fitting procedure for a transformed version of `GPModel`, the
following is possible:

```python
from probmods import fit, Transformed

@fit.dispatch
def fit(model: Transformed[GPModel], x, y):
    ...  # Custom fitting procedure.
```

### Automatic Model Tests

The function `probmods.test.check_model` can be used to perform some basic
assertions on a model.
See the documentation for more details.


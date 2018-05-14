
# coding: utf-8

# In[1]:


import numpy as np

import matplotlib.pyplot as plt


# ---
# ---
# # Optimization problem

# ## Loss
# 
# - Implement a loss function.

# In[2]:


class LossFunction:
    """Define a loss function to optimize.

    Stored in a `ProblemDefinition`, consumed by an `Optimizer`.
    """
    def __init__(self):
        pass
    
    def __call__(self, sample, pred_gradient):
        raise NotImplementedError()
    
    def gradient(self, sample, pred_gradient):
        raise NotImplementedError()


# In[3]:


class LeastSquares(LossFunction):
    def __call__(self, sample, pred):
        return 0.5 * np.linalg.norm(sample - pred) ** 2
    
    def gradient(self, sample, pred):
        return pred - sample


# ## Problem
# 
# - Define the function to minimize (maybe implicitly by only providing means of
#   computing the gradients)
# - Wrap the gradients computation in a friendly API
# - ...

# In[4]:


from collections import namedtuple
from typing import Sequence

class Problem:
    """Define an optimization problem.

    Automatically generate helpers for gradient computation.
    """
    Sample = namedtuple("Sample", ("inputs", "targets"))

    def __init__(self, loss: LossFunction, samples: Sequence[Sample]):
        self._loss = loss
        self.samples = samples
    
    @property
    def n_samples(self):
        return len(self.samples)


# ---
# ---
# # Estimators

# ## Estimator
# 
# - Attempt to make predictions for a given `Problem.Sample`

# In[5]:


class Estimator:
    """Base-class for estimators."""    
    def predict(self, inputs):
        """Attempt to predict a target given some inputs."""
        raise NotImplementedError()
    
    def update(self, gradients):
        """Update the estimator's parameters.

        To be called from an Optimizer.
        """
        raise NotImplementedError()


# ---
# ---
# # Optimizers

# ## Optimizer
# 
# - Optimize an estimator for a given problem.

# In[8]:


from typing import Iterable, Tuple

from project.utils import LoggerMixin


class Optimizer(LoggerMixin):
    """Base-class for optimizers."""    
    def optimize(self, problem, estimator, n_steps=1):
        """Run the optimizer on the current problem."""
        self._prepare(problem)
        losses = []

        for step, state in self.iterate(n_steps):
            if step % 100 == 1:
                self.logger.info("Step {}/{}".format(step, n_steps))
            losses.append(problem.loss(state))

        self._cleanup()
        return losses
    
    def _prepare(self, problem):
        """Prepare the optimizer state from the problem at hand.

        Override this method in subclasses.
        """
        raise NotImplementedError()
    
    def _cleanup(self):
        """Clean the optimizer state (optional)."""
        pass
    
    def iterate(self, n_steps: int) -> Iterable[Tuple[int, dict]]:
        raise NotImplementedError()


# ## Gradient Descent Optimizer
# 
# - Use the simple update rule $x_{t+1} = x_t - \gamma \, g(x_t)$ where $g$ is a
#   function to override in sub-classes
# - Default implementation uses the true gradient $g(x_t) = \nabla f(x_t)$.

# In[5]:


class GradientDescent(Optimizer):
    """Gradient Descent algorithm."""
    def __init__(self, learning_rate: float, *args, **kwargs):
        super().__init__(self, *args, **kwargs)
        self.learning_rate = learning_rate

    def get_update(self, state):
        return self.problem.full_gradient(state)

    def iterate(self, n_steps):
        for step in range(n_steps):
            # TODO: state - self.learning_rate * self.get_update(state)
            yield step, {}

    def _prepare(self, problem):
        self.problem = problem

    def _cleanup(self):
        del self.problem


# ## Stochastic Gradient Descent
# 
# - Define the stochastic variant of Gradient Descent,
# - Add a batch variant as well

# In[6]:

from project.utils import RandomStateMixin


class SGD(GradientDescent, RandomStateMixin):
    """Stochastic Gradient Descent algorithm."""
    def get_update(self, state):
        index = self.random.randint(self.problem.n_samples)
        return self.problem.partial_gradient(state, index)


# ## SGD with Control Variates
# 
# - Define a base-class for SGD variants using control variates for correcting
#   the partial gradient

# In[7]:


class ControlVariatesOptimizer(SGD):
    """"""


# ## SVRG

# In[ ]:


class SVRG(ControlVariatesOptimizer):
    pass


# ## Hessian control

# In[8]:


class HessianCV(ControlVariatesOptimizer):
    pass


# ---
# ---
# # Monitoring

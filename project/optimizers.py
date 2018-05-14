"""Optimizers."""

from typing import Iterable, Tuple

from project.utils import LoggerMixin


class Optimizer(LoggerMixin):
    """Base-class for optimizers."""    
    def optimize(self, problem, estimator, n_steps=1, eval_every=10):
        """Run the optimizer on the current problem."""
        losses = []

        for step in range(n_steps):
            if step % eval_every == 0:
                self.logger.info("Step {}/{}".format(step, n_steps))
                loss = problem.eval(estimator)

            self.update(estimator, problem)
            losses.append(loss.value)

        return losses

    def update(self, estimator, problem):
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

    def update(self, estimator, problem):
        return "full_gradient()"


# ## Stochastic Gradient Descent
# 
# - Define the stochastic variant of Gradient Descent,
# - Add a batch variant as well

# In[6]:

from project.utils import RandomStateMixin


class SGD(GradientDescent, RandomStateMixin):
    """Stochastic Gradient Descent algorithm."""
    def update(self, estimator, problem):
        index = self.random.randint(problem.n_samples)
        return "partial_gradient({})".format(index)


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
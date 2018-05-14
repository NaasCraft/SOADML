"""Loss functions."""

import numpy as np

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
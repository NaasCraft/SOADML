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

"""Optimization problem, with data preparation tools."""

from collections import namedtuple
from typing import Sequence


class Problem:
    """Define an optimization problem.

    Embeds the `Problem.Sample` named tuple, and provides methods to generate a
    set of samples.

    The problem will be consumed by an `Optimizer` to tune an `Estimator`.
    """
    Sample = namedtuple('Problem.Sample', ('inputs', 'targets'))
    Loss = namedtuple('Problem.Loss', ('value', 'gradient'))

    def __init__(self,
                 loss: 'project.losses.LossFunction',
                 samples: Sequence[Sample]):
        self._loss = loss
        self.samples = samples

    @property
    def n_samples(self):
        return len(self.samples)
    
    def eval(self, estimator: 'project.estimators.Estimator'):
        predictions = estimator.predict(self.samples)
        return Loss(
            value=np.mean([self._loss(p) for p in predictions]),
            gradients=[self._loss.gradient(p) for p in predictions]
        )

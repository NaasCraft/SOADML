# Stochastic Optimization and Automatic Differentiation for Machine Learning (2018)

ENSAE 3rd year Project 2018 - _Teacher_: Marco Cuturi ([website](http://marcocuturi.net/)) - _Student_: Guillaume Demonet

__Subject__:

> **Incremental methods with second order information**  
> Implement and discuss the efficiency of the algorithm proposed in [this paper](https://arxiv.org/abs/1710.07462) by benchmarking it against SVRG (only) on a problem of your choice.

_Links to `references/`_:

- (_main article_) [`HessianTracking.pdf`](./references/HessianTracking.pdf)
- [`SAGA.pdf`](./references/SAGA.pdf)
- [`SVRG.pdf`](./references/SVRG.pdf)
- `...`

---

## Implementation TODO

- [ ] generic context for minimizing a loss function
- [ ] control variates framework : definition, benchmark and evaluation
- [ ] implementation of SVRG in this framework
- [ ] implementation of the Hessian-based versions (3, from the article)
- [ ] optimized algorithms (parallelism, JIT compilation  (+ tensorflow approach ?))
- [ ] multiple control graphs to output (time, space, accuracy, correlation of
  the control variates, etc.)
  
```python
# quick mock-up

class ProblemDefinition:
    """Define an optimization problem.
    
    Automatically generate helpers for gradient computation.
    """
    pass

class Solver:
    """Base-class for solvers"""
```
```{toctree}
:hidden:
:maxdepth: 1

api/overview.md
examples/index.md
claude-skills.md
```


# gpCAM

:::{note}
**New in gpCAM 8.7: `GPOptimizer` handles multi-task GPs.** Give it `y_data` with several
columns, one per output, and choose how the outputs are modeled with `multi_task_mode`:
`'independent'` (the default), `'cartesian'` (full multi-task learning over input × output
space), or `'kronecker'` (a separable kernel on a full grid, for very many outputs).
`fvGPOptimizer` is deprecated: `GPOptimizer(..., multi_task_mode='cartesian')` is its exact
replacement. Kernels, means and noise functions written for `[x, task]` points need
`'cartesian'`. See the [API overview](api/overview.md) and [HISTORY](https://github.com/lbl-camera/gpCAM/blob/master/HISTORY.rst).
:::


```{div} centered-heading
Decision-Making under Uncertainty, HPC Uncertainty Quantification, Autonomous Data Acquisition, and Bayesian Optimization.
```

```{div} text-center

gpCAM is an API and software designed to make new methodologies for Gaussian process modeling, Bayesian optimization, and decision-making under uncertainty faster, more straightforward, and more widely available. The tool is powered by a flexible and powerful Gaussian process regression at the core. The flexibility stems from the modular design of gpCAM, which allows the user to implement and import their own Python functions to customize and control almost every aspect of the software. Due to a synergy between computational and mathematical function definitions that are fundamental to gpCAM, there are virtually no limits to its customizability. That makes it possible to easily tune the algorithm to account for various types of domain knowledge, consider non-standard inputs, such as distributions and arbitrary structures, and scale it to millions of data points. This makes gpCAM the go-to solution for stochastic function approximation in scientific applications.
```

---

```{div} centered-heading
Questions?
```

```{div} text-center

Contact [MarcusNoack@lbl.gov](mailto:MarcusNoack@lbl.gov) to get more information on the project.
```

---

```{div} text-center

gpCAM is a software tool created by CAMERA

The Center for Advanced Mathematics for Energy Research Application
```

```{image} _static/CAMERA_bright.png
:width: 759px
:align: center
```

---

![_static/doe-os.png](_static/doe-os.png)

Supported by the US Department of Energy Office of Science
[Advanced Scientific Computing Research](https://www.energy.gov/science/ascr/advanced-scientific-computing-research)

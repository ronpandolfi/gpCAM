# API Reference

```{toctree}
:hidden:
:maxdepth: 1

gpOptimizer.md
fvgpOptimizer.md
transformedOptimizers.md
gpMCMC.md
kernels.md
logging.md
```

gpCAM's main class is the [gpOptimizer](gpOptimizer.md), for single- and multi-task
Gaussian processes and Bayesian optimization. It is capable of dealing with non-Euclidean
input spaces and world-record-holding scaling of exact GPs. The
[fvgpOptimizer](fvgpOptimizer.md), the multi-task class before 8.7, is deprecated.

These optimizer classes are a thin Bayesian-optimization layer on top of the
[fvGP](https://fvgp.readthedocs.io) engine — most of the GP machinery (kernels,
training, posterior evaluation) is inherited from `fvgp.GP`, and acquisition-function
optimization can run on supercomputers via [HGDL](https://hgdl.readthedocs.io).

To get to know gpCAM, check out the [examples](../examples/index.md), download the
repository and look in `./tests`, or visit the project
[website](https://gpcam.lbl.gov/).

## New in gpCAM 8.7 (fvgp 4.11)

- **One class for single- and multi-task GPs.** The shape of `y_data` decides: (V) or (V, 1)
  is a single-task GP, (V, No) with No > 1 a multi-task GP, modeled as `multi_task_mode`
  says:
  - `'independent'` (the default): No GPs that share one kernel and its hyperparameters;
  - `'cartesian'`: a single-task GP over the product of input and output space, any kernel,
    missing outputs allowed (what `fvGPOptimizer` did);
  - `'kronecker'`: a separable kernel on a full grid, with a learned output covariance by
    default.

  ```python
  from gpcam import GPOptimizer
  gpo = GPOptimizer(x_data, y_data, multi_task_mode="cartesian")   # was fvGPOptimizer(x_data, y_data)
  ```
- **`fvGPOptimizer` is deprecated** (`FutureWarning`) and stays until gpCAM moves to
  fvgp 5.0. It now is a `GPOptimizer` with `multi_task_mode='cartesian'`.
- **The built-in multi-task acquisition functions work in every mode**, and `x_out` selects
  outputs: indices for `'independent'` and `'kronecker'`, output coordinates for
  `'cartesian'`, all outputs by default. `total correlation` is not available for
  `'kronecker'`.
- **`get_data()`** returns the data as the GP stores it, in every mode; for `'cartesian'`
  that is the (V * No, D + 1) product-space form.
- **The posterior results follow one contract** (fvgp 4.11): single-task results have no
  output axis; multi-task `m(x)`, `v(x)` are (V, n), `S` is (V, V, n, n), flat arrays are
  task-major.

## New in fvgp 4.8 (gpCAM 8.4)

Because the optimizers inherit from `fvgp.GP`, the following fvgp 4.8 additions are
available directly on any `GPOptimizer`/`fvGPOptimizer` instance and are listed on the
[gpOptimizer](gpOptimizer.md) / [fvgpOptimizer](fvgpOptimizer.md) pages:

- **Model-validation metrics** — `mae`, `mape`, `msll`, `interval_score`, `mpiw`, and
  `coverage_curve` for quantifying predictive accuracy and calibration, plus
  `plot_observed_vs_predicted` for a quick diagnostic plot.
- **New kernels** — `bump` and `sle_kernel` (in addition to the existing library), all
  re-exported through `gpcam.kernels` and documented on the [Kernels](kernels.md) page.
- **Linear-algebra modes** — the `linalg_mode` argument now accepts `CholInv`/`Inv`
  (the replacement for the removed `calc_inv` option) and preconditioned sparse solvers
  such as `sparseCGpre`/`sparseMINRESpre` for large gp2Scale problems.

## See Also

- [Repository](https://github.com/lbl-camera/gpCAM/)
- The [fvGP](https://fvgp.readthedocs.io/en/latest/) package
- The [HGDL](https://hgdl.readthedocs.io/en/latest/) package

```{div} centered-heading
Have suggestions for the API or found a bug?
```

```{div} text-center
Please submit an issue on [GitHub](https://github.com/lbl-camera/gpCAM/).
```

# fvgpOptimizer

:::{warning}
**Deprecated since gpCAM 8.7.** [`GPOptimizer`](gpOptimizer.md) handles multi-task GPs
itself; `GPOptimizer(x_data, y_data, multi_task_mode="cartesian")` is exactly what
`fvGPOptimizer(x_data, y_data)` did. `fvGPOptimizer` still works and warns
(`FutureWarning`); it is removed when gpCAM moves to fvgp 5.0.
:::

```{eval-rst}
.. autoclass:: gpcam.gp_optimizer.fvGPOptimizer
    :members:
    :inherited-members:
```

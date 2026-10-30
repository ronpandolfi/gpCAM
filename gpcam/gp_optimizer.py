#!/usr/bin/env python
import warnings
import numpy as np
from scipy.special import expit, logit
from .gp_optimizer_base import GPOptimizerBase


# TODO (for gpCAM)
#    see fvgp "gp.py" for TODOs

class GPOptimizer(GPOptimizerBase):
    """
    This class is an optimization extension of the :doc:`fvgp <fvgp:index>` package
    for single- and multi-task Gaussian Processes.
    Gaussian Processes can be initialized, trained, and conditioned; also
    the posterior can be evaluated and used via an acquisition function,
    and plugged into optimizers to find maxima.

    .. versionchanged:: 8.7.0
       ``GPOptimizer`` does multi-task GPs as well: the shape of ``y_data`` and
       ``multi_task_mode`` decide. It replaces :py:class:`fvGPOptimizer`, which is deprecated;
       ``GPOptimizer(..., multi_task_mode='cartesian')`` is what ``fvGPOptimizer`` did.

    V ... number of input points

    D ... input space dimensionality

    No ... number of outputs

    N ... arbitrary integers (N1, N2,...)


    The shape of ``y_data`` decides what is modeled: (V) or (V, 1) is a single-task GP;
    (V, No) with No > 1 is a multi-task GP, in one of three modes chosen with
    ``multi_task_mode`` (see there): ``'independent'`` (the default) for No tasks that share
    one GP's hyperparameters but are otherwise unrelated, ``'cartesian'`` for full multi-task
    learning with any kernel and missing outputs, and ``'kronecker'`` for data on a full grid
    with a separable kernel, scaling to millions of input-output pairs.

    The main logic of the ``'cartesian'`` mode is that any multi-task GP is just a single-task GP
    over a Cartesian product space of input and output space, as long as the kernel
    is flexible enough. This is the best
    way to give the user optimal control and power. At various instances, for example,
    prior-mean function, noise function, and kernel function definitions, you will
    see that the input ``x`` is defined over this combined space.
    For example, if your input space is a Euclidean 2d space and your output
    is labelled [0,1], the input to the mean, kernel, and noise functions might be

    x =

    [[0.2, 0.3,0],[0.9,0.6,0],

    [0.2, 0.3,1],[0.9,0.6,1]]

    This has to be understood and taken into account when customizing a multi-task GP.
    The data are also stored in this form: :py:attr:`x_data`, :py:attr:`y_data` and
    :py:attr:`noise_variances` are (V * No, D + 1), (V * No, 1) and (V * No), less the
    missing outputs. In the other modes they are stored as given. The examples will provide
    deeper insights.


    Parameters
    ----------
    x_data : np.ndarray or list, optional
        The input point positions. Shape (V x D), where D is the :py:attr:`fvgp.GP.input_set_dim`.
        The index set dimension, :py:attr:`fvgp.GP.index_set_dim`, is D for a single-task GP
        and with ``multi_task_mode='independent'``, and D + 1 with ``'cartesian'`` and
        ``'kronecker'``: the output index is one more dimension.
        If dealing with non-Euclidean inputs
        x_data should be a list, not a numpy array.
        In this case, both the index set and the input space dim are set to 1.
        If not provided here the GP will be initiated after ``tell()``.
    y_data : np.ndarray, optional
        The values of the data points. Shape (V) or (V, 1) for a single-task GP, (V, No) for a
        multi-task GP (see ``multi_task_mode``). With ``multi_task_mode='cartesian'`` not every
        entry in ``x_data`` needs all its outputs: ``y_data`` may have np.nan as the
        corresponding entries.
        If not provided here the GP will be initiated after ``tell()``.
    init_hyperparameters : np.ndarray, optional
        Vector of hyperparameters used to initiate the GP.
        The default is an array of ones with the right length for the anisotropic Matern
        kernel with automatic relevance determination (ARD); with ``multi_task_mode='cartesian'``
        the output direction is simply considered a separate dimension. If ``gp2Scale`` is
        enabled, the default kernel changes to the anisotropic Wendland kernel. With
        ``multi_task_mode='kronecker'`` and the default coregionalization kernel, the default is
        :py:meth:`fvgp.kernels.CoregionalizationKernel.default_hyperparameters`, drawn from the data.
        The full hyperparameter vector is passed to the kernel, mean, and noise callables,
        but the index ranges used by each callable are **disjoint and user-defined**.
        Each callable must only read the indices reserved for it. The gradient
        computation relies on this: when a hyperparameter index belongs to the mean
        function its kernel derivative is assumed zero, and vice versa.
    noise_variances : np.ndarray, optional
        A numpy array defining the uncertainties/noise in the
        ``y_data`` in form of a point-wise variance. Shape (V).
        Note: if no noise_variances are provided here, the noise_function
        callable will be used; if the callable is not provided, the noise variances
        will be set to ``abs(np.mean(y_data)) / 100.0``. If
        noise covariances are required (correlated noise), make use of the ``noise_function``.
        Only provide a noise function OR ``noise_variances``, not both.
        With ``multi_task_mode='cartesian'`` the shape is (V, No), one variance per observation,
        with np.nan where ``y_data`` has np.nan. With ``'independent'`` and ``'kronecker'`` it
        stays (V): one variance per input point, the same for every output.
    compute_device : str, optional
        One of ``cpu`` or ``gpu``, determines how linear algebra computations are executed. The default is ``cpu``.
        For ``gpu``, pytorch or cupy has to be installed manually. For advanced options see ``args``.
        If ``gp2Scale`` is enabled but no kernel is provided, the choice of the ``compute_device``
        will be particularly important. In that case, the default Wendland kernel will be computed on
        the cpu or the gpu which will significantly change the compute time depending on the compute
        architecture.
    kernel_function : Callable, optional
        A symmetric positive definite covariance function (a kernel)
        that calculates the covariance between
        data points. It is a function of the form k(x1,x2,hyperparameters, [args]).
        ``args`` is optional and is used to make :py:attr:`fvgp.GP.args` available.
        The input ``x1`` is a N1 x D array of positions, ``x2`` is a N2 x D
        array of positions (N1 x D+1 and N2 x D+1 with ``multi_task_mode='cartesian'``, the
        output index last), the hyperparameters argument
        is a 1d array of length D+1 for the default kernel and of a different
        length for user-defined kernels.
        The default is a stationary anisotropic Matern kernel of first-order
        differentiability, which performs automatic relevance determination (ARD)
        through one length scale per input dimension. With ``multi_task_mode='cartesian'`` the
        output direction is simply considered an additional dimension; this kernel should then
        only be used for tests and in the simplest of cases.
        If ``gp2Scale`` is enabled, the default changes to a compactly supported anisotropic
        Wendland kernel instead -- compact support is what makes the covariance matrix sparse.
        Which one is used depends on ``compute_device``:
        :py:func:`fvgp.kernels.wendland_anisotropic_gp2Scale_cpu` or
        :py:func:`fvgp.kernels.wendland_anisotropic_gp2Scale_gpu`.
        See :py:mod:`fvgp.kernels` for those, their support-aware sparse variants, and the
        other built-in kernels.
        The output is a matrix, an N1 x N2 numpy array.
        This callable receives the full hyperparameter vector but must only use
        the indices reserved for the kernel (disjoint from mean and noise indices).
        With ``multi_task_mode='kronecker'`` the kernel receives the inputs only and returns a list of
        factors instead, and so does its gradient; see ``multi_task_mode``.
    kernel_function_grad : Callable, optional
        A function that calculates the derivative of the ``kernel_function`` with respect to the hyperparameters.
        If provided, it will be used for local training (optimization) and can speed up the calculations.
        It accepts as input ``x1`` (a N1 x D array of positions),
        ``x2`` (a N2 x D array of positions; D+1 with ``multi_task_mode='cartesian'``) and
        ``hyperparameters`` (a 1d array of length D+1 for the default kernel).
        The default is an analytical gradient for the default kernel or a finite difference calculation otherwise.
        If ``gradient_per_direction`` is True, the function's input is x1, x2, hyperparameters (numpy array), and a direction (int).
        The output is a numpy array of shape (len(hps) x N).
        If ``gradient_per_direction`` is ``False``, the function's input is x1, x2, and hyperparameters.
        The output is a numpy array of shape (len(hyperparameters) x N1 x N2). See ``gradient_per_direction``.
    prior_mean_function : Callable, optional
        A function f(x, hyperparameters, [args]) that evaluates the prior mean at a set of input position.
        It accepts as input
        an array of positions (of shape N1 x D) and hyperparameters (a 1d array of length D+1 for the default kernel).
        Optionally, the third argument ``args`` can be defined.
        The return value is a 1d array of length N1. With ``multi_task_mode`` ``'independent'`` or
        ``'kronecker'`` it may instead be of shape (N1, No), one mean per output; a 1d mean is
        shared by all of them. Any other shape raises. With ``'cartesian'`` the positions are
        the (N1 x D+1) [x, output] points and the return value is 1d.
        If prior_mean_function is not provided, :py:meth:`fvgp.GP._default_mean_function` is used,
        which is each output's average of ``y_data``: shape (N1, No) with ``'independent'`` and
        ``'kronecker'``; with ``'cartesian'`` every (point, output) pair gets the average of its
        output (the overall average for an output without observations).
        This callable receives the full hyperparameter vector but must only use
        the indices reserved for the mean function (disjoint from kernel and noise indices).
    prior_mean_function_grad : Callable, optional
        A function that evaluates the gradient of the ``prior_mean_function`` at
        a set of input positions with respect to the hyperparameters.
        It accepts as input an array of positions (of size N1 x D) and hyperparameters
        (a 1d array of length D+1 for the default kernel).
        The return value is a 2d array of
        shape (len(hyperparameters) x N1), or (len(hyperparameters) x N1 x No) for a mean of shape
        (N1, No). If None is provided, either
        zeros are returned since the default mean function does not depend on hyperparameters,
        or a finite-difference approximation
        is used if ``prior_mean_function`` is provided.
    noise_function : Callable, optional
        The noise function is a callable f(x,hyperparameters, [args]) that returns the noise
        variances as a vector (1d np.ndarray) of len(x): fvGP models noise as diagonal.
        Correlated noise belongs in ``kernel_function``. Returning a matrix of shape
        (len(x), len(x)), dense or sparse, still works but is deprecated (FutureWarning), and
        ``linalg_mode="low_rank_<r>"`` rejects it.
        The third argument ``args`` is optional.
        The input ``x`` is a numpy array of shape (N x D), (N x D+1) [x, output] points with
        ``multi_task_mode='cartesian'``. With ``'independent'`` and ``'kronecker'`` it returns one
        variance per input point, the same for every output; Kronecker mode rejects matrix noise.
        The hyperparameter array is the same that is communicated to mean and kernel functions.
        Only provide a noise function OR a noise variance vector, not both.
        This callable receives the full hyperparameter vector but must only use
        the indices reserved for the noise function (disjoint from kernel and mean indices).
    noise_function_grad : Callable, optional
        A function that evaluates the gradient of the ``noise_function``
        at an input position with respect to the hyperparameters.
        It accepts as input an array of positions (of size N x D) and
        hyperparameters (a 1d array of length D+1 for the default kernel).
        The return value is a 2d np.ndarray of shape (len(hyperparameters) x N), the derivatives
        of the noise variances; with ``gradient_per_direction=True`` a 1d array of length N per
        direction. The matrix forms (len(hyperparameters) x N x N), or N x N per direction, are
        deprecated along with matrix-valued noise.
        If None is provided, either
        zeros are returned since the default noise function does not depend on
        hyperparameters, or, if ``noise_function`` is provided but no noise function gradient,
        a finite-difference approximation will be used.
        The same rules regarding ``gradient_per_direction`` as for the kernel definition apply here.
        That means the function will have an additional ``direction`` parameter.
    gp2Scale: bool, optional
        Turns on gp2Scale. This will distribute the covariance computations across multiple workers.
        This is an advanced feature for HPC GPs up to 10
        million data points. If gp2Scale is used, the default kernel is an anisotropic
        Wendland kernel which is compactly supported. There are a few
        things to consider (read on); this is an advanced option.
        If no kernel is provided, the ``compute_device`` option should be revisited.
        The default kernel will use the specified device to compute covariances.
        The default is False.
    gp2Scale_batch_size : int, optional
        How finely the covariance is cut up for distributed computing in gp2Scale.
        The default is 10000. It is a **maximum, not a target**: every chunk is exactly
        this size except the last along each axis, which carries the remainder -- 100000
        points at 15000 give six chunks of 15000 and one of 10000.

        What it sizes depends on ``gp2Scale_distribution``. For ``"blockwise"`` it is the
        side of a square block, so a worker evaluates
        ``gp2Scale_batch_size x gp2Scale_batch_size``. For ``"rowwise"`` it is the
        **strip width**: a worker evaluates ``gp2Scale_batch_size x N``, the whole row at
        once. With a dense kernel that allocation is what bounds worker memory, so pick
        it against the memory a worker has -- at N=100000 a strip width of 10000 is 8 GB
        in float64. Compactly supported kernels that return a sparse block
        (:py:func:`fvgp.kernels.wendland_anisotropic_gp2Scale_cpu_sparse`) never
        materialize it.

        Row-wise also produces far fewer tasks -- ``N / gp2Scale_batch_size`` against
        roughly ``(N / gp2Scale_batch_size)^2 / 2`` for block-wise -- so on a large
        cluster it usually wants a smaller value than block-wise would.

        These two meanings are **not comparable numbers**. At ``gp2Scale_batch_size=10000``
        and N=1e6, block-wise sanctions 0.8 GB per task and row-wise 80 GB, so the same
        setting can leave a computation local under one distribution and distribute it
        under the other. That is deliberate -- each mode gets the memory its own user
        actually declared.

        The budget also decides **whether to distribute at all**: anything that fits in one
        task is computed in a single kernel call instead, because scheduling costs more
        than the work. A 10 000 x 2 cross-covariance takes 0.83 ms directly against 1.8 s
        spread over 1000 dask tasks. That local call happens on the client, so it can hold
        up to one task's budget in client memory -- worth remembering under ``"rowwise"``,
        where the budget grows with N.

        Cross-covariances -- the posterior's ``k(x_data, x_pred)`` and the ``k(x_old,
        x_new)`` block of an append -- are **always cut into strips**, whatever
        ``gp2Scale_distribution`` says, with the strip width taken from the budget above.
        Blocking a tall thin matrix multiplies the task count without reducing per-task
        memory, since the long axis has to be traversed either way.
    gp2Scale_distribution : str, optional
        How the covariance computation is cut across the workers in gp2Scale.
        ``"blockwise"`` (default) sends (row block, column block) pairs and, for the
        symmetric prior covariance, schedules only the upper triangle, so the cluster
        performs half the kernel evaluations and the host mirrors the result.
        ``"rowwise"`` sends whole row strips -- one kernel call per strip against every
        column, ``gp2Scale_batch_size`` rows wide -- and has each worker return a
        finished sparse strip, so the assembly is a concatenation rather than a global
        re-sort on the host. Row-wise cannot exploit symmetry and so doubles the kernel evaluations, but
        it removes the host as a bottleneck and lowers its peak memory considerably; it is
        the better choice when assembly, not kernel evaluation, dominates the run time.
    dask_client : dask.distributed.Client, optional
        A dask client for gp2Scale, asynchronous training, and certain linear algebra operations.
        On HPC architecture, this client is provided by the job script. Please have a look at the examples.
        A local client is used as the default.
    linalg_mode : str, optional
        Controls the linear-algebra backend used to solve (K+V)x=b and compute log|K+V|.
        The default is ``None``, which selects ``"Chol"`` for standard GPs and automatically
        picks the best sparse mode for gp2Scale GPs.

        **Recommended for standard (non-gp2Scale) GPs:**

        * ``"Chol"`` *(default)* — Cholesky factorization; numerically stable and memory-efficient.
        * ``"CholInv"`` — Cholesky factorization, then explicitly stores the inverse; speeds up posterior
          covariance evaluation 3–10×. Avoid for datasets larger than ~5 000 points due to memory
          and numerical cost. Training always uses the Cholesky factor for stability.
        * ``"Inv"`` — computes and stores the explicit inverse directly (no Cholesky) even during training. Only suitable for
          very small datasets where posterior covariance is computed many times.

        **Specialized for gp2Scale (sparse covariance matrices):**

        * ``"sparseLU"`` — sparse LU factorization; good default for sparse systems up to ~50 000 points.
        * ``"sparseCG"`` — sparse conjugate-gradient iterative solver.
        * ``"sparseMINRES"`` — sparse MINRES iterative solver.
        * ``"sparseSolve"`` — direct sparse solve via scipy.
        * ``"low_rank_<r>"`` — for a kernel of known rank r (finite-feature kernels: linear,
          polynomial, truncated bases). K is computed as usual (dense, or sparse under gp2Scale)
          and factored by a rank-r pivoted Cholesky, K = L L^T, at O(N r^2). Solves then use
          the Woodbury identity and the log-determinant the matrix determinant lemma: both
          exact and deterministic, so under gp2Scale this replaces the stochastic
          log-determinant. The likelihood gradient's solves cost O(N^2 r) instead of O(N^3).
          Requires vector (diagonal), strictly positive noise. If K has a lower rank, the factor
          stops there and stays exact; if a higher one, a warning reports the trace left out,
          and the mode approximates K by its rank-r factor. Equivalent to ``"low_rank"`` with
          ``args["low_rank_rank"] = r``.
        * ``"sparseCGpre"`` — preconditioned conjugate-gradient. The preconditioner type
          is selected by ``args["sparse_preconditioner_type"]`` (default ``"ilu"``;
          also compiled incomplete Cholesky ``"ichol"``/``"ic"``/``"incomplete_cholesky"``,
          zero-fill ``"ichol0"``, legacy Python IC(0) ``"native_ic"``/``"native_ichol"``,
          ``"block_jacobi"``, ``"schwarz"``/``"additive_schwarz"``, or ``"amg"``
          (requires pyamg). The compiled IC options require the optional ``ilupp`` package.
        * ``"sparseMINRESpre"`` — preconditioned MINRES; same preconditioner choices.
          In both preconditioned modes, training builds a preconditioner only once it is
          likely to be reused (see ``args["sparse_preconditioner_lazy"]``): until the
          hyperparameters settle, and whenever building fails, a training evaluation is
          solved with unpreconditioned MINRES instead. A preconditioner containing NaN or
          inf -- an incomplete factorization can break down without raising -- counts as
          failed, and any CG solve that fails is redone with unpreconditioned MINRES. A
          failed build is not retried while K+V stays within
          ``sparse_preconditioner_max_matrix_drift`` of the matrix it failed for.
        * ``"sparseCGpre_<type>"`` / ``"sparseMINRESpre_<type>"`` — shortcut that sets
          ``args["sparse_preconditioner_type"]`` to ``<type>`` (e.g. ``"sparseCGpre_amg"``).

        **Preconditioner aliases (incomplete Cholesky):**

        The IC family has a compiled fast path and a pure-Python legacy path. The
        names in this list are the canonical backends that ``sparse_preconditioner_type``
        resolves to internally; the items after "aliases:" all map to the same backend.

        * ``"ichol"`` — thresholded incomplete Cholesky via the optional
          ``ilupp`` package (fast). **Aliases:** ``"ic"``, ``"incomplete_cholesky"``.
          Tunable through ``sparse_preconditioner_ichol_fill_in`` and
          ``sparse_preconditioner_ichol_threshold``.
        * ``"ichol0"`` — zero-fill incomplete Cholesky via ``ilupp`` (cheap to build,
          weaker as a preconditioner). No aliases.
        * ``"native_incomplete_cholesky"`` — pure-Python IC(0); slower, intended as
          a legacy/debugging fallback that always works without optional deps.
          **Aliases:** ``"native_ic"``, ``"native_ichol"``,
          ``"legacy_ic"``, ``"legacy_ichol"``,
          ``"legacy_incomplete_cholesky"``. (``"native_"`` and ``"legacy_"`` are
          interchangeable — "native" means built-in to fvGP, "legacy" reminds you
          this is the original pre-``ilupp`` path.)

        The two ``ilupp`` backends (``"ichol"``/``"ichol0"``) raise a clear
        ``ImportError`` with install instructions if ``ilupp`` is missing
        (``pip install ilupp``). All three IC backends honor the shared
        ``sparse_preconditioner_shift`` / ``_shift_growth`` / ``_shift_attempts``
        diagonal-shift-retry knobs (see the ``args`` section below).

        **Custom solver (any GP):**

        Pass an iterable of three callables ``[f_factor, f_solve, f_logdet]``:

        * ``f_factor(K)`` — receives the covariance matrix and returns a factorization object
          (or the matrix itself if no factorization is needed).
        * ``f_solve(obj, b)`` — solves the linear system and returns the solution vector.
        * ``f_logdet(obj)`` — returns the log-determinant as a scalar.

        **Migration note:** the ``calc_inv`` option from earlier gpCAM versions was removed;
        use ``linalg_mode="CholInv"`` (or ``"Inv"``) for the equivalent stored-inverse behavior.
    gradient_per_direction : bool, optional
        Only of interest if the gradient and/or Hessian of the log marginal likelihood is/are used for the training.
        If True, components of the derivative of the log marginal likelihood are
        calculated sequentially, leading to a slow-down
        but much less RAM usage. If the derivative of the kernel (and noise function) with
        respect to the hyperparameters (kernel_function_grad) is
        going to be provided, it has to be tailored: for ``gradient_per_direction=True`` it should be
        of the form f(x, hyperparameters, direction)
        and return a 2d numpy array of shape len(x1) x len(x2).
        If ``gradient_per_direction=False``, the function should be of the form f(x, hyperparameters)
        and return a numpy array of shape
        H x len(x1) x len(x2), where H is the number of hyperparameters.
        CAUTION: This array will be stored and is very large.
    cost_function : Callable, optional
        A function encoding the cost of motion through the input
        space and the cost of a measurement. Its inputs
        are an ``origin`` (np.ndarray of size V x D), ``x``
        (np.ndarray of size V x D), and the value of ``cost_func_params``;
        ``origin`` is the starting position, and ``x`` is the
        destination position. The return value is a 1d array of
        length V describing the costs as floats. The 'score' from
        acquisition_function is divided by this
        returned cost to determine the next measurement point.
        The default is a no-op.
    logging : bool, optional
        If True, logging is enabled. The default is False.
    args: dict, optional
        Advanced options. Recognized keys are:

        Stochastic-Lanczos logdet (sparse modes):

        - "random_logdet_lanczos_degree" : int; default = 20
        - "random_logdet_error_rtol" : float; default = 0.01 — target error relative to
          |log|K+V||. Sampling stops once it is reached or at
          "random_logdet_max_num_samples", whichever comes first. The log marginal
          likelihood contains -1/2 log|K+V|, so 1% means a likelihood error of about
          0.005 * |log|K+V||.
        - "random_logdet_error_atol" : float; default = None — target absolute error of
          log|K+V|. If given alone it replaces the relative default; with both given, the
          looser of the two applies.
        - "random_logdet_verbose" : True/False; default = False
        - "random_logdet_print_info" : True/False; default = False
        - "random_logdet_lanczos_compute_device" : str; default = "cpu"/"gpu"
        - "random_logdet_min_num_samples" : int; default = 10 — fewest Hutchinson
          probes to draw
        - "random_logdet_max_num_samples" : int; default = 5000 — most to draw.
          The probe count is the fidelity dial of this estimator: its variance falls
          as 1/t while its cost grows as t, so it trades accuracy against compute.
          The variance actually achieved is reported by
          :py:meth:`fvgp.gp_marginal_likelihood.GPmarginalLikelihood.log_likelihood_variance`.

        Sparse iterative solver tolerances and iteration limits:

        - "sparse_cg_tol" : float; default = 1e-5
        - "sparse_minres_tol" : float; default = 1e-5
        - "low_rank_rank" : int — the rank r for ``linalg_mode="low_rank"`` (set by
          ``"low_rank_<r>"``)
        - "low_rank_residual_tol" : float; default = 1e-8 — warn when the rank-r factor
          leaves out more than this fraction of trace(K)
        - "low_rank_tol" : float; default = 1e-12 — the factorization stops early, exact,
          once the remaining diagonal is below this fraction of max(diag(K))
        - "sparse_cg_maxiter" : int; default = 5000 for plain CG, 500 with a
          preconditioner (a working one converges in tens of iterations); None for
          scipy's default (10 N). A CG solve that does not converge within it, or produces
          a non-finite iterate (a broken preconditioner), is redone with unpreconditioned
          MINRES, with a warning.
        - "sparse_minres_maxiter" : int; default = None (use scipy default)
        - "sparse_krylov_maxiter" : int; default = None (applies to both if the
          solver-specific key is not set)
        - "sparse_block_krylov" : True/False; default = False — use a block CG
          variant when there are multiple RHS columns
        - "sparse_krylov_mode" : "single"/"block"; equivalent toggle
        - "sparse_krylov_block_size" : int — RHS block size for block CG

        Iterative-solver acceleration (``sparseCG``/``sparseMINRES`` and the
        ``*pre`` variants):

        - "sparse_krylov_warm_start" : True/False; default = False — feed the
          previous training iteration's ``KVinvY`` as ``x0`` to the next solve.
          Honored only for ``train(method='mcmc')``, whose steps are small enough for
          the previous solution to be a good guess, and only while K+V has not drifted
          (see ``sparse_preconditioner_max_matrix_drift``): a warm start from distant
          hyperparameters is worse than a cold one, and on a truncated solve it leaves
          an error that depends on which hyperparameters ran before it.
        - "sparse_preconditioner_type" : str; default = "ilu". One of "ilu",
          "ichol"/"ic"/"incomplete_cholesky", "ichol0", "native_ic"/"native_ichol",
          "block_jacobi", "schwarz"/"additive_schwarz", "amg" (requires pyamg)
        - "sparse_preconditioner_max_matrix_drift" : float; default = 0.1 — relative
          change in K+V, measured by trace and Frobenius norm, beyond which the cached
          preconditioner and any warm start are stale and rebuilt. This, rather than a
          count of reuses, is what decides reuse: a fixed count cannot tell many tiny
          MCMC steps from one jump across the domain.
        - "sparse_preconditioner_refresh_interval" : int; default = None (no cap) —
          optional hard cap on consecutive reuses, applied on top of the drift test
          above. ``set_KV`` always force-refreshes.
        - "sparse_preconditioner_lazy" : True/False; default = True — during training,
          build a new preconditioner only when it is likely to be reused (see
          "sparse_preconditioner_settle_steps"), and otherwise solve that evaluation with
          plain MINRES. Rebuilt for every evaluation, even a good preconditioner is slower
          than MINRES without one. False builds whenever the cached one is stale.
        - "sparse_preconditioner_settle_steps" : int; default = 3 — consecutive training
          evaluations, each within "sparse_preconditioner_max_matrix_drift" of the previous
          one, required before a new preconditioner is built. Separate from the drift test
          above, which compares K+V with the matrix the cached preconditioner was built for.
        - "sparse_preconditioner_block_size" : int — block size for block_jacobi
          and additive_schwarz partitions
        - "sparse_preconditioner_schwarz_overlap" : int — overlap layers for
          additive Schwarz
        - "sparse_preconditioner_drop_tol" / "sparse_preconditioner_fill_factor"
          — forwarded to scipy ``spilu`` for "ilu"
        - "sparse_preconditioner_ichol_fill_in" / "sparse_preconditioner_ichol_threshold"
          — forwarded to ``ilupp`` thresholded IC for "ichol"
        - "sparse_preconditioner_amg_*" — forwarded to pyamg
          (``max_levels``, ``max_coarse``, ``strength``, ``cycle``, etc.)
        - "sparse_preconditioner_shift" / "sparse_preconditioner_shift_growth" /
          "sparse_preconditioner_shift_attempts" — diagonal shift retry knobs
          (defaults: 0.0 / 10.0 / 5). Applies to "ichol" and "ichol0" (compiled),
          "native_ic"/"native_ichol" (legacy), "block_jacobi", and
          "additive_schwarz" when a local factorization encounters a non-PD pivot

        Practical sparse-solver guidance:

        - Keep compact-support covariance matrices genuinely sparse before using global
          factor-based preconditioners. If the kernel support is too broad, ILU/IC
          failures are usually memory/fill failures, not proof that Krylov solvers are bad.
        - Prefer ``sparseCGpre_*`` as the primary path for covariance systems, which are
          symmetric positive definite in the usual case. ``MINRES`` can be useful as a
          comparison, but it may need a stricter ``sparse_minres_tol`` to satisfy a raw
          relative residual check.
        - Sweep preconditioner parameters at the target scale. For ILU, ``drop_tol`` and
          ``fill_factor`` control a memory/solve-time tradeoff; a slightly more expressive
          factor can cost only marginally more to build but reduce solve time substantially.
        - Preconditioner reuse across nearby K+V updates is automatic and needs no tuning:
          the cached factor is kept until K+V has actually drifted past
          ``sparse_preconditioner_max_matrix_drift``. Reach for
          ``sparse_preconditioner_refresh_interval`` only to impose an extra hard cap.
        - During training a preconditioner is built lazily: while the hyperparameters
          still jump (an MCMC burn-in, or any method other than ``mcmc``, which cannot
          reuse one), evaluations are solved with unpreconditioned MINRES and nothing is
          built. Once ``sparse_preconditioner_settle_steps`` evaluations in a row stay
          close, one is built and reused for as long as the drift test allows.
          ``sparse_krylov_warm_start`` is a separate opt-in and is honored only for
          ``train(method='mcmc')``, whose steps are small enough for the previous solution
          to be a good starting guess; the other methods sample non-locally, where a warm
          start is worse than a cold one.

        Cholesky compute-device routing:

        - "Chol_factor_compute_device" : str; default = "cpu"/"gpu"
        - "update_Chol_factor_compute_device": str; default = "cpu"/"gpu"
        - "Chol_solve_compute_device" : str; default = "cpu"/"gpu"
        - "Chol_logdet_compute_device" : str; default = "cpu"/"gpu"

        GPU backend:

        - "GPU_engine" : "torch"/"cupy"; default = first available
        - "GPU_device" : str; e.g. "cuda:1" or "mps"
        - "GPU_device_index" : int — explicit CUDA device index

        Kronecker mode (``multi_task_mode='kronecker'``):

        - "kronecker_coregionalization_rank" : int; default = 1 — the rank of the default
          coregionalization kernel
        - "kronecker_coregionalization_max_outputs" : int; default = 500 — up to this many
          outputs the default kernel is the coregionalization kernel, above it the flat Matern

        All other keys will be stored and are available as part of the object instance and
        in kernel, mean, and noise functions.
    multi_task_mode : str, optional
        How the outputs of a multi-task GP, a ``y_data`` of shape (V, No) with No > 1, are
        modeled; a single-task GP ignores it. One of:

        ``'independent'`` (default) -- No GPs that share kernel, noise and hyperparameters but are
        otherwise unrelated: the outputs are No right-hand sides of one GP over the inputs,
        so only one V x V covariance is factored. ``y_data`` (V, No) is used as given, with no
        NaN, and the kernel, mean and noise functions receive the inputs only. The noise is one
        variance per input point, the same for every output (``noise_variances`` of shape
        (V)). The default mean is each output's average; a ``prior_mean_function`` may return
        (len(x),), shared, or (len(x), No), one mean per output. The likelihood is the average
        of the outputs' likelihoods, and so is its gradient. ``x_out`` in the posterior
        selects outputs by index; the outputs share one posterior covariance over the inputs
        and are uncorrelated, so ``"S"`` is block-diagonal in them (see Notes).

        ``'cartesian'`` -- full multi-task learning. Every (input, output) pair
        becomes one point of a single-task GP over the Cartesian product of input and output
        space, and the kernel, mean and noise functions see those points, with the output
        index as the last coordinate (see the class description). Any kernel, and missing
        outputs (NaN in ``y_data``), are allowed; the cost is that of a GP over V * No points.

        ``'kronecker'`` -- for data on a full grid -- every output observed at every input --
        with a separable kernel, K = K_x kron F_1 kron ... kron F_m. K is then never formed: each
        factor is eigendecomposed on its own, and the likelihood, its gradient and the posterior
        are exact at O(sum n_f^3) plus O(N sum n_f) per solve, N = V * No. This makes, e.g.,
        1000 inputs with a 100 x 100 image each (N = 1e7) tractable on one node, where K itself
        would be full rank and dense.

        * ``x_data`` and ``y_data`` (V, No) are given as always, but no Cartesian product of
          inputs and outputs is built, and ``y_data`` must not contain NaN.
        * The noise is one variance per input point, the same for every output, as for a GP
          with several right-hand sides: ``noise_variances`` has shape (V), and a
          ``noise_function`` returns a vector of len(x) (its gradient, as usual, one
          derivative per point). The default is a constant.
        * ``kernel_function(x1, x2, hyperparameters)`` receives the inputs only (N1 x D) and
          returns a list ``[K_x(x1, x2), F_1, ..., F_m]``: the input factor (N1 x N2), then the
          output factors, each square over its whole output axis, with sizes multiplying to No.
          Output o is the row-major index over the output axes, as in ``np.kron(F_1, F_2)``. The
          kernel creates the output geometry itself; for a 100 x 100 image (No = 10000),
          ``[K_x, K_rows, K_cols]`` with two 100 x 100 factors keeps every eigendecomposition
          small.
        * Without a kernel, the default is the coregionalization kernel
          :py:class:`fvgp.kernels.CoregionalizationKernel`, K_x kron B with B = W W^T + diag(kappa):
          a Matern kernel on the inputs (unit variance), and a learned covariance between the
          outputs, through ``rank`` latent processes shared by all outputs (W, No x rank) and
          one independent process per output (kappa). Its D + No * (rank + 1) hyperparameters
          are ``[length scales, W (row-major), kappa]``; their defaults and bounds come from the
          data, and its likelihood gradient is exact. ``args["kronecker_coregionalization_rank"]``
          sets the rank (default 1). Measured with 500 inputs, a likelihood evaluation plus
          its gradient takes 0.1 s at No = 100 and 1.2 s at No = 1000.
        * Above ``args["kronecker_coregionalization_max_outputs"]`` outputs (default 500),
          where B and the No * (rank + 1) hyperparameters cost more than they are worth, the
          default is instead a Matern kernel on the inputs times a Matern kernel on the flat
          output index 0 ... No-1, D + 2 hyperparameters, with a warning: it respects no
          output geometry but a 1-d sequence.
        * ``prior_mean_function`` receives the inputs only and returns an array of shape
          (len(x), No), or (len(x)) for a mean shared by all outputs.
        * ``x_out`` in the posterior selects outputs by index (0 ... No-1).
        * ``kernel_function_grad`` returns the factors' derivatives, per hyperparameter a list
          ``[dK_x, dF_1, ..., dF_m]``, with None (or zeros) for a factor that does not depend
          on it: a list of such lists over all hyperparameters, or with
          ``gradient_per_direction=True`` ``f(x1, x2, hyperparameters, direction)`` returning
          one. The likelihood gradient applies the product rule. Without it the factors are
          differentiated by finite differences, two kernel calls per hyperparameter.
        * ``linalg_mode`` is ignored, and ``gp2Scale`` cannot be combined with it.
        * Not supported: ``posterior_mean_grad``, ``posterior_covariance_grad``, the joint
          priors, and the entropy, mutual information and total correlation methods.


    Attributes
    ----------
    x_data : np.ndarray
        Datapoint positions.
    y_data : np.ndarray
        Datapoint values.
    noise_variances : np.ndarray
        Datapoint observation variances.
    hyperparameters : np.ndarray
        Current hyperparameters in use.
    K : np.ndarray
        Current prior covariance matrix of the GP.
    m : np.ndarray
        Current prior mean vector.
    V : np.ndarray
        the noise covariance matrix.

    Notes
    -----
    **Posterior results.** The posterior and information methods return dicts whose keys
    and shapes are the same in every ``multi_task_mode``. V is the number of prediction
    points, n the number of requested outputs (``x_out``, all by default) and D the input
    dimension. The shape follows ``y_data``: a single-task result has no output axis, a
    multi-task result has one.

    * ``x``: ``x_pred`` as given. ``x_out``: the requested outputs -- output indices for
      ``'independent'`` and ``'kronecker'``, output coordinates for ``'cartesian'`` -- or
      None for a single-task GP, for which passing ``x_out`` raises.
    * ``m(x)``, ``v(x)``: (V), or (V, n), the value for x_i and output t at ``[i, t]``.
    * ``m(x)_flat``, ``v_flat``: (V), or (V * n), **task-major**: pair (i, t) at i + V * t.
    * ``S``: (V, V), or (V, V, n, n) with ``S[i, j, t, u]`` = Cov(f(x_i, t), f(x_j, u)).
      ``S_flat``: (V, V), or (V * n, V * n), task-major. Both are None with
      ``variance_only=True``, which never forms S.
    * ``dm/dx``, ``dv/dx``: (V, D), or (V, D, n); with a ``direction``, (V), or (V, n).
      ``dS/dx``: shaped like ``S`` with a ``direction``, else None. ``direction``: as passed.
    * The information methods return their quantity with ``x`` and ``x_out``, over the
      requested (point, output) pairs. Their ``comp_mean`` and ``comp_cov`` take the flat
      layout of ``m(x)_flat`` and ``S_flat``.

    ``'independent'`` outputs give exactly the results of a ``'cartesian'`` GP whose kernel
    is zero between different outputs: ``S`` is block-diagonal in the outputs, and the
    entropy, mutual information and total correlation add up over them.
    """

    def __init__(
            self,
            x_data=None,
            y_data=None,
            init_hyperparameters=None,
            noise_variances=None,
            compute_device="cpu",
            kernel_function=None,
            kernel_function_grad=None,
            noise_function=None,
            noise_function_grad=None,
            prior_mean_function=None,
            prior_mean_function_grad=None,
            gp2Scale=False,
            dask_client=None,
            gp2Scale_batch_size=10000,
            gp2Scale_distribution="blockwise",
            linalg_mode=None,
            gradient_per_direction=False,
            cost_function=None,
            logging=False,
            args=None,
            multi_task_mode="independent"
    ):
        super().__init__(x_data=x_data,
                         y_data=y_data,
                         init_hyperparameters=init_hyperparameters,
                         noise_variances=noise_variances,
                         compute_device=compute_device,
                         kernel_function=kernel_function,
                         kernel_function_grad=kernel_function_grad,
                         noise_function=noise_function,
                         noise_function_grad=noise_function_grad,
                         prior_mean_function=prior_mean_function,
                         prior_mean_function_grad=prior_mean_function_grad,
                         gp2Scale=gp2Scale,
                         dask_client=dask_client,
                         gp2Scale_batch_size=gp2Scale_batch_size,
                         gp2Scale_distribution=gp2Scale_distribution,
                         linalg_mode=linalg_mode,
                         gradient_per_direction=gradient_per_direction,
                         cost_function=cost_function,
                         logging=logging,
                         args=args,
                         multi_task_mode=multi_task_mode)


class fvGPOptimizer(GPOptimizer):
    """
    Deprecated since 8.7.0, removed when gpCAM moves to fvgp 5.0.0: :py:class:`GPOptimizer`
    does multi-task GPs itself. A ``y_data`` of shape (V, No) with No > 1 makes it one,
    modeled as ``multi_task_mode`` says; ``fvGPOptimizer(x, y, ...)`` is
    ``GPOptimizer(x, y, ..., multi_task_mode="cartesian")``, and any other ``multi_task_mode``
    given to fvGPOptimizer is passed on the same way. See :py:class:`GPOptimizer` for all
    parameters and methods.

    .. deprecated:: 8.7.0
       Use ``GPOptimizer(..., multi_task_mode="cartesian")``.

    Unlike fvGPOptimizer before 8.7, a ``y_data`` with a single column is a single-task GP;
    :py:meth:`get_data` returns the data as the GP stores it (the ``"transformed x data"``
    and ``"transformed y data"`` keys are gone); and fvgp no longer keeps the data before the
    Cartesian transform (``fvgp_x_data``, ``fvgp_y_data``, ``fvgp_noise_variances``).
    """

    def __init__(
            self,
            x_data=None,
            y_data=None,
            init_hyperparameters=None,
            noise_variances=None,
            compute_device="cpu",
            kernel_function=None,
            kernel_function_grad=None,
            noise_function=None,
            noise_function_grad=None,
            prior_mean_function=None,
            prior_mean_function_grad=None,
            gp2Scale=False,
            dask_client=None,
            gp2Scale_batch_size=10000,
            gp2Scale_distribution="blockwise",
            linalg_mode=None,
            gradient_per_direction=False,
            cost_function=None,
            logging=False,
            args=None,
            multi_task_mode="cartesian"
    ):
        warnings.warn("fvGPOptimizer is deprecated and will be removed when gpCAM moves to fvgp 5.0.0: "
                      "GPOptimizer handles multi-task GPs. "
                      f"Use GPOptimizer(..., multi_task_mode={multi_task_mode!r}).", FutureWarning, stacklevel=2)
        super().__init__(x_data=x_data,
                         y_data=y_data,
                         init_hyperparameters=init_hyperparameters,
                         noise_variances=noise_variances,
                         compute_device=compute_device,
                         kernel_function=kernel_function,
                         kernel_function_grad=kernel_function_grad,
                         noise_function=noise_function,
                         noise_function_grad=noise_function_grad,
                         prior_mean_function=prior_mean_function,
                         prior_mean_function_grad=prior_mean_function_grad,
                         gp2Scale=gp2Scale,
                         dask_client=dask_client,
                         gp2Scale_batch_size=gp2Scale_batch_size,
                         gp2Scale_distribution=gp2Scale_distribution,
                         linalg_mode=linalg_mode,
                         gradient_per_direction=gradient_per_direction,
                         cost_function=cost_function,
                         logging=logging,
                         args=args,
                         multi_task_mode=multi_task_mode)


class LogGPOptimizer(GPOptimizer):
    """
    A single-task :py:class:`GPOptimizer` for strictly positive observations in (0, inf).

    Observations are modeled in log-space (the GP sees ``log(y)``), and posterior
    predictions are mapped back to the original scale with ``exp`` via
    :py:meth:`evaluate_posterior`, which guarantees strictly positive predictions and
    credible intervals. ``exp`` of a Gaussian is lognormal, so the original-scale mean
    and standard deviation are available in closed form.

    All constructor arguments are identical to :py:class:`GPOptimizer`. Note that the
    inherited :py:meth:`posterior_mean` / :py:meth:`posterior_covariance` operate in
    log-space; use :py:meth:`evaluate_posterior` for the original (positive) scale.

    Acquisition functions: :py:meth:`ask` optimizes the GP in log-space. Because ``log``
    is monotone increasing, ranking acquisitions (``variance``, ``ucb``, ``lcb``,
    ``maximum``, ``minimum``) still identify the same locations as on the original scale.
    For ``target probability``, pass bounds already in log-space (``np.log(a)``, ``np.log(b)``).
    """

    def _prepare(self, y):
        y = np.asarray(y, dtype=float)
        if np.any(y <= 0.0):
            raise ValueError("LogGPOptimizer requires strictly positive observations (y > 0).")
        return y

    def _forward(self, y):
        return np.log(y)

    def _inverse(self, z):
        return np.exp(z)

    def _forward_deriv(self, y):
        return 1.0 / y

    def _moments(self, mu, var):
        # exp(Normal(mu, var)) is lognormal
        mean = np.exp(mu + var / 2.0)
        std = np.sqrt((np.exp(var) - 1.0) * np.exp(2.0 * mu + var))
        return mean, std


class LogitGPOptimizer(GPOptimizer):
    """
    A single-task :py:class:`GPOptimizer` for observations bounded in a closed interval
    ``[lower, upper]`` (default ``[0, 1]``).

    Observations are linearly rescaled to ``[0, 1]`` and modeled in logit (log-odds)
    space (the GP sees ``logit((y - lower) / (upper - lower))``). Posterior predictions
    are mapped back to ``[lower, upper]`` via the inverse (affine ∘ sigmoid) through
    :py:meth:`evaluate_posterior`, which guarantees predictions and credible intervals
    stay inside ``(lower, upper)``. Because ``logit(0)`` / ``logit(1)`` are infinite,
    normalized observations are clipped to ``[eps, 1 - eps]`` (a warning is emitted when
    clipping occurs). The logistic-normal distribution has no closed-form moments, so
    the original-scale mean and standard deviation are estimated by Monte-Carlo.

    Parameters
    ----------
    eps : float, optional
        Clipping margin for the open interval; normalized observations are clipped to
        ``[eps, 1 - eps]`` before the logit transform. The default is 1e-6.
    n_samples : int, optional
        Number of Monte-Carlo samples used to estimate the original-scale mean/std in
        :py:meth:`evaluate_posterior`. The default is 10000.
    range : tuple of float, optional
        ``(lower, upper)`` bounds of the observation domain. Observations are linearly
        rescaled to ``[0, 1]`` before the logit transform, and posterior predictions
        are mapped back to ``[lower, upper]``. The default is ``(0.0, 1.0)``.

    Notes
    -----
    All other constructor arguments are identical to :py:class:`GPOptimizer`. The
    inherited :py:meth:`posterior_mean` / :py:meth:`posterior_covariance` operate in
    logit-space (on the rescaled ``[0, 1]`` data); use :py:meth:`evaluate_posterior`
    for the original ``[lower, upper]`` scale. The acquisition-function note for
    :py:class:`LogGPOptimizer` applies here too (pass ``target probability`` bounds
    in logit-space, where they refer to the rescaled ``[0, 1]`` data).
    """

    def __init__(self, x_data=None, y_data=None, eps=1e-6, n_samples=10000,
                 range=(0.0, 1.0), **kwargs):
        lower, upper = float(range[0]), float(range[1])
        if not lower < upper:
            raise ValueError(
                f"LogitGPOptimizer range must have lower < upper; got {range}.")
        self.range = (lower, upper)
        self.eps = eps
        self.n_samples = n_samples
        super().__init__(x_data=x_data, y_data=y_data, **kwargs)

    def _prepare(self, y):
        a, b = self.range
        y = np.asarray(y, dtype=float)
        # Linearly rescale [a, b] -> [0, 1] before clipping to (eps, 1-eps).
        y_norm = (y - a) / (b - a)
        if np.any(y_norm < self.eps) or np.any(y_norm > 1.0 - self.eps):
            warnings.warn(
                f"LogitGPOptimizer clipped observations near the boundary of "
                f"range=({a}, {b}) before the logit transform.")
        return np.clip(y_norm, self.eps, 1.0 - self.eps)

    def _forward(self, y_norm):
        # `y_norm` is the prepared (rescaled, clipped) value in (eps, 1-eps).
        return logit(y_norm)

    def _inverse(self, z):
        # Sigmoid maps to (0, 1); affine maps back to (lower, upper).
        a, b = self.range
        return a + (b - a) * expit(z)

    def _forward_deriv(self, y_norm):
        # Composition g(y) = logit((y - a) / (b - a)) gives
        #   g'(y) = 1 / ((b - a) * y_norm * (1 - y_norm))
        a, b = self.range
        return 1.0 / ((b - a) * y_norm * (1.0 - y_norm))

    def _moments(self, mu, var):
        # Logistic-normal (rescaled to [a, b]) has no closed-form moments; use MC.
        # `_samples` already pushes through self._inverse, which handles the rescale.
        samples = self._samples(mu, var, self.n_samples)
        return samples.mean(axis=-1), samples.std(axis=-1)

    def __getstate__(self):
        state = super().__getstate__()
        state["eps"] = self.eps
        state["n_samples"] = self.n_samples
        state["range"] = self.range
        return state

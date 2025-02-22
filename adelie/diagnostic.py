from . import adelie_core as core
from .constraint import (
    ConstraintBase32,
    ConstraintBase64,
)
from .glm import (
    GlmBase32,
    GlmBase64,
    GlmMultiBase32,
    GlmMultiBase64,
)
from . import logger
from . import matrix
from .matrix import (
    MatrixNaiveBase32,
    MatrixNaiveBase64,
)
from .state import (
    render_dual_groups,
)
from IPython.display import HTML
from itertools import cycle
from scipy.sparse import csr_matrix
from typing import Union
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from scipy.special import softmax, expit
from sklearn.metrics import roc_auc_score


def auc_roc(
    etas: np.ndarray,
    y: np.ndarray,
    multinomial: bool
):
    if multinomial and y.shape[1] == 2:
        multinomial = False
        y = np.argmax(y, axis=1)

    n_lambdas = etas.shape[0]
    if multinomial:
        val_probs = softmax(etas, axis=-1).squeeze()
        y_true = np.argmax(y, axis=1)
        if len(np.unique(y_true)) != y.shape[1]:
            return None
        return np.array([
            roc_auc_score(y_true, val_probs[i, :, :], multi_class='ovr')
                for i in range(n_lambdas)
        ])
    else:
        proba = expit(etas)
        val_probs = np.stack((1 - proba, proba), axis=-1).squeeze()

        if len(np.unique(y)) != 2:
            return None

        return np.array([
            roc_auc_score(y, val_probs[i, :, 1])
                for i in range(n_lambdas)
        ])

def test_error_hamming(
    etas: np.ndarray,
    y: np.ndarray,
    multinomial: bool
):
    n_lambdas = etas.shape[0]

    if multinomial:
        val_probs = softmax(etas, axis=-1).squeeze()
        y_true = np.argmax(y, axis=1)

        return np.array([
            np.sum(np.argmax(val_probs[i, :, :], axis=1) != y_true) / len(y_true)
                for i in range(n_lambdas)
        ])
    else:
        proba = expit(etas)
        val_probs = np.stack((1 - proba, proba), axis=-1).squeeze()

        return np.array([
            np.sum(np.argmax(val_probs[i, :, :], axis=1) != y) / len(y)
                for i in range(n_lambdas)
        ])
    
def test_error_mse(
    etas: np.ndarray,
    y: np.ndarray,
):
    n_lambdas = etas.shape[0]

    return np.array([
        np.sum((etas[i, :] - y)**2) / len(y)
            for i in range(n_lambdas)
    ])
        


def predict(
    X: Union[np.ndarray, MatrixNaiveBase32, MatrixNaiveBase64],
    betas: Union[np.ndarray, csr_matrix],
    intercepts: np.ndarray,
    *,
    offsets: np.ndarray =None,
    n_threads: int =1,
):
    """Computes the linear predictions.

    The single-response linear prediction is given by
    
    .. math::
        \\begin{align*}
            \\hat{\\eta} = X\\beta + \\beta_0 \\mathbf{1} + \\eta^0
        \\end{align*}

    The multi-response linear prediction is given by

    .. math::
        \\begin{align*}
            \\hat{\\eta} = 
            (X\\otimes I_K) \\beta + 
            (\\mathbf{1}\\otimes I_k) \\beta_0 + 
            \\eta^0
        \\end{align*}

    The single or multi-response is detected based on the shape of ``intercepts``.
    If ``intercepts`` one-dimensional, we assume single-response.
    Otherwise, we assume multi-response.

    Parameters
    ----------
    X : (n, p) Union[ndarray, MatrixNaiveBase32, MatrixNaiveBase64]
        Feature matrix.
        It is typically one of the matrices defined in :mod:`adelie.matrix` submodule or :class:`numpy.ndarray`.
    betas : (L, p) or (L, p*K) Union[ndarray, csr_matrix]
        Coefficient vectors :math:`\\beta`.
    intercepts : (L,) or (L, K) ndarray
        Intercepts :math:`\\beta_0`.
    offsets : (n,) or (n, K) ndarray, optional
        Observation offsets :math:`\\eta^0`.
        Default is ``None``, in which case, it is set to 
        ``np.zeros(n)`` if ``y`` is single-response
        and ``np.zeros((n, K))`` if multi-response.
    n_threads : int, optional
        Number of threads.
        Default is ``1``.

    Returns
    -------
    linear_preds : (L, n) or (L, n, K) ndarray
        Linear predictions.
    """
    intercepts = np.atleast_1d(intercepts)
    is_multi = len(intercepts.shape) == 2
    if is_multi:
        K = intercepts.shape[1]
        X = matrix.kronecker_eye(X, K, n_threads=n_threads)
        n = X.rows() // K
        y_shape = (n, K)
    else:
        if isinstance(X, np.ndarray):
            X = matrix.dense(X, method="naive", n_threads=n_threads)
        n = X.rows()
        y_shape = (n,)

    dtype = (
        np.float32
        if isinstance(X, MatrixNaiveBase32) else
        np.float64
    )

    if offsets is None:
        offsets = np.zeros(y_shape, dtype=dtype)

    if isinstance(betas, np.ndarray):
        betas = np.atleast_2d(betas)

    L = betas.shape[0]

    etas = np.zeros((L,) + y_shape, order="C", dtype=dtype)
    if isinstance(betas, np.ndarray):
        for i in range(etas.shape[0]):
            X.btmul(0, X.cols(), betas[i], etas[i].ravel())
    elif isinstance(betas, csr_matrix):
        X.sp_tmul(betas, etas.reshape((L, -1))) 
    else:
        raise RuntimeError("beta is not one of np.ndarray or scipy.sparse.csr_matrix.")
    etas += intercepts[:, None] + offsets

    return etas


def objective(
    X: Union[np.ndarray, MatrixNaiveBase32, MatrixNaiveBase64], 
    glm: Union[GlmBase32, GlmBase64, GlmMultiBase32, GlmMultiBase64],
    betas: Union[np.ndarray, csr_matrix], 
    intercepts: np.ndarray,
    lmdas: np.ndarray, 
    *,
    groups: np.ndarray =None, 
    alpha: float =1, 
    penalty: np.ndarray =None,
    offsets: np.ndarray =None,
    relative: bool =True,
    add_penalty: bool =True,
    n_threads: int =1,
):
    """Computes the group elastic net objective.

    See :func:`adelie.solver.grpnet` for details.

    Parameters
    ----------
    X : (n, p) Union[ndarray, MatrixNaiveBase32, MatrixNaiveBase64]
        Feature matrix :math:`X`.
        It is typically one of the matrices defined in :mod:`adelie.matrix` submodule or :class:`numpy.ndarray`.
    glm : Union[GlmBase32, GlmBase64, GlmMultiBase32, GlmMultiBase64] 
        GLM object.
        It is typically one of the GLM classes defined in :mod:`adelie.glm` submodule.
    betas : (L, p) or (L, p*K) Union[ndarray, csr_matrix]
        Coefficient vectors :math:`\\beta`.
    intercepts : (L,) or (L, K) ndarray
        Intercepts :math:`\\beta_0`.
    lmdas : (L,) ndarray 
        Regularization parameters :math:`\\lambda`.
        It is only used when ``add_penalty=True``.
        Otherwise, the user may pass ``None``.
    groups : (G,) ndarray, optional
        List of starting indices to each group where `G` is the number of groups.
        ``groups[i]`` is the starting index of the ``i`` th group. 
        If ``glm`` is of multi-response type, then
        ``groups[i]`` is the starting *feature* index of the ``i`` th group.
        In either case, ``groups[i]`` must then be a value in the range :math:`\\{1,\\ldots, p\\}`.
        Default is ``None``, in which case it is set to ``np.arange(p)``.
    alpha : float, optional
        Elastic net parameter :math:`\\alpha`.
        It must be in the range :math:`[0,1]`.
        It is only used when ``add_penalty=True``.
        Otherwise, the user may pass ``None``.
        Default is ``1``.
    penalty : (G,) ndarray, optional
        Penalty factor for each group in the same order as ``groups``.
        It must be a non-negative vector.
        It is only used when ``add_penalty=True``.
        Default is ``None``, in which case, it is set to ``np.sqrt(group_sizes)``.
    offsets : (n,) or (n, K) ndarray, optional
        Observation offsets :math:`\\eta^0`.
        Default is ``None``, in which case, it is set to 
        ``np.zeros(n)`` if ``y`` is single-response
        and ``np.zeros((n, K))`` if multi-response.
    relative : bool, optional
        If ``True``, then the full loss, :math:`\\ell(\\eta^\\star)`, is computed at the saturated model
        and the difference :math:`\\ell(\\eta)-\\ell(\\eta^\\star)` is provided,
        which will always be non-negative.
        This effectively computes loss *relative* to the saturated model.
        Default is ``True``.
    add_penalty : bool, optional
        If ``False``, the regularization term is removed 
        so that only the loss part is calculated. 
        Default is ``True``.
    n_threads : int, optional
        Number of threads.
        Default is ``1``.
    
    Returns
    -------
    obj : (L,) ndarray
        Group elastic net objectives.

    See Also
    --------
    adelie.solver.grpnet
    """
    X_raw = X
    y = glm.y

    if groups is None:
        p = X.shape[1]
        groups = np.arange(p, dtype=int)

    if glm.is_multi:
        K = y.shape[1]
        X = matrix.kronecker_eye(X, K, n_threads=n_threads)
        p = X.cols() // K
        groups = groups * K 
        group_sizes = np.concatenate([groups, [p*K]], dtype=int)
        group_sizes = group_sizes[1:] - group_sizes[:-1]
    else:
        if isinstance(X, np.ndarray):
            X = matrix.dense(X, method="naive", n_threads=n_threads)
        p = X.cols()
        group_sizes = np.concatenate([groups, [p]], dtype=int)
        group_sizes = group_sizes[1:] - group_sizes[:-1]

    
    dtype = (
        np.float32
        if isinstance(X, MatrixNaiveBase32) else
        np.float64
    )

    if penalty is None:
        penalty = np.sqrt(group_sizes).astype(dtype)

    etas = predict(
        X=X_raw,
        betas=betas,
        intercepts=intercepts,
        offsets=offsets,
        n_threads=n_threads,
    )

    # compute loss part
    objs = np.array([
        glm.loss(etas[i])
        for i in range(etas.shape[0])
    ], dtype=dtype)

    # relative to saturated model
    if relative:
        objs -= glm.loss_full()

    # compute regularization part
    if add_penalty:
        penalty_f = None
        if isinstance(betas, np.ndarray):
            penalty_f = {
                np.float32: core.solver.compute_penalty_dense_32,
                np.float64: core.solver.compute_penalty_dense_64,
            }[dtype]
        elif isinstance(betas, csr_matrix): 
            penalty_f = {
                np.float32: core.solver.compute_penalty_sparse_32,
                np.float64: core.solver.compute_penalty_sparse_64,
            }[dtype]
        objs += lmdas * penalty_f(
            groups,
            group_sizes,
            penalty,
            alpha,
            betas,
            n_threads,
        )

    return objs


def residuals(
    glm: Union[GlmBase32, GlmBase64, GlmMultiBase32, GlmMultiBase64],
    etas: np.ndarray,
):
    """Computes the residuals.

    The residual is given by 
    
    .. math::
        \\begin{align*}
            \\hat{r} = -\\nabla \\ell(\\eta)
        \\end{align*}

    Parameters
    ----------
    glm : Union[GlmBase32, GlmBase64, GlmMultiBase32, GlmMultiBase64] 
        GLM object.
        It is typically one of the GLM classes defined in :mod:`adelie.glm` submodule.
    etas : (L, n) or (L, n, K) ndarray
        Linear predictions.

    Returns
    -------
    resids : (L, n) or (L, n, K) ndarray
        Residuals.

    See Also
    --------
    adelie.diagnostic.predict
    """
    dtype = (
        np.float32
        if isinstance(glm, (GlmBase32, GlmMultiBase32)) else
        np.float64
    )
    resids = np.empty(etas.shape, dtype=dtype)
    for eta, resid in zip(etas, resids):
        glm.gradient(eta, resid)
    return resids


def gradients(
    X: Union[np.ndarray, MatrixNaiveBase32, MatrixNaiveBase64],
    resids: np.ndarray,
    *, 
    n_threads: int =1,
):
    """Computes the gradients.

    The gradient for the single-response is given by

    .. math::
        \\begin{align*}
            \\hat{\\gamma} = X^{\\top} \\hat{r}
        \\end{align*}

    The gradient for the multi-response is given by

    .. math::
        \\begin{align*}
            \\hat{\\gamma} = (X\\otimes I_K)^{\\top} \\mathrm{vec}(\\hat{r}^\\top)
        \\end{align*}

    In both cases, :math:`\\hat{r}` is the residual as in
    :func:`adelie.diagnostic.residuals`.

    Parameters
    ----------
    X : (n, p) Union[ndarray, MatrixNaiveBase32, MatrixNaiveBase64]
        Feature matrix.
        It is typically one of the matrices defined in :mod:`adelie.matrix` submodule or :class:`numpy.ndarray`.
    resids : (L, n) or (L, n, K) ndarray
        Residuals.
    n_threads : int, optional
        Number of threads.
        Default is ``1``.

    Returns
    -------
    grads : (L, p) or (L, p, K) ndarray
        Gradients.

    See Also
    --------
    adelie.diagnostic.residuals
    """
    is_multi = len(resids.shape) == 3

    if is_multi:
        K = resids.shape[2]
        X = matrix.kronecker_eye(X, K, n_threads=n_threads)
        grad_shape = (X.cols() // K, K)
    else:
        if isinstance(X, np.ndarray):
            X = matrix.dense(X, method="naive", n_threads=n_threads)
        grad_shape = (X.cols(),)

    dtype = (
        np.float32
        if isinstance(X, MatrixNaiveBase32) else
        np.float64
    )

    grads = np.empty((resids.shape[0],) + grad_shape, dtype=dtype)
    ones = np.ones(np.prod(resids.shape[1:]))
    for i in range(grads.shape[0]):
        X.mul(resids[i].ravel(), ones, grads[i].ravel())
    return grads


def gradient_norms(
    grads: np.ndarray,
    betas: csr_matrix,
    duals: csr_matrix,
    lmdas: np.ndarray,
    *, 
    constraints: list[Union[ConstraintBase32, ConstraintBase64]] =None,
    groups: np.ndarray =None,
    alpha: float =1,
    penalty: np.ndarray =None,
):
    """Computes the group-wise gradient norms.

    The group-wise gradient norm is given by :math:`\\hat{h} \\in \\mathbb{R}^{G}` where

    .. math::
        \\begin{align*}
            \\hat{h}_g = \\|
                \\hat{\\gamma}_g - 
                \\lambda (1-\\alpha) \\omega_g \\beta_g -
                \\phi_g'(\\beta_g)^\\top \\mu_g
            \\|_2  \\quad g=1,\\ldots, G
        \\end{align*}

    where
    :math:`\\hat{\\gamma}_g` is the gradient as in :func:`adelie.diagnostic.gradients`,
    :math:`\\lambda` is the regularization,
    :math:`\\alpha` is the elastic net proportion,
    :math:`\\omega_g` is the penalty factor,
    :math:`\\beta_g` is the coefficient block for group :math:`g`,
    :math:`\\phi_g` is the constraint function for group :math:`g`,
    and :math:`\\mu_g` is the dual block for group :math:`g`.

    Parameters
    ----------
    grads : (L, p) or (L, p, K) ndarray
        Gradients.
    betas : (L, p) or (L, p*K) csr_matrix
        Coefficient vectors :math:`\\beta`.
    duals : (L, d) csr_matrix
        Dual vectors :math:`\\mu`.
    lmdas : (L,) ndarray
        Regularization parameters :math:`\\lambda`.
    constraints : (G,) list[Union[ConstraintBase32, ConstraintBase64]], optional
        List of constraints for each group.
        ``constraints[i]`` is the constraint object corresponding to group ``i``.
        If ``constraints[i]`` is ``None``, then the ``i`` th group is unconstrained.
        If ``None``, every group is unconstrained.
        Default is ``None``.
    groups : (G,) ndarray, optional
        List of starting indices to each group where `G` is the number of groups.
        ``groups[i]`` is the starting index of the ``i`` th group. 
        If ``glm`` is of multi-response type, then
        ``groups[i]`` is the starting *feature* index of the ``i`` th group.
        In either case, ``groups[i]`` must then be a value in the range :math:`\\{1,\\ldots, p\\}`.
        Default is ``None``, in which case it is set to ``np.arange(p)``.
    alpha : float, optional
        Elastic net parameter :math:`\\alpha`.
        It must be in the range :math:`[0,1]`.
        Default is ``1``.
    penalty : (G,) ndarray, optional
        Penalty factor for each group in the same order as ``groups``.
        It must be a non-negative vector.
        Default is ``None``, in which case, it is set to ``np.sqrt(group_sizes)``.

    Returns
    -------
    norms : (L, G) ndarray
        Gradient norms.

    See Also
    --------
    adelie.diagnostic.gradients
    """
    is_multi = len(grads.shape) == 3

    if groups is None:
        groups = np.arange(p)

    if is_multi:
        p, K = grads.shape[1:]
        groups = groups * K
        group_sizes = np.concatenate([groups, [p*K]], dtype=int)
        group_sizes = group_sizes[1:] - group_sizes[:-1]
    else:
        p = grads.shape[-1]
        group_sizes = np.concatenate([groups, [p]], dtype=int)
        group_sizes = group_sizes[1:] - group_sizes[:-1]

    if penalty is None:
        penalty = np.sqrt(group_sizes)
    penalty = np.repeat(penalty, group_sizes)

    L = grads.shape[0]
    grads = grads.reshape((L, -1)) - betas.multiply(lmdas[:, None] * (1 - alpha) * penalty[None])

    if not (constraints is None):
        assert len(constraints) == len(groups)
        dtype = None
        for constraint in constraints:
            if constraint is None: continue
            dtype = (
                np.float32
                if isinstance(constraint, ConstraintBase32) else
                np.float64
            )
            break
        if not (dtype is None):
            dual_groups = render_dual_groups(constraints)
            mu_grads = np.zeros(grads.shape, dtype=dtype)
            for k in range(L):
                beta_curr = betas[k].toarray()[0].astype(dtype)
                mu_curr = duals[k].toarray()[0].astype(dtype)
                mu_grads_curr = mu_grads[k].astype(dtype)
                for constraint, g, gs, dg in zip(
                    constraints,
                    groups,
                    group_sizes,
                    dual_groups,
                ):
                    if constraint is None: continue
                    constraint.gradient_static(
                        beta_curr[g:g+gs],
                        mu_curr[dg:dg+constraint.dual_size],
                        mu_grads_curr[g:g+gs],
                    ) 
            grads -= mu_grads

    return np.array([
        np.linalg.norm(grads[:, g:g+gs], axis=-1)
        for g, gs in zip(groups, group_sizes)
    ]).T


def gradient_scores(
    grad_norms: np.ndarray,
    lmdas: np.ndarray,
    *, 
    alpha: float =1,
    penalty: np.ndarray =None,
):
    """Computes the gradient scores.

    The gradient score is given by

    .. math::
        \\begin{align*}
            \\hat{s}_g = 
            \\begin{cases}
                \\hat{h}_g \\cdot (\\alpha p_g)^{-1} ,& \\alpha p_g > 0 \\\\
                \\lambda ,& \\alpha p_g = 0
            \\end{cases}
            \\qquad
            g = 1,\\ldots, G
        \\end{align*}

    where :math:`\\hat{h}` is the gradient norm as in
    :func:`adelie.diagnostic.gradient_norms`.

    Parameters
    ----------
    grad_norms : (L, G) ndarray
        Gradient norms.
    lmdas : (L,) ndarray
        Regularization parameters :math:`\\lambda`.
    penalty : (G,) ndarray 
        Penalty factor for each group.
        It must be a non-negative vector.
    alpha : float, optional
        Elastic net parameter :math:`\\alpha`.
        It must be in the range :math:`[0,1]`.
        Default is ``1``.

    Returns
    -------
    scores : (L, G) ndarray
        Gradient scores.  

    See Also
    --------
    adelie.diagnostic.gradient_norms
    """
    denom = alpha * penalty
    scores = np.divide(grad_norms, denom[None], where=denom[None] > 0)
    scores[:, denom <= 0] = lmdas[:, None]
    return scores


def coefficient(
    lmda: float,
    betas: csr_matrix,
    intercepts: np.ndarray,
    lmdas: np.ndarray,
):
    """Computes the coefficient at :math:`\\lambda` using linear interpolation of solutions.

    The linearly interpolated coefficient is given by
    
    .. math::
        \\begin{align*}
            \\hat{\\beta}(\\lambda)
            =
            \\frac{\\lambda - \\lambda_{k+1}}{\\lambda_{k} - \\lambda_{k+1}}
            \\hat{\\beta}(\\lambda_k)
            +
            \\frac{\\lambda_{k} - \\lambda}{\\lambda_{k} - \\lambda_{k+1}}
            \\hat{\\beta}(\\lambda_{k+1})
        \\end{align*}

    if :math:`\\lambda \\in [\\lambda_{k+1}, \\lambda_k]`.
    If :math:`\\lambda` lies above the largest value in ``lmdas`` or below the smallest value,
    then we simply take the solution at the respective ends.
    The same formula holds for intercepts.

    Parameters
    ----------
    lmda : float
        New regularization parameter at which to find the solution.
    betas : (L, p) csr_matrix
        Coefficient vectors :math:`\\beta`.
    intercepts : (L,) ndarray
        Intercepts.
    lmdas : (L,) ndarray
        Regularization parameters :math:`\\lambda`.

    Returns
    -------
    beta : (1, p) csr_matrix
        Linearly interpolated coefficient vector at :math:`\\lambda`.
    intercept : float
        Linearly interpolated intercept at :math:`\\lambda`.
    """
    if len(lmdas) == 0:
        raise RuntimeError("lmdas must be non-empty!")
    if len(lmdas) == 1:
        return betas, lmdas
    order = np.argsort(lmdas)
    idx = np.searchsorted(
        lmdas,
        lmda,
        sorter=order,
    )
    idx = lmdas.shape[0] - idx
    if idx == 0 or idx == lmdas.shape[0]:
        logger.logger.warning(
            "lmda is not within the range of the saved lambdas. " +
            "Returning boundary solution."
        )
        idx = np.clip(idx, 0, lmdas.shape[0]-1)
        return betas[idx], intercepts[idx]

    left, right = betas[idx-1], betas[idx]
    weight = (lmda - lmdas[idx]) / (lmdas[idx-1] - lmdas[idx])
    beta = left.multiply(weight) + right.multiply(1-weight)
    left, right = intercepts[idx-1], intercepts[idx]
    intercept = weight * left + (1-weight) * right

    return beta, intercept


def plot_coefficients(
    betas: csr_matrix,
    lmdas: np.ndarray,
    groups: np.ndarray,
    group_sizes: np.ndarray,
    *,
    l2_norm: bool =False,
):
    """Plots the coefficient profile.

    Parameters
    ----------
    betas : (L, p) csr_matrix
        Coefficient vectors :math:`\\beta`.
    lmdas : (L,) ndarray
        Regularization parameters :math:`\\lambda`.
    groups : (G,) ndarray
        List of starting indices to each group where `G` is the number of groups.
        ``groups[i]`` is the starting index of the ``i`` th group. 
    group_sizes : (G,) ndarray
        List of group sizes corresponding to each element of ``groups``.
        ``group_sizes[i]`` is the size of the ``i`` th group.
    l2_norm : bool, optional
        If ``True``, the :math:`\\ell_2` norms of each coefficient group
        is plotted rather than the raw coefficient values.
        This may be more intuitive to visualize since there is only one path
        associated with a group.
        Default is ``False``.

    Returns
    -------
    fig, ax
    """
    tls = -np.log(lmdas)

    fig, ax = plt.subplots(figsize=(9, 6), layout="constrained")
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    colors_it = cycle(colors)

    if l2_norm:
        for g, gs in zip(groups, group_sizes):
            curr_block = betas[:, g:g+gs]
            if curr_block.nnz == 0:
                continue
            curr_block = curr_block.toarray()
            curr_block_norms = np.linalg.norm(curr_block, axis=-1)
            color = next(colors_it)
            ax.plot(tls, curr_block_norms, linestyle="-", color=color)

        ax.set_title(r"Coefficient $\ell_2$-Norm Profile")
        ax.set_ylabel(r"$\|\beta\|_2$")
        ax.set_xlabel(r"-$\log(\lambda)$")
    else:
        for g, gs in zip(groups, group_sizes):
            curr_block = betas[:, g:g+gs]
            if curr_block.nnz == 0:
                continue
            curr_block = curr_block.toarray()
            color = next(colors_it)
            ax.plot(tls, curr_block, linestyle="-", color=color)

        ax.set_title(r"Coefficient Profile")
        ax.set_ylabel(r"$\beta$")
        ax.set_xlabel(r"-$\log(\lambda)$")

    return fig, ax


def plot_devs(
    lmdas: np.ndarray,
    devs: np.ndarray,
):
    """Plots the deviance profile.

    Parameters
    ----------
    lmdas : (L,) ndarray
        Regularization parameters :math:`\\lambda`.
    devs : (L,) ndarray
        Deviances.

    Returns
    -------
    fig, ax
    """
    tls = -np.log(lmdas)

    fig, ax = plt.subplots(figsize=(9, 6), layout="constrained")
    ax.plot(tls, devs, linestyle='-', color='r', marker='.')
    ax.set_title(r"Deviance Profile")
    ax.set_ylabel(r"Deviance Explained (\%)")
    ax.set_xlabel(r"$-\log(\lambda)$")

    return fig, ax


def plot_set_sizes(
    groups: np.ndarray,
    screen_sizes: np.ndarray,
    active_sizes: np.ndarray,
    lmdas: np.ndarray,
    screen_rule: str,
    *,
    ratio: bool =False,
    exclude: list =[],
    axes = None,
):
    """Plots the set sizes.

    Parameters
    ----------
    groups : (G,) ndarray
        List of starting indices to each group where `G` is the number of groups.
        ``groups[i]`` is the starting index of the ``i`` th group. 
    screen_sizes : (L,) ndarray
        Screen set sizes.
    active_sizes : (L,) ndarray
        Active set sizes.
    lmdas : (L,) ndarray
        Regularization parameters :math:`\\lambda`.
    ratio : bool, optional
        ``True`` if plot should normalize the set sizes
        by the total number of groups.
        Default is ``False``.
    exclude : list, optional
        The sets to exclude from plotting.
        It must be a subset of the following:

            - ``"active"``: active set.
            - ``"screen"``: screen set.

        Default is ``[]``.
    axes
        Matplotlib axes object.

    Returns
    -------
    fig, ax
        If ``axes`` is ``None``, both are returned.
    ax
        If ``axes`` is not ``None``, then only ``ax`` is returned.
    """
    make_ax = axes is None
    ax = axes

    include = ["active", "screen"]
    if len(exclude) > 0:
        include = list(set(include) - set(exclude))
    
    include_map = {
        "active": 0,
        "screen": 1,
    } 

    ys = [
        active_sizes,
        screen_sizes,
    ]
    if ratio:
        ys = [y / len(groups) for y in ys]

    labels = [
        "active",
        screen_rule,
    ]
    colors = [
        "tab:red",
        "tab:blue",
    ]
    markers = ["o", "v"]

    y_sizes = np.array([y.shape[0] for y in ys])
    iters = np.min(y_sizes)
    if not np.all(y_sizes == iters):
        logger.logger.warning(
            "The sets do not all have the same set sizes. " +
            "The plot will only show up to the smallest set."
        )
    tls = -np.log(lmdas[:iters])
    ys = [y[:iters] for y in ys]

    ys = [ys[include_map[s]] for s in include]
    labels = [labels[include_map[s]] for s in include]
    colors = [colors[include_map[s]] for s in include]
    markers = [markers[include_map[s]] for s in include]

    if make_ax:
        fig, ax = plt.subplots(figsize=(9, 6), layout="constrained")

    for y, marker, color, label in zip(ys, markers, colors, labels):
        ax.plot(
            tls,
            y, 
            linestyle="None", 
            marker=marker,
            markerfacecolor="None",
            color=color,
            label=label,
        )
    ax.legend()
    ax.set_title("Set Size Profile")
    if ratio:
        ax.set_ylabel("Proportion of Groups")
    else:
        ax.set_ylabel("Number of Groups")
    ax.set_xlabel(r"$-\log(\lambda)$")

    if make_ax:
        return fig, ax
    return ax


def plot_benchmark(
    total_time: np.ndarray,
    benchmark_screen: np.ndarray,
    benchmark_fit_screen: np.ndarray,
    benchmark_fit_active: np.ndarray,
    benchmark_kkt: np.ndarray,
    benchmark_invariance: np.ndarray,
    n_valid_solutions: np.ndarray,
    lmdas: np.ndarray,
    *,
    relative: bool =False,
):
    """Plots benchmark times.

    Parameters
    ----------
    total_time : float
        Total time taken for the core routine.
    benchmark_screen : (B,) ndarray
        Benchmark timings for screening.
    benchmark_fit_screen : (B,) ndarray
        Benchmark timings for fitting on screen set.
    benchmark_fit_active : (B,) ndarray
        Benchmark timings for fitting on active set.
    benchmark_kkt : (B,) ndarray
        Benchmark timings for KKT checks.
    benchmark_invariance : (B,) ndarray
        Benchmark timings for invariance step.
    n_valid_solutions : (B,) ndarray
        Flags that indicate whether each iteration resulted in a valid solution.
    lmdas : (L,) ndarray
        Regularization parameters :math:`\\lambda`.
    relative : bool, optional
        If ``True``, the time breakdown plot is relative to the total time,
        therefore plotting the proportion of time spent in each category.
        Otherwise, the absolute times are shown.
        Default is ``False``.

    Returns
    -------
    fig, ax
    """
    def _squash_times(ts):
        idx = 0
        new_ts = []
        while idx < len(n_valid_solutions):
            if n_valid_solutions[idx]:
                new_ts.append(ts[idx]) 
            else:
                t = 0
                while (idx < len(n_valid_solutions)) and (not n_valid_solutions[idx]):
                    t += ts[idx]
                    idx += 1
                if idx < len(n_valid_solutions):
                    t += ts[idx]
                new_ts.append(t)
            idx += 1
        return np.array(new_ts)

    times = [
        _squash_times(benchmark_screen),
        _squash_times(benchmark_fit_screen),
        _squash_times(benchmark_fit_active),
        _squash_times(benchmark_kkt),
        _squash_times(benchmark_invariance),
    ]
    n_iters = np.min([lmdas.shape[0]] + [len(t) for t in times])
    times = [t[:n_iters] for t in times]
    lmdas = lmdas[:n_iters]
    tlmdas = -np.log(lmdas)

    colors = [
        "green",
        "orange",
        "red",
        "purple",
        "brown",
    ]
    markers = [
        ".", "v", "^", "+", "*",
    ]
    labels = [
        "screen", "fit-screen", "fit-active", "kkt", "invariance",
    ]

    fig, axes = plt.subplots(1, 2, figsize=(9, 6), layout="constrained")
    for tm, color, marker, label in zip(times, colors, markers, labels):
        axes[0].plot(
            tlmdas,
            tm,
            linestyle="None",
            color=color,
            marker=marker,
            markerfacecolor="None",
            label=label,
        )
    axes[0].legend()
    axes[0].set_title("Benchmark Profile")
    axes[0].set_ylabel("Time (s)")
    axes[0].set_xlabel(r"$-\log(\lambda)$")
    axes[0].set_yscale("log")

    total_times = np.array([np.sum(t) for t in times])
    total_times_sum = np.sum(total_times)
    total_times = np.concatenate([
        total_times, 
        [total_time - total_times_sum] # unaccounted time
    ])
    if relative:
        total_times /= total_time
    axes[1].bar(
        np.arange(len(total_times)),
        total_times,
        color=colors + ["grey"],
        edgecolor=colors + ["grey"],
        linewidth=1.5,
        label=labels + ["other"],
        alpha=0.5,
    )
    axes[1].legend()
    axes[1].set_title("Time Breakdown")
    if relative:
        axes[1].set_ylabel("Proportion of Time")
    else:
        axes[1].set_ylabel("Time (s)")
    axes[1].set_xlabel("Category")

    return fig, axes


def plot_kkt(
    lmdas: np.ndarray,
    scores: np.ndarray, 
    *,
    idx: int =None,
    relative: bool =False,
):
    """Plots KKT failures.

    This function only plots a subset of the scores.
    Specifically (assuming ``idx`` is not ``None``),
    if there are ``n`` number of scores in ``scores[idx]``
    above the threshold ``lmdas[idx]``, then these ``n`` scores 
    as well as the ``n`` largest scores below the threshold are shown.
    If ``idx`` is ``None``, then the same plot is drawn at every index
    as an animation.

    Parameters
    ----------
    lmdas : (L,) ndarray
        Regularization parameters :math:`\\lambda`.
    scores : (L, G) ndarray
        Gradient scores.
    idx : int, optional
        Index of ``lmdas`` and ``scores`` at which to plot the KKT failures.
        If ``None``, then an animation of the plots at every index is shown.
        Default is ``None``.
    relative : bool, optional
        If ``True``, then plots the relative error ``score / lmda[:, None] - 1``.
        Otherwise, the absolute values are used.
        Default is ``False``.

    Returns
    -------
    fig, ax
        If ``idx`` is not ``None``, then a figure and axes is returned.
    anim
        If ``idx`` is ``None``, then an animation of the plots is returned.
    """
    G = scores.shape[-1]

    if relative:
        scores = scores / lmdas[:, None] - 1
        baseline = np.zeros(scores.shape[0])
    else:
        baseline = lmdas

    do_anim = idx is None
    idx = 0 if do_anim else idx

    gns = np.arange(G)

    colors = ["blue", "red"]
    labels = ["success", "failure"]
    alphas = [0.6, 0.8]

    fig, ax = plt.subplots(figsize=(9, 6), layout="constrained")

    is_failure = scores[idx] > baseline[idx]
    xs = [
        gns[~is_failure],
        gns[is_failure],
    ]
    ys = [
        scores[idx, ~is_failure],
        scores[idx, is_failure],
    ]
    scats = [None] * 2
    for i, (x, y, color, label, alpha) in enumerate(zip(xs, ys, colors, labels, alphas)):
        scats[i] = ax.scatter(
            x, y,
            color=color,
            marker='.',
            facecolor="None",
            label=label,
            alpha=alpha,
        )
    ax.legend()
    bound = np.maximum((np.max(scores[idx]) - baseline[idx]) * 1.05, 1e-5)
    ax.set_ylim(
        bottom=baseline[idx]-bound,
        top=baseline[idx]+bound,
    )
    ax.axhline(baseline[idx], linestyle='--', linewidth=1, color="green")
    ax.set_title("Active Score Error (Largest)")
    if relative:
        ax.set_ylabel(r"$s_g / \lambda - 1")
    else:
        ax.set_ylabel(r"$s_g$")
    ax.set_xlabel("Group Number")

    if do_anim:
        plt.close(fig)
    else:
        return fig, ax

    def update(idx):
        s = scores[idx]

        is_failure = s > baseline[idx]
        xs = [
            gns[~is_failure],
            gns[is_failure],
        ]
        ys = [
            s[~is_failure],
            s[is_failure],
        ]
        for i, (x, y, color, label, alpha) in enumerate(zip(xs, ys, colors, labels, alphas)):
            data = np.stack([x, y]).T
            scats[i].set_offsets(data)
            scats[i].set(
                color=color,
                facecolor="None",
                alpha=alpha,
                label=label,
            )
        bound = np.maximum((np.max(s) - baseline[idx]) * 1.05, 1e-5)
        ax.set_ylim(
            bottom=baseline[idx]-bound,
            top=baseline[idx]+bound,
        )
        return (scats[0], scats[1],)

    anim = animation.FuncAnimation(
        fig=fig, 
        func=update, 
        frames=lmdas.shape[0]-1, 
        interval=200, 
        repeat=False,
    )

    return HTML(anim.to_html5_video())


class DiagnosticCov:
    """Diagnostic class for covariance states.

    Parameters
    ----------
    state
        A state object of covariance type from solving group elastic net.
    """
    def __init__(self, state):
        self.state = state
        self.betas = state.betas
        self.duals = state.duals

        # keep same format as DiagnosticNaive for consistency
        self._args = {}
        constraints = state.constraints
        if np.all([c is None for c in constraints]):
            constraints = None
        self._args["constraints"] = constraints
        self._args["groups"] = state.groups
        self._args["penalty"] = state.penalty

        A, v = state.A, state.v
        L, p = self.betas.shape
        self.gradients = np.empty((L, p))
        for i in range(L):
            beta_i = self.betas[i]
            A.mul(beta_i.indices, beta_i.data, self.gradients[i]) 
        self.gradients = v[None] - self.gradients
        self.gradient_norms = gradient_norms(
            grads=self.gradients,
            betas=self.betas,
            duals=self.duals,
            lmdas=self.state.lmdas,
            constraints=self._args["constraints"],
            groups=self._args["groups"],
            alpha=self.state.alpha,
            penalty=self._args["penalty"],
        )
        self.gradient_scores = gradient_scores(
            grad_norms=self.gradient_norms,
            lmdas=self.state.lmdas,
            alpha=self.state.alpha,
            penalty=self._args["penalty"],
        )

    def plot_coefficients(self, **kwargs):
        """Plots the coefficient profile.

        See Also
        --------
        adelie.diagnostic.plot_coefficients
        """
        return plot_coefficients(
            betas=self.betas,
            lmdas=self.state.lmdas,
            groups=self.state.groups,
            group_sizes=self.state.group_sizes,
            **kwargs,
        )

    def plot_devs(self):
        """Plots the deviance profile.

        See Also
        --------
        adelie.diagnostic.plot_devs
        """
        return plot_devs(
            lmdas=self.state.lmdas,
            devs=self.state.devs,
        )

    def plot_set_sizes(self, **kwargs):
        """Plots the set sizes.

        See Also
        --------
        adelie.diagnostic.plot_set_sizes
        """
        return plot_set_sizes(
            groups=self.state.groups,
            screen_sizes=self.state.screen_sizes,
            active_sizes=self.state.active_sizes,
            lmdas=self.state.lmdas,
            screen_rule=self.state.screen_rule,
            **kwargs,
        )

    def plot_benchmark(self, **kwargs):
        """Plots benchmark times.

        See Also
        --------
        adelie.diagnostic.plot_benchmark
        """
        return plot_benchmark(
            total_time=self.state.total_time,
            benchmark_screen=self.state.benchmark_screen,
            benchmark_fit_screen=self.state.benchmark_fit_screen,
            benchmark_fit_active=self.state.benchmark_fit_active,
            benchmark_kkt=self.state.benchmark_kkt,
            benchmark_invariance=self.state.benchmark_invariance,
            n_valid_solutions=self.state.n_valid_solutions,
            lmdas=self.state.lmdas,
            **kwargs,
        )

    def plot_kkt(self, **kwargs):
        """Plots KKT failures.

        See Also
        --------
        adelie.diagnostic.plot_kkt
        """
        return plot_kkt(
            lmdas=self.state.lmdas,
            scores=self.gradient_scores,
            **kwargs,
        )
    

class DiagnosticNaive:
    """Diagnostic class for naive states.

    Parameters
    ----------
    state
        A state object of naive type from solving group elastic net.
    """
    def __init__(self, state):
        self.state = state
        self.betas = state.betas
        self.duals = state.duals
        self._n_classes = self.state._glm.y.shape[-1]
        self._is_multi = state._glm.is_multi
        self._args = {}
        if self._is_multi:
            p_begin = self.state.multi_intercept * self._n_classes
            self._args["groups"] = state.groups[p_begin:] // self._n_classes - 1
            constraints = state.constraints[p_begin:]
            if np.all([c is None for c in constraints]):
                constraints = None
            self._args["constraints"] = constraints
            self._args["penalty"] = state.penalty[p_begin:]
        else:
            constraints = state.constraints
            if np.all([c is None for c in constraints]):
                constraints = None
            self._args["constraints"] = constraints
            self._args["groups"] = state.groups
            self._args["penalty"] = state.penalty

        self.linear_preds = predict(
            X=self.state._X,
            betas=self.betas,
            intercepts=self.state.intercepts,
            offsets=self.state._offsets,
            n_threads=self.state.n_threads,
        )
        self.residuals = residuals(
            glm=self.state._glm,
            etas=self.linear_preds,
        )
        self.gradients = gradients(
            X=self.state._X,
            resids=self.residuals,
            n_threads=self.state.n_threads,
        )
        self.gradient_norms = gradient_norms(
            grads=self.gradients,
            betas=self.betas,
            duals=self.duals,
            lmdas=self.state.lmdas,
            constraints=self._args["constraints"],
            groups=self._args["groups"],
            alpha=self.state.alpha,
            penalty=self._args["penalty"],
        )
        self.gradient_scores = gradient_scores(
            grad_norms=self.gradient_norms,
            lmdas=self.state.lmdas,
            alpha=self.state.alpha,
            penalty=self._args["penalty"],
        )

    def plot_coefficients(self, **kwargs):
        """Plots the coefficient profile.

        See Also
        --------
        adelie.diagnostic.plot_coefficients
        """
        p_begin = (
            self.state.multi_intercept * self._n_classes
            if self._is_multi else
            0
        )
        return plot_coefficients(
            betas=self.betas,
            lmdas=self.state.lmdas,
            groups=self.state.groups[p_begin:]-p_begin,
            group_sizes=self.state.group_sizes[p_begin:],
            **kwargs,
        )

    def plot_devs(self):
        """Plots the deviance profile.

        See Also
        --------
        adelie.diagnostic.plot_devs
        """
        return plot_devs(
            lmdas=self.state.lmdas,
            devs=self.state.devs,
        )

    def plot_set_sizes(self, **kwargs):
        """Plots the set sizes.

        See Also
        --------
        adelie.diagnostic.plot_set_sizes
        """
        return plot_set_sizes(
            groups=self.state.groups,
            screen_sizes=self.state.screen_sizes,
            active_sizes=self.state.active_sizes,
            lmdas=self.state.lmdas,
            screen_rule=self.state.screen_rule,
            **kwargs,
        )

    def plot_benchmark(self, **kwargs):
        """Plots benchmark times.

        See Also
        --------
        adelie.diagnostic.plot_benchmark
        """
        return plot_benchmark(
            total_time=self.state.total_time,
            benchmark_screen=self.state.benchmark_screen,
            benchmark_fit_screen=self.state.benchmark_fit_screen,
            benchmark_fit_active=self.state.benchmark_fit_active,
            benchmark_kkt=self.state.benchmark_kkt,
            benchmark_invariance=self.state.benchmark_invariance,
            n_valid_solutions=self.state.n_valid_solutions,
            lmdas=self.state.lmdas,
            **kwargs,
        )

    def plot_kkt(self, **kwargs):
        """Plots KKT failures.

        See Also
        --------
        adelie.diagnostic.plot_kkt
        """
        return plot_kkt(
            lmdas=self.state.lmdas,
            scores=self.gradient_scores,
            **kwargs,
        )


def diagnostic(state):
    """Creates a diagnostic class object appropriate for the state.

    Parameters
    ----------
    state
        A state object from solving group elastic net.
    
    Returns
    -------
    dg
        Diagnostic class object.

    See Also
    --------
    adelie.diagnostic.DiagnosticCov
    adelie.diagnostic.DiagnosticNaive
    """
    if "naive" in type(state).__name__:
        return DiagnosticNaive(state)
    elif "cov" in type(state).__name__:
        return DiagnosticCov(state)
    raise TypeError("state must be one of the supported state types in adelie.")

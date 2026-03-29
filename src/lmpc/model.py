import numpy as np

from types import ModuleType
from typing import cast
from juliacall import Main as jl
from juliacall import AnyValue

jl = cast(ModuleType, jl)
jl.seval("using LinearMPC")
LinearMPC = jl.LinearMPC

# Julia wrapper factory: converts a Python callable into a Julia Function.
# The wrapper calls the Python function with the given arguments and converts
# the returned Python array into a Julia Vector{Float64}.
_jl_callable_wrapper = jl.seval(
    "py_f -> (args...) -> pyconvert(Vector{Float64}, py_f(args...))"
)


def _wrap_python_callable(f):
    return _jl_callable_wrapper(f)


# Standard step for double-precision finite-difference linearisation.
_LINEARIZATION_EPS = 1e-6


def _linearize_python_callables(f, h, xo, uo, d):
    """Finite-difference linearisation of Python callables f and h at (xo, uo).

    LinearMPC.jl's ``linearize`` uses ForwardDiff, which cannot propagate
    dual numbers through Python callbacks.  This function computes the same
    Jacobians in Python using forward finite differences, then delegates
    model construction to the Julia ``LinearMPC.Model`` constructor.

    Parameters
    ----------
    f  : callable, (x, u, d) -> array-like — next-state (DT) or ẋ (CT)
    h  : callable, (x, u, d) -> array-like — output
    xo : ndarray — operating-point state
    uo : ndarray — operating-point input
    d  : ndarray — operating-point disturbance (may be empty)

    Returns
    -------
    A, B, Bd, f_offset, C, Dd, h_offset : ndarrays
        Jacobian matrices and affine offsets for the linearised model.
    """
    eps = _LINEARIZATION_EPS
    nx, nu, nd = len(xo), len(uo), len(d)

    f0 = np.asarray(f(xo, uo, d), dtype=float)
    h0 = np.asarray(h(xo, uo, d), dtype=float)
    ny = len(h0)

    A = np.zeros((nx, nx))
    for i in range(nx):
        xp = xo.copy(); xp[i] += eps
        A[:, i] = (np.asarray(f(xp, uo, d), dtype=float) - f0) / eps

    B = np.zeros((nx, nu))
    for i in range(nu):
        up = uo.copy(); up[i] += eps
        B[:, i] = (np.asarray(f(xo, up, d), dtype=float) - f0) / eps

    Bd = np.zeros((nx, nd))
    for i in range(nd):
        dp = d.copy(); dp[i] += eps
        Bd[:, i] = (np.asarray(f(xo, uo, dp), dtype=float) - f0) / eps

    f_offset = f0 - A @ xo - B @ uo - Bd @ d

    C = np.zeros((ny, nx))
    for i in range(nx):
        xp = xo.copy(); xp[i] += eps
        C[:, i] = (np.asarray(h(xp, uo, d), dtype=float) - h0) / eps

    Dd = np.zeros((ny, nd))
    for i in range(nd):
        dp = d.copy(); dp[i] += eps
        Dd[:, i] = (np.asarray(h(xo, uo, dp), dtype=float) - h0) / eps

    h_offset = h0 - C @ xo - Dd @ d

    return A, B, Bd, f_offset, C, Dd, h_offset


class Model:
    """Python wrapper for Julia LinearMPC.Model.

    Supports three construction modes:

    **Nonlinear dynamics** (Python callables)::

        model = Model(f, h, xo, uo)        # discrete-time
        model = Model(f, h, xo, uo, Ts)    # continuous-time (ZOH discretised)

    where ``f(x, u, d) -> array`` gives the next state (DT) or the state
    derivative (CT), and ``h(x, u, d) -> array`` gives the output.
    Linearisation is performed numerically (finite differences) at the
    operating point ``(xo, uo)``.

    **Linear discrete-time matrices**::

        model = Model(F, G)
        model = Model(F, G, Gd=Gd, C=C, Dd=Dd, Ts=Ts)

    **Linear continuous-time matrices** (ZOH discretised)::

        model = Model(A, B, Ts)
        model = Model(A, B, Ts, Bd=Bd, C=C, Dd=Dd)
    """
    jl_model: AnyValue

    def __init__(self, F_or_f, G_or_h=None, Ts_or_xo=None, uo=None, Ts=None,
                 Bd=None, Gd=None, C=np.zeros([0, 0]), Dd=np.zeros([0, 0]),
                 f_offset=np.zeros(0), h_offset=np.zeros(0),
                 xo=np.zeros(0), d=None):
        if callable(F_or_f):
            # --- Nonlinear: Model(f, h, xo, uo[, Ts]) ---
            # LinearMPC.jl's linearize() uses ForwardDiff and cannot propagate
            # dual numbers through Python callbacks, so we linearise via finite
            # differences in Python and then call LinearMPC.Model with the
            # resulting matrices and the original callables as true_dynamics/true_h.
            f, h = F_or_f, G_or_h
            xo_val = np.asarray(Ts_or_xo, dtype=float)
            uo_val = np.asarray(uo, dtype=float)
            d_val  = np.zeros(0) if d is None else np.asarray(d, dtype=float)

            A_lin, B_lin, Bd_lin, f_off, C_lin, Dd_lin, h_off = \
                _linearize_python_callables(f, h, xo_val, uo_val, d_val)

            # Wrap the Python callables so Julia can call them during simulation.
            jl_f = _wrap_python_callable(f)
            jl_h = _wrap_python_callable(h)

            if Ts is not None:
                # Continuous-time: pass CT Jacobians; Julia applies ZOH to
                # produce the DT model matrices and stores jl_f as true_dynamics.
                self.jl_model = LinearMPC.Model(
                    A_lin, B_lin, float(Ts),
                    Bd=Bd_lin, C=C_lin, Dd=Dd_lin,
                    f_offset=f_off, h_offset=h_off,
                    xo=xo_val, uo=uo_val,
                    true_dynamics=jl_f, true_h=jl_h)
            else:
                # Discrete-time: Jacobians ARE the DT system matrices.
                self.jl_model = LinearMPC.Model(
                    A_lin, B_lin,
                    Gd=Bd_lin, C=C_lin, Dd=Dd_lin,
                    f_offset=f_off, h_offset=h_off,
                    xo=xo_val, uo=uo_val,
                    true_dynamics=jl_f, true_h=jl_h)
        elif Ts_or_xo is not None and np.isscalar(Ts_or_xo):
            # --- CT linear: Model(A, B, Ts, Bd=..., C=..., Dd=...) ---
            Bd_val = np.zeros([0, 0]) if Bd is None else np.asarray(Bd, dtype=float)
            self.jl_model = LinearMPC.Model(
                np.asarray(F_or_f, dtype=float),
                np.asarray(G_or_h, dtype=float),
                float(Ts_or_xo),
                Bd=Bd_val, C=C, Dd=Dd,
                f_offset=f_offset, h_offset=h_offset, xo=xo)
        else:
            # --- DT linear: Model(F, G, Gd=..., C=..., Dd=..., Ts=...) ---
            Gd_val = np.zeros([0, 0]) if Gd is None else np.asarray(Gd, dtype=float)
            kw = dict(Gd=Gd_val, C=C, Dd=Dd,
                      f_offset=f_offset, h_offset=h_offset, xo=xo)
            if Ts is not None:
                kw['Ts'] = float(Ts)
            self.jl_model = LinearMPC.Model(
                np.asarray(F_or_f, dtype=float),
                np.asarray(G_or_h, dtype=float),
                **kw)

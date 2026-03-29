import numpy as np

from types import ModuleType
from typing import cast
from juliacall import Main as jl
from juliacall import AnyValue

jl = cast(ModuleType, jl)
jl_version = (jl.VERSION.major, jl.VERSION.minor, jl.VERSION.patch)

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


class ParameterRange:
    """Python wrapper for Julia LinearMPC.ParameterRange.

    Instances are returned by :meth:`MPC.range` and by :func:`mpc_examples`,
    and can be passed directly to :class:`ExplicitMPC` and :meth:`MPC.certify`.
    Fields (``xmin``, ``xmax``, ``rmin``, ``rmax``, ``dmin``, ``dmax``,
    ``umin``, ``umax``, ``lmin``, ``lmax``) are Julia arrays that support
    in-place slice assignment, e.g. ``pr.xmin[:] = [-5, -5]``.
    """
    jl_range: AnyValue

    def __init__(self, jl_range):
        self.jl_range = jl_range

    @property
    def xmin(self): return self.jl_range.xmin
    @property
    def xmax(self): return self.jl_range.xmax
    @property
    def rmin(self): return self.jl_range.rmin
    @property
    def rmax(self): return self.jl_range.rmax
    @property
    def dmin(self): return self.jl_range.dmin
    @property
    def dmax(self): return self.jl_range.dmax
    @property
    def umin(self): return self.jl_range.umin
    @property
    def umax(self): return self.jl_range.umax
    @property
    def lmin(self): return self.jl_range.lmin
    @property
    def lmax(self): return self.jl_range.lmax


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


class MPC:
    jl_mpc:AnyValue
    def __init__(self, A_or_model, B=None, Ts=None, Bd=None, Gd=None,
                 C=np.zeros([0,0]), Dd=np.zeros([0,0]), Np=10, Nc=None):
        if Nc is None: Nc = Np
        if isinstance(A_or_model, Model):
            self.jl_mpc = LinearMPC.MPC(A_or_model.jl_model, Np=Np, Nc=Nc)
        elif Ts is None or (Gd is not None and Bd is None):# discrete-time system
            if Gd is None: Gd = np.zeros([0,0]) 
            self.jl_mpc = LinearMPC.MPC(A_or_model,B,Gd=Gd,C=C,Dd=Dd,Np=Np,Nc=Nc)
        else:
            if Bd is None: Bd = np.zeros([0,0]) 
            self.jl_mpc = LinearMPC.MPC(A_or_model,B,Ts,Bd=Bd,C=C,Dd=Dd,Np=Np,Nc=Nc)

    def compute_control(self,x,r=None, d=None, uprev=None, l=None):
        return  LinearMPC.compute_control(self.jl_mpc, x, r = r, d=d, uprev=uprev, l=l)
    
    # Setting up problem 
    def setup(self):
        LinearMPC.setup_b(self.jl_mpc)

    def set_bounds(self, umin=np.zeros(0), umax=np.zeros(0), ymin=np.zeros(0), ymax=np.zeros(0)):
        LinearMPC.set_bounds_b(self.jl_mpc, umin = umin, umax = umax, ymin = ymin, ymax = ymax)

    def set_input_bounds(self, umin=np.zeros(0), umax=np.zeros(0)):
        LinearMPC.set_input_bounds_b(self.jl_mpc, umin = umin, umax = umax)

    def add_constraint(self,Ax = None, Au= None, 
                        Ar = np.zeros((0,0)), Aw = np.zeros((0,0)), 
                        Ad = np.zeros((0,0)), Aup = np.zeros((0,0)),
                        ub = np.zeros(0), lb = np.zeros(0),
                        ks = None, soft=False, binary=False, prio = 0):
        ks = range(2,self.jl_mpc.Np+1) if ks is None else [k+1 for k in ks]
        LinearMPC.add_constraint_b(self.jl_mpc, Ax=Ax, Au=Au, Ar=Ar, Ad=Ad, Aup=Aup, ub=ub, lb=lb,
                                 ks=ks, soft=soft, binary=binary, prio = prio)

    def set_output_bounds(self, ymin=np.zeros(0), ymax=np.zeros(0), ks =None, soft = True, binary=False, prio = 0):
        ks = range(2,self.jl_mpc.Np+1) if ks is None else [k+1 for k in ks]
        LinearMPC.set_output_bounds_b(self.jl_mpc,ymin=ymin,ymax=ymax, 
                                    ks=ks, soft=soft, binary=binary, prio=prio)

    def set_objective(self, uids=None, Q=None, R=None ,Rr=None, S=None, Qf=None, Qfx=None):
        Q  = np.zeros((0,0)) if Q  is None else Q if np.isscalar(Q) else np.array(Q)
        R  = np.zeros((0,0)) if R  is None else R if np.isscalar(R) else np.array(R)
        Rr = np.zeros((0,0)) if Rr is None else Rr if np.isscalar(Rr) else np.array(Rr)
        S  = np.zeros((0,0)) if S  is None else np.array(S)
        Qf = np.zeros((0,0)) if Qf is None else np.array(Qf)
        Qfx = np.zeros((0,0)) if Qfx is None else np.array(Qfx)

        if uids is None: 
            LinearMPC.set_weights_b(self.jl_mpc, Q=Q, R=R, Rr=Rr,S=S,Qf=Qf)
        else:
            LinearMPC.set_weights_b(self.jl_mpc, jl.Vector[jl.Int](uids), 
                                    Q=Q, R=R, Rr=Rr,S=S,Qf=Qf)

    def set_terminal_cost(self):
        LinearMPC.set_terminal_cost_b(self.jl_mpc)

    def set_prestabilizing_feedback(self, K=None):
        if K is not None:
            LinearMPC.set_prestabilizing_feedback_b(self.jl_mpc,K)
        else:
            LinearMPC.set_prestabilizing_feedback_b(self.jl_mpc)

    def move_block(self,move):
        if not isinstance(move,list):
            LinearMPC.move_block_b(self.jl_mpc,move)
        else:
            if any(isinstance(i, list) for i in move):
                jl_move = jl.Vector[jl.Vector[jl.Int]]([jl.Vector[jl.Int](mb) for mb in move])
            else:
                jl_move = jl.Vector[jl.Int](move)
            LinearMPC.move_block_b(self.jl_mpc,jl_move)

    def set_horizon(self,Np):
        LinearMPC.set_horizon_b(self.jl_mpc,Np)

    def set_binary_controls(self,bin_ids):
        LinearMPC.set_binary_controls_b(self.jl_mpc,bin_ids)

    def set_disturbance(self,wmin,wmax):
        LinearMPC.set_disturbance_b(self.jl_mpc,wmin,wmax)
 
    def set_x0_uncertainty(self,x0_uncertainty):
        LinearMPC.set_x0_uncertainty_b(self.jl_mpc,x0_uncertainty)

    def settings(self,settings):
        LinearMPC.settings_b(self.jl_mpc, settings)

    def set_state_observer(self,F=None,G=None,Gd=None,C=None,Dd=None,
                           f_offset=None,h_offset=None,Q=None,R=None,x0=None):
        LinearMPC.set_state_observer_b(self.jl_mpc, F=F,G=G,Gd=Gd,C=C,Dd=Dd,
                                       f_offset=f_offset,h_offset=h_offset,
                                       Q=Q,R=R,x0=x0)
    def get_state(self):
        return LinearMPC.get_state(self.jl_mpc)
    def set_state(self,x0):
        return LinearMPC.set_state_b(self.jl_mpc,x0)
    def correct_state(self,y):
        return LinearMPC.correct_state_b(self.jl_mpc,y)
    def predict_state(self,u):
        return LinearMPC.predict_state_b(self.jl_mpc,u)


    def set_operating_point(self,xo=None,uo=None,relinearize=True):
        LinearMPC.set_operating_point_b(self.jl_mpc,xo=xo,uo=uo,relinearize=relinearize)

    def set_offest(self,xo=None,uo=None,doff=None,fo=None,ho=None):
        xo   = np.zeros(0)  if xo   is None else np.array(xo)
        uo   = np.zeros(0)  if uo   is None else np.array(uo)
        doff = np.zeros(0)  if doff is None else np.array(doff)
        fo   = np.zeros(0)  if fo   is None else np.array(fo)
        ho   = np.zeros(0)  if ho   is None else np.array(ho)
        LinearMPC.set_operating_point_b(self,xo=xo,uo=uo,doff=doff,fo=fo,ho=ho)

    # code generation 
    def codegen(self, fname="mpc_workspace", dir="codegen", opt_settings=None, src=True, float_type="double"):
        LinearMPC.codegen(self.jl_mpc,fname=fname,dir=dir,opt_settings=opt_settings,src=src,float_type=float_type)

    # certification
    def certify(self, range=None, AS0=[], single_soft=True):
        jl_range = range.jl_range if isinstance(range, ParameterRange) else range
        return CertificationResult(LinearMPC.certify(self.jl_mpc, range=jl_range,
                                                     AS0=np.asarray(AS0,dtype=int),
                                                     single_soft=single_soft))

    def range(self, xmin=None,xmax=None,rmin=None,rmax=None,dmin=None,dmax=None,umin=None,umax=None):
        jl_range = LinearMPC.ParameterRange(self.jl_mpc)
        if xmin is not None: jl_range.xmin[:] = xmin
        if xmax is not None: jl_range.xmax[:] = xmax
        if rmin is not None: jl_range.rmin[:] = rmin
        if rmax is not None: jl_range.rmax[:] = rmax
        if dmin is not None: jl_range.dmin[:] = dmin
        if dmax is not None: jl_range.dmax[:] = dmax
        if umin is not None: jl_range.umin[:] = umin
        if umax is not None: jl_range.umax[:] = umax
        return ParameterRange(jl_range)

    def mpqp(self, singlesided = False, single_soft = True):
        mpqp_jl = LinearMPC.mpc2mpqp(self.jl_mpc)
        if singlesided:
            mpqp_jl = LinearMPC.make_singlesided(mpqp_jl,single_soft=single_soft)
        mpqp = {
                "jl_src" : mpqp_jl,
                "H": np.array(mpqp_jl.H,copy=False, order='F'),
                "f": np.array(mpqp_jl.f,copy=False, order='F'),
                "f_theta": np.array(mpqp_jl.f_theta,copy=False, order='F'),
                "H_theta": np.array(mpqp_jl.H_theta,copy=False, order='F'),
                "A": np.array(mpqp_jl.A,copy=False, order='F'),
                "W": np.array(mpqp_jl.W,copy=False, order='F'),
                "senses": np.array(mpqp_jl.senses,copy=False, order='F'),
                "prio": np.array(mpqp_jl.senses,copy=False, order='F'),
                "has_binaries": mpqp_jl.has_binaries
                #"break_points": np.array(mpqp_jl.break_points,copy=False, order='F')
                }
        if singlesided:
            mpqp["b"] = np.array(mpqp_jl.b,copy=False, order='F')
        else:
            mpqp["bu"] = np.array(mpqp_jl.bu,copy=False, order='F')
            mpqp["bl"] = np.array(mpqp_jl.bl,copy=False, order='F')
        return mpqp

# Explicit MPC
class ExplicitMPC:
    jl_mpc:AnyValue
    def __init__(self,mpc,range=None,build_tree=False):
        jl_range = range.jl_range if isinstance(range, ParameterRange) else range
        self.jl_mpc = LinearMPC.ExplicitMPC(mpc.jl_mpc,range=jl_range,build_tree=build_tree)

    def codegen(self,fname="empc",dir="codegen", opt_settings=None, src=True, 
                float_type="double"):
        LinearMPC.codegen(self.jl_mpc,fname=fname,dir=dir,opt_settings=opt_settings,src=src,float_type=float_type)

# Certification 
class CertificationResult:
    jl_result:AnyValue
    def __init__(self,result):
        self.jl_result = result

class Simulation:
    jl_sim:AnyValue
    ts: np.ndarray
    ys: np.ndarray
    us: np.ndarray
    xs: np.ndarray
    rs: np.ndarray
    ds: np.ndarray
    xhats: np.ndarray
    yms: np.ndarray
    solve_times: np.ndarray

    def __init__(self,mpc,f=None,x0= None,N=1000, r=None,d=None,l=None):
        if x0 is None: x0 = np.zeros(mpc.jl_mpc.model.nx)
        if f  is None: 
            self.jl_sim = LinearMPC.Simulation(mpc.jl_mpc,x0=x0,N=N,r=r,d=d,l=l)
        else:
            jl_f = _wrap_python_callable(f) if callable(f) else f
            self.jl_sim = LinearMPC.Simulation(jl_f,mpc.jl_mpc,x0=x0,N=N,r=r,d=d,l=l)

        self.ts = np.array(self.jl_sim.ts,copy=False, order='F')
        self.ys = np.array(self.jl_sim.ys,copy=False, order='F')
        self.us = np.array(self.jl_sim.us,copy=False, order='F')
        self.xs = np.array(self.jl_sim.xs,copy=False, order='F')
        self.rs = np.array(self.jl_sim.rs,copy=False, order='F')
        self.ds = np.array(self.jl_sim.ds,copy=False, order='F')
        self.xhats = np.array(self.jl_sim.xhats,copy=False, order='F')
        self.yms = np.array(self.jl_sim.yms,copy=False, order='F')
        self.solve_times= np.array(self.jl_sim.solve_times,copy=False, order='F')


def mpc_examples(name, Np=None, Nc=None, params=None, settings=None):
    """Load a named MPC example, returning a Python ``(MPC, ParameterRange)`` tuple.

    Wraps ``LinearMPC.mpc_examples`` so that results are ready-to-use Python
    objects.  All keyword arguments are forwarded to the Julia function.

    Parameters
    ----------
    name : str
        Example name (e.g. ``"invpend"``, ``"aircraft"``, ``"dcmotor"``).
    Np : int, optional
        Prediction horizon.  Uses the Julia default when omitted.
    Nc : int, optional
        Control horizon.  Defaults to *Np* when omitted.
    params : dict, optional
        Extra parameters forwarded to Julia (e.g. ``{"nx": 2}``).
    settings : optional
        Julia ``MPCSettings`` object forwarded verbatim.

    Returns
    -------
    mpc : MPC
    parameter_range : ParameterRange
    """
    kwargs = {}
    if settings is not None:
        kwargs['settings'] = settings
    if params is not None:
        kwargs['params'] = params

    if Np is not None and Nc is not None:
        jl_mpc, jl_range = LinearMPC.mpc_examples(name, Np, Nc, **kwargs)
    elif Np is not None:
        jl_mpc, jl_range = LinearMPC.mpc_examples(name, Np, **kwargs)
    else:
        jl_mpc, jl_range = LinearMPC.mpc_examples(name, **kwargs)

    mpc = MPC.__new__(MPC)
    mpc.jl_mpc = jl_mpc
    return mpc, ParameterRange(jl_range)

import numpy as np

from types import ModuleType
from typing import cast
from juliacall import Main as jl
from juliacall import AnyValue

jl = cast(ModuleType, jl)
jl_version = (jl.VERSION.major, jl.VERSION.minor, jl.VERSION.patch)

jl.seval("using LinearMPC")
LinearMPC = jl.LinearMPC


class MPC:
    jl_mpc:AnyValue
    def __init__(self,F,G,Gd=None,C=None,Dd=None,Ts=1, Np=10, Nc=None):
        if Nc is None: Nc = Np
        self.jl_mpc = LinearMPC.MPC(F,G,Gd=Gd,C=C,Dd=Dd,Ts=Ts,Np=Np,Nc=Nc)
    def __init__(self,A,B,Ts,Bd=None,C=None,Dd=None,Np=10, Nc=None):
        if Nc is None: Nc = Np
        self.jl_mpc = LinearMPC.MPC(A,B,Ts,Bd=Bd,C=C,Dd=Dd,Np=Np,Nc=Nc)

    def compute_control(self,x,r=None, d=None, uprev=None):
        return  LinearMPC.compute_control(self.jl_mpc, x, r = r, d=d, uprev=uprev)
    
    # Setting up problem 
    def setup(self):
        LinearMPC.setup_b(self.jl_mpc)

    def set_bounds(self, umin=np.zeros(0), umax=np.zeros(0)):
        LinearMPC.set_bounds_b(self.jl_mpc, umin = umin, umax = umax)

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

    def set_weights(self, Q=None, R=None ,Rr=None, S=None, rho=None, Qf=None):
        if Q is not None: Q = np.array(Q)
        if R is not None: R = np.array(R)
        if Rr is not None: Rr = np.array(Rr)
        if Qf is not None: Qf = np.array(Qf)
        LinearMPC.set_weights_b(self.jl_mpc, Q=Q, R=R, Rr=Rr,S=S, rho=rho, Qf=Qf)

    def set_terminal_cost(self):
        LinearMPC.set_terminal_cost(self.jl_mpc)

    def set_prestabilizing_feedback(self, K=None):
        if K is not None:
            LinearMPC.set_prestabilizing_feedback_b(self.jl_mpc,K)
        else:
            LinearMPC.set_prestabilizing_feedback_b(self.jl_mpc)

    def move_block(self,move):
        LinearMPC.move_block_b(self.jl_mpc,move)

    # code generation 
    def codegen(self, fname="mpc_workspace", dir="codegen", opt_settings=None, src=True, float_type="double"):
        LinearMPC.codegen(self.jl_mpc,fname=fname,dir=dir,opt_settings=opt_settings,src=src,float_type=float_type)

# Explicit MPC
class ExplicitMPC:
    jl_empc:AnyValue
    def __init__(self,mpc,range=None,build_tree=False):
        self.jl_empc = LinearMPC.ExplicitMPC(mpc.jl_mpc,range=range,build_tree=build_tree)

    def plot_regions(self,th1,th2,x=None,r=None,d=None,uprev=None, show_fixed=True,show_zero=False):
        jl.display(LinearMPC.plot_regions(self.jl_empc,th1,th2,x=x,r=r,d=d,uprev=uprev,
                                          show_fixed=show_fixed,show_zero=show_zero))

    def plot_feedback(self,u,th1,th2,x=None,r=None,d=None,uprev=None,show_fixed=True, show_zero=False):
        jl.display(LinearMPC.plot_feedback(self.jl_empc,u,th1,th2,x=x,r=r,d=d,uprev=uprev,
                                           show_fixed=show_fixed,show_zero=show_zero))

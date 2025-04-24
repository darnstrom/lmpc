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
    def __init__(self,F,G):
        LinearMPC.MPC(F,G)

    def compute_control(self,x,r=None, d=None, uprev=None):
        return  ParametricDAQP.compute_control(self.jl_mpc,x, r = r, d=d, uprev=uprev)

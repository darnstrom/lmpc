import numpy
from lmpc import MPC,ExplicitMPC

A =  numpy.array([[0.0,1], [0,0]])
B = numpy.array([[0.0],[1]])

# Set MPC
mpc = MPC(A,B,0.1,Np=2)

# Constraints
mpc.set_bounds(umin=[-1.0],umax=[3.0])
mpc.set_output_bounds(ymin=[-1.0,-2],ymax=[3.0,4])

# Objective
mpc.set_weights(Q=[1.0,10000.0],R=[1000.0],Rr=[1.0])

# Compute control
mpc.compute_control([1,1])

# Setup
mpc.codegen()

# Explicit MPC
empc = ExplicitMPC(mpc)
empc.plot_regions("x1","x2")
empc.plot_feedback("u1","x1","x2")

# Certification
parameters = mpc.range(xmin=[-5,-5],xmax=[5,5],rmin=[-1,-1],rmax=[1,1])
result = mpc.certify(range=parameters)

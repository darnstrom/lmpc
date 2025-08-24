import numpy
from lmpc import MPC,ExplicitMPC,Simulation

A =  numpy.array([[0.0,1], [0,0]])
B = numpy.array([[0.0],[1]])

# Set MPC
mpc = MPC(A,B,0.1,Np=2)

# Constraints
mpc.set_bounds(umin=[-1.0],umax=[3.0])
mpc.set_output_bounds(ymin=[-1.0,-2],ymax=[3.0,4])

# Objective
mpc.set_objective(Q=[1.0,10000.0],R=[1000.0],Rr=[1.0])

# Compute control
mpc.compute_control([1,1])

# Generate mpQP
print(mpc.mpqp())
print(mpc.mpqp(singlesided=True))
# Setup
mpc.codegen()

# Explicit MPC
parameters = mpc.range(xmin=[-5,-5],xmax=[5,5],rmin=[-1,-1],rmax=[1,1])
empc = ExplicitMPC(mpc,range=parameters)
empc.plot_regions("x1","x2")
empc.plot_feedback("u1","x1","x2")
empc.codegen()

# Certification
result = mpc.certify(range=parameters)

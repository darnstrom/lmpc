import numpy as np
from lmpc import MPC,Simulation
F,G = np.array([[1, 0.1],[0, 1]]), np.array([[0.005], [0.1]])

mpc = MPC(F,G,Np=25,C=np.array([[1, 0],]))
mpc.set_state_observer(Q=np.eye(2),R=np.eye(1))
#mpc.set_prestabilizing_feedback()
#mpc.set_disturbance([-0.005,-0.005],[0.005,0.005])
mpc.set_bounds(umin=[-0.2],umax=[0.2],ymin=[-0.5],ymax=[0.5]) 
#mpc.set_bounds(umin=[-0.2],umax=[0.2]) 


sim = Simulation(mpc,r=[0.5])


import matplotlib.pyplot as plt

plt.plot(sim.ts,sim.ys[0,:])
print(sim.ys[0,-1])
plt.show()

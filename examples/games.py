import numpy as np
from lmpc import MPC,Simulation

## Setting up problem matrices (directly taken from the nashopt example)
np.random.seed(1)
nx,nu,ny = 6,3,3
A = np.random.rand(nx, nx)
A = A / max(abs(np.linalg.eigvals(A)))*0.95
B = np.random.randn(nx, nu)
C = np.random.randn(ny, nx)
DC = C@np.linalg.inv(np.eye(nx)-A)@B
C = np.linalg.inv(DC)@C  # scale C to have DC gain = I

## ==== Actually setting up the problem with LinearMPC.jl ====
mpc = MPC(A,B,C=C,Np=10)
mpc.set_objective([1], Q=[1,0,0], Rr=[0.5]) # First argument is control ids for a player
mpc.set_objective([2], Q=[0,1,0], Rr=[0.5])
mpc.set_objective([3], Q=[0,0,1], Rr=[0.5])

mpc.set_bounds(umin=np.zeros(nu),umax= 4*np.ones(nu))

## Simulating 40 steps with a reference of y1 = 1, y2 = 2, y3 = 3
sim = Simulation(mpc,x0=np.zeros(nx),r=[1,2,3],N=40)

## Plotting stuff similar to nashopt example (not part of LinearMPC.jl)
import matplotlib.pyplot as plt
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
fig, ax = plt.subplots(2, 1, figsize=(8, 8))

for i in range(ny):
    ax[0].plot(sim.ts, sim.ys[i,:], color=colors[i], linewidth=4, label=f'$y_{i+1}$')
    ax[0].plot(sim.ts, sim.rs[i,:], '--', color=colors[i], linewidth=2)
ax[0].grid()

for i in range(nu):
    ax[1].step(sim.ts, sim.us[i,:], linewidth=2,color=colors[i], label=f'$u_{i+1}$')
ax[1].grid()
ax[1].set_xlabel('Time step')
plt.show()

"""
Pytest tests for the lmpc Python package, ported from the Julia test suite at:
https://github.com/darnstrom/LinearMPC.jl/blob/main/test/runtests.jl
"""

import numpy as np
import pytest
from lmpc import MPC, ExplicitMPC, Simulation, Model, ParameterRange, mpc_examples
from juliacall import Main as jl

# Access the underlying Julia LinearMPC module (already loaded by the lmpc import).
LinearMPC = jl.LinearMPC


# ---------------------------------------------------------------------------
# Helper: wrap a Julia MPC object returned by mpc_examples into a Python MPC
# ---------------------------------------------------------------------------
def _python_mpc(jl_mpc):
    """Return a Python MPC shell wrapping an existing Julia MPC object."""
    mpc = MPC.__new__(MPC)
    mpc.jl_mpc = jl_mpc
    return mpc


# ---------------------------------------------------------------------------
# Basic setup
# ---------------------------------------------------------------------------
class TestBasicSetup:
    def test_basic_setup(self):
        """Create an MPC and exercise most configuration helpers."""
        rng = np.random.default_rng(1234)
        A = rng.standard_normal((3, 3))
        B = rng.standard_normal((3, 1))
        C = np.array([[1.0, 0, 0], [0, 1.0, 0]])
        Bd = rng.standard_normal((3, 1))
        Dd = np.eye(2)

        mpc = MPC(A, B, 0.1, C=C, Bd=Bd, Dd=Dd, Np=10, Nc=5)
        mpc.set_objective(Q=[1.0, 3.0], R=np.array([[2.0]]), Rr=np.array([[1.0]]))
        mpc.set_bounds(umin=[-0.5], umax=[0.5])
        mpc.set_prestabilizing_feedback()
        mpc.set_output_bounds(ymin=[0.0, 0.0], ymax=[5.0, 1.0])
        mpc.setup()
        mpc.set_horizon(5)
        mpc.setup()
        mpc.settings({"reference_tracking": False})


# ---------------------------------------------------------------------------
# MPC examples
# ---------------------------------------------------------------------------
class TestMPCExamples:
    @pytest.mark.parametrize("name", ["invpend", "dcmotor", "aircraft", "nonlin",
                                       "satellite", "ballplate"])
    def test_mpc_examples_standard(self, name):
        """Each named example can be loaded and converted to an mpQP."""
        jl_mpc, _ = LinearMPC.mpc_examples(name)
        LinearMPC.mpc2mpqp(jl_mpc)

    def test_mpc_examples_chained(self):
        jl_mpc, _ = LinearMPC.mpc_examples("chained", 10, 10, params={"nx": 2})
        LinearMPC.mpc2mpqp(jl_mpc)

    def test_mpc_examples_invpend_contact(self):
        jl_mpc, _ = LinearMPC.mpc_examples("invpend_contact", 6, 6, params={"nwalls": 1})
        LinearMPC.mpc2mpqp(jl_mpc)


# ---------------------------------------------------------------------------
# Compute control
# ---------------------------------------------------------------------------
class TestComputeControl:
    def test_invpend_control_value(self):
        """compute_control on the inverted-pendulum example matches the reference."""
        jl_mpc, _ = LinearMPC.mpc_examples("invpend")
        u = LinearMPC.compute_control(jl_mpc, [5.0, 5, 0, 0])
        assert abs(float(u[0]) - 1.7612519326) < 1e-6


# ---------------------------------------------------------------------------
# Prestabilizing feedback
# ---------------------------------------------------------------------------
class TestPrestabilizingFeedback:
    def test_control_unchanged_better_conditioning(self):
        """
        Adding a prestabilizing feedback should not change the optimal control
        but should improve the conditioning of the Hessian.
        """
        A = np.array([[0.0, 1], [10, 0]])
        B = np.array([[0.0], [1]])
        mpc = MPC(A, B, 0.1, Np=30)
        mpc.set_bounds(umin=[-1], umax=[1])

        u_nom = mpc.compute_control(np.zeros(2), r=np.array([1.0, 0]))
        cond_nom = np.linalg.cond(mpc.mpqp()["H"])

        mpc.set_prestabilizing_feedback()

        u_prestab = mpc.compute_control(np.zeros(2), r=np.array([1.0, 0]))
        cond_prestab = np.linalg.cond(mpc.mpqp()["H"])

        assert np.linalg.norm(np.array(u_nom) - np.array(u_prestab)) < 1e-10
        assert cond_prestab < cond_nom


# ---------------------------------------------------------------------------
# Move blocking
# ---------------------------------------------------------------------------
class TestMoveBlocking:
    def _aircraft_mpc(self, Np=10):
        jl_mpc, _ = LinearMPC.mpc_examples("aircraft", Np)
        return _python_mpc(jl_mpc)

    def test_empty_blocking(self):
        Np = 10
        mpc = self._aircraft_mpc(Np)
        nu = int(mpc.jl_mpc.model.nu)
        mpc.move_block([])
        mpqp = mpc.mpqp()
        assert len(mpqp["f"]) == Np * nu

    def test_explicit_blocks(self):
        Np = 10
        mpc = self._aircraft_mpc(Np)
        nu = int(mpc.jl_mpc.model.nu)
        mpc.move_block([1, 1, 2, 3, 3])
        mpqp = mpc.mpqp()
        assert len(mpqp["f"]) == 5 * nu

    def test_pad_blocks(self):
        """Short block list is padded to fill the horizon."""
        mpc = self._aircraft_mpc()
        mpc.move_block([1, 1])
        mpc.mpqp()
        blocks = [list(b) for b in mpc.jl_mpc.move_blocks]
        assert blocks == [[1, 9], [1, 9]]

    def test_clip_blocks(self):
        """Block list that overshoots the horizon is clipped."""
        mpc = self._aircraft_mpc()
        mpc.move_block([2, 3, 3, 6, 8, 9])
        mpc.mpqp()
        blocks = [list(b) for b in mpc.jl_mpc.move_blocks]
        assert blocks == [[2, 3, 3, 2], [2, 3, 3, 2]]

    def test_scalar_block(self):
        mpc = self._aircraft_mpc()
        mpc.move_block(2)
        mpc.mpqp()
        blocks = [list(b) for b in mpc.jl_mpc.move_blocks]
        assert blocks == [[2, 2, 2, 2, 2], [2, 2, 2, 2, 2]]

    def test_scalar_block_non_divisible(self):
        mpc = self._aircraft_mpc()
        mpc.move_block(3)
        mpc.mpqp()
        blocks = [list(b) for b in mpc.jl_mpc.move_blocks]
        assert blocks == [[3, 3, 3, 1], [3, 3, 3, 1]]

    def test_nested_block_list(self):
        mpc = self._aircraft_mpc()
        mpc.move_block([[1, 2, 3], [4, 2]])
        mpc.mpqp()
        blocks = [list(b) for b in mpc.jl_mpc.move_blocks]
        assert blocks == [[1, 2, 7], [4, 6]]

    def test_nested_block_list_clip(self):
        mpc = self._aircraft_mpc()
        mpc.move_block([[1, 2, 3, 15, 20], [2]])
        mpc.mpqp()
        blocks = [list(b) for b in mpc.jl_mpc.move_blocks]
        assert blocks == [[1, 2, 3, 4], [10]]


# ---------------------------------------------------------------------------
# Explicit MPC
# ---------------------------------------------------------------------------
class TestExplicitMPC:
    def test_explicit_mpc_invpend(self):
        """ExplicitMPC gives the same control as the implicit solver."""
        jl_mpc, range_ = LinearMPC.mpc_examples("invpend")
        mpc = _python_mpc(jl_mpc)
        empc = ExplicitMPC(mpc, range=range_)
        LinearMPC.build_tree_b(empc.jl_mpc)
        u = LinearMPC.compute_control(empc.jl_mpc, [5.0, 5, 0, 0])
        assert abs(float(u[0]) - 1.7612519326) < 1e-6


# ---------------------------------------------------------------------------
# Invariant sets
# ---------------------------------------------------------------------------
class TestInvariantSets:
    def test_stable_system(self):
        """BBM17 Example 10.12 – stable system with additive disturbance."""
        F = jl.seval("[0.5 0 ; 1 -0.5]")
        xmin = jl.seval("-10*ones(2)")
        xmax = jl.seval("10*ones(2)")
        wmin = jl.seval("-ones(2)")
        wmax = jl.seval("ones(2)")
        _, h = LinearMPC.invariant_set(F, xmin, xmax, wmin=wmin, wmax=wmax, eps_shrink=0.0)
        h_np = np.array(h)
        expected = np.array([10.0, 10.0, 10.0, 10.0, 8.05, 8.05])
        assert np.linalg.norm(h_np - expected) < 1e-1

    def test_unstable_system_with_control(self):
        """BBM17 Example 10.13 – unstable system stabilised by bounded control."""
        F = jl.seval("[1.5 0 ; 1 -1.5]")
        G = jl.seval("[1.0; 0;;]")
        xmin = jl.seval("-10*ones(2)")
        xmax = jl.seval("10*ones(2)")
        wmin = jl.seval("-0.1*ones(2)")
        wmax = jl.seval("0.1*ones(2)")
        umin = jl.seval("-[5.0]")
        umax = jl.seval("[5.0]")
        _, h = LinearMPC.invariant_set(
            F, xmin, xmax,
            G=G, umin=umin, umax=umax,
            wmin=wmin, wmax=wmax,
            eps_shrink=0.0,
        )
        h_np = np.array(h)
        expected = np.array([3.72, 3.72, 2.008, 2.008])
        assert np.linalg.norm(h_np - expected) < 1e-2


# ---------------------------------------------------------------------------
# Reference preview
# ---------------------------------------------------------------------------
class TestReferencePreview:
    def _make_mpc(self):
        A = np.array([[0.0, 1], [10, 0]])
        B = np.array([[0.0], [1]])
        C = np.array([[1.0, 0], [0, 1.0]])
        mpc = MPC(A, B, 0.1, C=C, Np=5, Nc=3)
        mpc.set_bounds(umin=[-20.0], umax=[20.0])
        mpc.set_objective(Q=[1.0, 1.0], R=[0.1], Rr=np.array([[0.1]]))
        return mpc

    def test_default_is_disabled(self):
        mpc = self._make_mpc()
        assert mpc.jl_mpc.settings.reference_preview is False

    def test_single_control_output(self):
        mpc = self._make_mpc()
        u = mpc.compute_control(np.array([1.0, 0.0]), r=np.array([0.0, 0.0]))
        assert len(u) == 1

    def test_preview_gives_different_control(self):
        """Reference preview should yield different control when reference changes."""
        mpc = self._make_mpc()
        r_dynamic = np.array([[0.0, 1.0, 2.0, 1.0, 0.0],
                               [0.0, 0.0, 0.5, 1.0, 0.5]])
        u_no_preview = mpc.compute_control(np.array([1.0, 0.0]),
                                           r=np.array([0.0, 0.0]))
        mpc.settings({"reference_preview": True})
        mpc.setup()
        u_with_preview = mpc.compute_control(np.array([1.0, 0.0]), r=r_dynamic)
        assert np.linalg.norm(np.array(u_no_preview) - np.array(u_with_preview)) > 1e-1

    def test_parameter_dimensions(self):
        """Parameter dims should reflect the reference preview horizon."""
        mpc = self._make_mpc()
        mpc.settings({"reference_preview": True})
        mpc.setup()
        nx, nr, nd, nuprev, *_ = LinearMPC.get_parameter_dims(mpc.jl_mpc)
        assert int(nx) == 2
        assert int(nr) == 2 * 5   # ny × Np
        assert int(nd) == 0
        assert int(nuprev) == 1   # Rr != 0 → uprev tracking


# ---------------------------------------------------------------------------
# Reference preview simulation
# ---------------------------------------------------------------------------
class TestReferencePreviewSimulation:
    def test_preview_improves_tracking(self):
        """
        A simulation with reference preview should yield lower tracking error
        than one without, for a changing reference.
        """
        A = np.array([[1.0, 1], [0, 1]])
        B = np.array([[0.0], [1]])
        C = np.array([[1.0, 0], [0, 1.0]])
        mpc = MPC(A, B, C=C, Np=5, Nc=3)
        mpc.set_bounds(umin=[-2.0], umax=[2.0], ymin=[-1.0, -0.5], ymax=[1.0, 0.5])
        mpc.set_objective(Q=[1.0, 1.0], R=[0.1])

        N_sim = 20
        r_traj = np.zeros((2, N_sim))
        r_traj[0, :] = np.concatenate([np.zeros(10), np.ones(10)])
        x0 = np.array([1.0, 0.0])

        # Without preview
        sim_no_preview = Simulation(mpc, x0=x0, N=N_sim, r=r_traj)

        # With preview
        mpc.settings({"reference_preview": True})
        mpc.setup()
        sim_preview = Simulation(mpc, x0=x0, N=N_sim, r=r_traj)

        assert sim_preview.xs.shape == (2, N_sim)
        assert sim_preview.us.shape == (1, N_sim)

        # Preview should reduce overall tracking error
        e_preview = sim_preview.ys - sim_preview.rs
        e_no_preview = sim_no_preview.ys - sim_no_preview.rs
        assert np.linalg.norm(e_preview) / np.linalg.norm(e_no_preview) < 0.9

        # Both should approximately reach the final reference
        assert np.linalg.norm(e_preview[:, -1]) < 1e-3
        assert np.linalg.norm(e_no_preview[:, -1]) < 1e-3


# ---------------------------------------------------------------------------
# Robust MPC
# ---------------------------------------------------------------------------
class TestRobustMPC:
    def test_disturbance_tightens_constraints(self):
        """
        Setting a disturbance model should tighten the constraint bounds and
        prevent constraint violation in closed-loop simulation.
        """
        F = np.array([[1.0, 1], [0, 1]])
        G = np.array([[1.0], [0.5]])
        mpc = MPC(F, G, Np=10)
        mpc.set_prestabilizing_feedback()
        mpc.set_bounds(umin=[-1], umax=[1])
        mpc.set_output_bounds(ymin=-0.15 * np.ones(2), ymax=np.ones(2), soft=False)

        # Nominal: bounds can be violated
        mpqp_nominal = mpc.mpqp()
        sim_nominal = Simulation(mpc, x0=np.array([0.9, 0.5]), N=100,
                                 r=np.array([0.0, 0]))
        assert np.min(sim_nominal.xs[1, :]) < -0.1

        # Set disturbance
        wmin = -np.array([1e-2, 1e-1])
        wmax = np.array([1e-2, 1e-1])
        mpc.set_disturbance(wmin, wmax)
        mpqp_tightened = mpc.mpqp()

        # Tightened bounds
        assert np.sum(mpqp_tightened["bu"]) < np.sum(mpqp_nominal["bu"])
        assert np.sum(mpqp_tightened["bl"]) > np.sum(mpqp_nominal["bl"])

        # Tightened simulation respects constraint
        sim_tight = Simulation(mpc, x0=np.array([0.9, 0.5]), N=100,
                               r=np.array([0.0, 0]))
        assert np.min(sim_tight.xs[1, :]) > -0.1


# ---------------------------------------------------------------------------
# Control trajectory
# ---------------------------------------------------------------------------
class TestControlTrajectory:
    def test_returns_single_control(self):
        """compute_control returns a 1-element vector for a SISO system."""
        A = np.array([[1.0, 1], [0, 1]])
        B = np.array([[0.0], [1]])
        C = np.array([[1.0, 0], [0, 1.0]])
        mpc = MPC(A, B, C=C, Np=5, Nc=5)
        u = mpc.compute_control(np.array([0.5, 1.0]))
        assert len(u) == 1


# ---------------------------------------------------------------------------
# x0 uncertainty
# ---------------------------------------------------------------------------
class TestX0Uncertainty:
    def test_converges_with_tightened_bounds(self):
        """
        x0 uncertainty tightens the output bounds; the closed-loop output
        settles at 0.4 (not 0.5) when ymax=0.5 and uncertainty=0.1.
        """
        F = np.array([[1.0, 0.1], [0, 1]])
        G = np.array([[0.005], [0.1]])
        mpc = MPC(F, G, Np=25, C=np.array([[1.0, 0]]))
        mpc.set_bounds(umin=[-0.2], umax=[0.2],
                       ymin=[-0.5], ymax=[0.5])
        mpc.set_x0_uncertainty(0.1 * np.ones(2))
        sim = Simulation(mpc, r=np.array([0.5]), N=1000)
        assert abs(sim.xs[0, -1] - 0.4) < 1e-6


# ---------------------------------------------------------------------------
# Linear cost
# ---------------------------------------------------------------------------
class TestLinearCost:
    def _make_mpc(self):
        A = np.array([[1.0, 1], [0, 1]])
        B = np.array([[0.0], [1]])
        C = np.array([[1.0, 0], [0, 1.0]])
        mpc = MPC(A, B, C=C, Np=5, Nc=3)
        mpc.set_bounds(umin=[0.0], umax=[2.0])
        mpc.set_objective(Q=[1.0, 1.0], R=[0.1])
        return mpc

    def test_disabled_by_default(self):
        mpc = self._make_mpc()
        assert mpc.jl_mpc.settings.linear_cost is False

    def test_control_length(self):
        mpc = self._make_mpc()
        u = mpc.compute_control(np.array([-1.0, 0.0]), r=[0.0, 0.0])
        assert len(u) == 1

    def test_linear_cost_ordering(self):
        """
        Positive linear cost should decrease control; negative should increase it.
        """
        mpc = self._make_mpc()
        u_no_lincost = mpc.compute_control(np.array([-1.0, 0.0]), r=[0.0, 0.0])

        mpc.settings({"linear_cost": True})
        mpc.setup()

        u_zero = mpc.compute_control(np.array([-1.0, 0.0]), r=[0.0, 0.0],
                                     l=np.zeros(1))
        u_pos = mpc.compute_control(np.array([-1.0, 0.0]), r=[0.0, 0.0],
                                    l=np.array([1.0]))
        u_neg = mpc.compute_control(np.array([-1.0, 0.0]), r=[0.0, 0.0],
                                    l=np.array([-1.0]))

        # Zero linear cost gives same result as no linear cost
        assert abs(float(u_zero[0]) - float(u_no_lincost[0])) < 1e-12
        # Ordering: positive cost reduces control, negative increases it
        assert float(u_pos[0]) < float(u_zero[0]) < float(u_neg[0])

    def test_linear_cost_parameter_dims(self):
        """Parameter dimensions should include nl = nu × Nc when linear_cost=True."""
        mpc = self._make_mpc()
        mpc.settings({"linear_cost": True})
        mpc.setup()
        nx, nr, nd, nuprev, nl = LinearMPC.get_parameter_dims(mpc.jl_mpc)
        assert int(nx) == 2
        assert int(nr) == 2
        assert int(nd) == 0
        assert int(nuprev) == 0
        assert int(nl) == 1 * 3   # nu × Nc


# ---------------------------------------------------------------------------
# Linear cost simulation
# ---------------------------------------------------------------------------
class TestLinearCostSimulation:
    def test_linear_cost_reduces_total_cost(self):
        """
        A controller that accounts for the linear cost term should achieve
        a lower value of the augmented cost than one that ignores it.
        """
        A = np.array([[0.0, -0.37], [0.37, 0.74]])
        B = np.array([[0.37], [0.26]])
        C = np.array([[1.0, 0], [0, 1.0]])
        mpc = MPC(A, B, C=C, Np=5, Nc=3)
        mpc.set_bounds(umin=[-2.0], umax=[2.0])
        mpc.set_objective(Q=[1.0, 1.0], R=[0.1])
        mpc.set_terminal_cost()
        mpc.settings({"linear_cost": True})
        mpc.setup()

        N_sim = 20
        r_traj = np.zeros((2, N_sim))
        l_traj = np.full((1, N_sim), -0.5)

        sim_with_l = Simulation(mpc, x0=np.array([1.0, 0.0]), N=N_sim,
                                r=r_traj, l=l_traj)
        sim_no_l = Simulation(mpc, x0=np.array([1.0, 0.0]), N=N_sim,
                              r=r_traj)

        assert sim_with_l.xs.shape == (2, N_sim)
        assert sim_with_l.us.shape == (1, N_sim)

        # The controller that minimises the augmented cost should do so
        Qf = np.array(mpc.jl_mpc.weights.Qf, copy=False, order="F")
        cost_l = (np.sum(sim_with_l.xs ** 2)
                  + 0.1 * np.sum(sim_with_l.us ** 2)
                  + np.dot(sim_with_l.us.ravel(), l_traj.ravel())
                  + sim_with_l.xs[:, -1] @ Qf @ sim_with_l.xs[:, -1])
        cost_no_l = (np.sum(sim_no_l.xs ** 2)
                     + 0.1 * np.sum(sim_no_l.us ** 2)
                     + np.dot(sim_no_l.us.ravel(), l_traj.ravel())
                     + sim_no_l.xs[:, -1] @ Qf @ sim_no_l.xs[:, -1])
        assert cost_l < cost_no_l


# ---------------------------------------------------------------------------
# MPQP preprocessing
# ---------------------------------------------------------------------------
class TestMPQPPreprocessing:
    def test_redundant_constraints_eliminated(self):
        """
        Constraints that are redundant w.r.t. box bounds on u should be
        removed, leaving an empty parametric part A of the mpQP.
        """
        A = np.array([[0.0, 1], [10, 0]])
        B = np.array([[0.0], [1]])
        mpc = MPC(A, B, 0.1)
        mpc.set_bounds(umin=np.array([-1.0]), umax=np.array([1.0]))
        mpc.add_constraint(Au=np.array([[-1.0]]), lb=[-0.9], ub=[1.5],
                           ks=range(10))
        mpc.add_constraint(Au=np.array([[1.0]]),  lb=[-0.5], ub=[2.0],
                           ks=range(10))
        mpc.setup()
        mpqp = mpc.mpqp()

        assert mpqp["A"].size == 0
        assert np.allclose(mpqp["bu"], 0.9 * np.ones(10))
        assert np.allclose(mpqp["bl"], -0.5 * np.ones(10))


# ---------------------------------------------------------------------------
# Set offset
# ---------------------------------------------------------------------------
class TestSetOffset:
    def test_offset_shifts_steady_state(self):
        """
        An output offset of 0.5 should shift the steady-state output from the
        reference 1.5 down to 1.5, while the control settles at uo + Δu = 10.5.
        """
        mpc = MPC(np.array([[0.778800783]]), np.array([[1.0]]),
                  C=np.array([[0.44239843385]]))
        mpc.set_objective(Q=[1.0], R=np.zeros((1, 1)), Rr=np.array([[0.1]]))
        LinearMPC.set_offset_b(mpc.jl_mpc,
                               uo=np.array([10.0]), ho=np.array([0.5]))
        sim = Simulation(mpc, x0=np.zeros(1), r=np.array([1.5]), N=50)
        assert abs(sim.us[0, -1] - 10.5) < 1e-6
        assert abs(sim.ys[0, -1] - 1.5) < 1e-6


# ---------------------------------------------------------------------------
# Unconstrained MPC
# ---------------------------------------------------------------------------
class TestUnconstrained:
    def test_converges_to_reference(self):
        """Unconstrained MPC with move blocking converges to the set-point."""
        mpc = MPC(np.array([[0.77880078307]]), np.array([[1.0]]),
                  C=np.array([[2.211992169]]))
        mpc.move_block([2, 2, 2, 24])
        mpc.set_objective(Q=np.array([[1.0]]),
                          Rr=np.zeros((1, 1)), R=np.zeros((1, 1)))
        sim = Simulation(mpc, x0=np.zeros(1), r=np.array([5.0]), N=20)
        assert abs(sim.ys[0, -1] - 5.0) < 1e-3


# ---------------------------------------------------------------------------
# Game-theoretic MPC
# ---------------------------------------------------------------------------
class TestGameTheoreticMPC:
    def test_asymmetric_hessian_and_convergence(self):
        """
        In game-theoretic MPC the Hessian is NOT symmetric.  Implicit and
        explicit solvers should agree, and both agents converge to their targets.
        """
        F = np.array([[1.0, 0.1], [0, 1]])
        G = np.array([[0.0, 0], [1, 1]])
        mpc = MPC(F, G, C=np.eye(2), Np=10)
        mpc.set_objective(uids=[1], Q=np.array([1.0, 0]), Rr=np.array([[1e3]]))
        mpc.set_objective(uids=[2], Q=np.array([0.0, 1]), Rr=np.array([[1e3]]))
        mpc.set_bounds(umin=-np.ones(2), umax=np.ones(2))
        mpc.move_block([1, 1, 8])

        H = mpc.mpqp()["H"]
        assert not np.allclose(H, H.T)

        empc = ExplicitMPC(mpc, build_tree=True)

        x0 = 10 * np.ones(2)
        r = np.array([10.0, 0])
        N = 500
        sim_imp = Simulation(mpc, x0=x0, r=r, N=N)
        sim_exp = Simulation(empc, x0=x0, r=r, N=N)

        assert np.allclose(sim_imp.us, sim_exp.us, atol=1e-4)
        assert abs(sim_imp.ys[0, -1] - 10.0) < 1e-4
        assert abs(sim_imp.ys[1, -1] - 0.0) < 1e-4
        assert abs(sim_exp.ys[0, -1] - 10.0) < 1e-4
        assert abs(sim_exp.ys[1, -1] - 0.0) < 1e-4


# ---------------------------------------------------------------------------
# Evaluate running cost
# ---------------------------------------------------------------------------
class TestEvaluateRunningCost:
    def test_cost_is_positive(self):
        """evaluate_cost should return a positive scalar for any non-trivial sim."""
        jl_mpc, _ = LinearMPC.mpc_examples("invpend")
        x0 = np.zeros(4)
        N_sim = 1000
        rs = np.hstack([
            np.zeros((2, 20)),
            np.tile(np.array([[10.0], [0.0]]), (1, 780)),
            np.tile(np.array([[9.0], [0.0]]), (1, 10)),
        ])
        sim = LinearMPC.Simulation(jl_mpc, x0=x0, N=N_sim, r=rs)
        cost = LinearMPC.evaluate_cost(jl_mpc, sim)
        assert float(cost) > 0


# ---------------------------------------------------------------------------
# Python mpc_examples, ParameterRange, and Model wrappers
# ---------------------------------------------------------------------------
class TestMPCExamplesPython:
    def test_returns_python_mpc_and_range(self):
        """mpc_examples() should return a Python MPC and a Python ParameterRange."""
        mpc, pr = mpc_examples("invpend")
        assert isinstance(mpc, MPC)
        assert isinstance(pr, ParameterRange)

    def test_compute_control_on_example(self):
        """compute_control on the Python-wrapped invpend example should match reference."""
        mpc, _ = mpc_examples("invpend")
        u = mpc.compute_control([5.0, 5, 0, 0])
        assert abs(float(u[0]) - 1.7612519326) < 1e-6

    def test_mpc_examples_with_np_nc(self):
        """mpc_examples with explicit Np/Nc returns valid Python objects."""
        mpc, pr = mpc_examples("aircraft", Np=10, Nc=2)
        assert isinstance(mpc, MPC)
        assert isinstance(pr, ParameterRange)

    def test_mpc_examples_with_params(self):
        """mpc_examples with params dict returns valid Python objects."""
        mpc, pr = mpc_examples("chained", Np=10, Nc=10, params={"nx": 2})
        assert isinstance(mpc, MPC)
        assert isinstance(pr, ParameterRange)


class TestParameterRange:
    def test_range_method_returns_python_range(self):
        """MPC.range() should return a Python ParameterRange."""
        A = np.array([[0.0, 1], [10, 0]])
        B = np.array([[0.0], [1]])
        mpc = MPC(A, B, 0.1, Np=5)
        pr = mpc.range(xmin=[-5, -5], xmax=[5, 5])
        assert isinstance(pr, ParameterRange)

    def test_range_fields_readable(self):
        """ParameterRange fields returned by mpc_examples should be readable."""
        _, pr = mpc_examples("invpend")
        xmin = np.array(pr.xmin)
        xmax = np.array(pr.xmax)
        assert xmin.shape == (4,)
        assert xmax.shape == (4,)

    def test_explicit_mpc_with_python_range(self):
        """ExplicitMPC should accept a Python ParameterRange and give correct control."""
        mpc, pr = mpc_examples("invpend")
        empc = ExplicitMPC(mpc, range=pr)
        LinearMPC.build_tree_b(empc.jl_mpc)
        u = LinearMPC.compute_control(empc.jl_mpc, [5.0, 5, 0, 0])
        assert abs(float(u[0]) - 1.7612519326) < 1e-6

    def test_certify_with_python_range(self):
        """MPC.certify() should accept a Python ParameterRange."""
        mpc, pr = mpc_examples("invpend")
        result = mpc.certify(range=pr)
        assert result.jl_result is not None

    def test_range_method_umin_umax(self):
        """MPC.range() should correctly apply umin/umax (regression for guard bug)."""
        A = np.array([[0.0, 1], [10, 0]])
        B = np.array([[0.0], [1]])
        mpc = MPC(A, B, 0.1, Np=5)
        mpc.set_bounds(umin=[-1.0], umax=[1.0])
        pr = mpc.range(xmin=[-5, -5], xmax=[5, 5], umin=[-0.5], umax=[0.5])
        assert float(pr.umin[0]) == pytest.approx(-0.5)
        assert float(pr.umax[0]) == pytest.approx(0.5)


class TestModel:
    def test_model_linear_dt(self):
        """Model with DT linear matrices should produce a valid MPC."""
        F = np.array([[1.0, 0.1], [0.0, 1.0]])
        G = np.array([[0.0], [0.1]])
        model = Model(F, G)
        mpc = MPC(model, Np=5)
        mpc.set_bounds(umin=[-1.0], umax=[1.0])
        u = mpc.compute_control(np.array([1.0, 0.0]))
        assert len(u) == 1

    def test_model_linear_ct(self):
        """Model with CT (A, B, Ts) should produce a valid MPC after ZOH."""
        A = np.array([[0.0, 1], [0, 0]])
        B = np.array([[0.0], [1]])
        model = Model(A, B, 0.1)
        mpc = MPC(model, Np=5)
        mpc.set_bounds(umin=[-1.0], umax=[1.0])
        u = mpc.compute_control(np.array([1.0, 0.0]))
        assert len(u) == 1

    def test_model_nonlinear_dt(self):
        """Model with DT nonlinear callables should linearise and compute control."""
        F = np.array([[1.0, 0.1], [0.0, 1.0]])
        G = np.array([[0.0], [0.1]])

        # Express the linear system as nonlinear functions (result must match)
        def f(x, u, d):
            x = np.asarray(x, dtype=float)
            u = np.asarray(u, dtype=float)
            return F @ x + G @ u

        def h(x, u, d):
            return np.asarray(x, dtype=float).copy()

        xo = np.zeros(2)
        uo = np.zeros(1)
        model = Model(f, h, xo, uo)   # DT nonlinear (no Ts)
        mpc = MPC(model, Np=5)
        mpc.set_bounds(umin=[-1.0], umax=[1.0])
        u = mpc.compute_control(np.array([1.0, 0.0]))
        assert len(u) == 1

    def test_model_nonlinear_ct(self):
        """Model with CT nonlinear callables uses ZOH and gives valid control."""
        A_ct = np.array([[0.0, 1], [0, 0]])
        B_ct = np.array([[0.0], [1]])

        def f(x, u, d):
            x = np.asarray(x, dtype=float)
            u = np.asarray(u, dtype=float)
            return A_ct @ x + B_ct @ u   # ẋ = Ax + Bu

        def h(x, u, d):
            return np.asarray(x, dtype=float).copy()

        xo = np.zeros(2)
        uo = np.zeros(1)
        model = Model(f, h, xo, uo, Ts=0.1)   # CT nonlinear
        mpc = MPC(model, Np=5)
        mpc.set_bounds(umin=[-1.0], umax=[1.0])
        u = mpc.compute_control(np.array([1.0, 0.0]))
        assert len(u) == 1

    def test_model_nonlinear_dt_simulation(self):
        """Simulation with a nonlinear DT Model uses true dynamics and converges."""
        F = np.array([[1.0, 0.1], [0.0, 1.0]])
        G = np.array([[0.0], [0.1]])

        def f(x, u, d):
            x = np.asarray(x, dtype=float)
            u = np.asarray(u, dtype=float)
            return F @ x + G @ u

        def h(x, u, d):
            return np.asarray(x, dtype=float).copy()

        model = Model(f, h, np.zeros(2), np.zeros(1))
        mpc = MPC(model, Np=10)
        mpc.set_objective(Q=[1.0, 1.0], R=[0.1])
        mpc.set_bounds(umin=[-2.0], umax=[2.0])

        sim = Simulation(mpc, x0=np.array([1.0, 0.0]), r=np.zeros(2), N=50)
        # Should converge to origin
        assert np.linalg.norm(sim.xs[:, -1]) < 1e-2

    def test_mpc_from_model_matches_matrices(self):
        """MPC built from a Model should give same control as one built from matrices."""
        F = np.array([[1.0, 0.1], [0.0, 1.0]])
        G = np.array([[0.0], [0.1]])

        mpc_mat = MPC(F, G, Np=5)
        mpc_mat.set_bounds(umin=[-1.0], umax=[1.0])

        model = Model(F, G)
        mpc_mod = MPC(model, Np=5)
        mpc_mod.set_bounds(umin=[-1.0], umax=[1.0])

        x0 = np.array([1.0, 0.5])
        u_mat = mpc_mat.compute_control(x0)
        u_mod = mpc_mod.compute_control(x0)
        assert np.linalg.norm(np.array(u_mat) - np.array(u_mod)) < 1e-10

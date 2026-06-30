import numpy as np
from scipy.optimize import minimize

from Game import GameDynamics

import scipy as sp
import casadi as ca

import os
import pathlib
import copy
import shutil
import atexit
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from types import SimpleNamespace

"""
To use this solver, install the prerequisites using the following steps
1. Install Julia:
- wget https://julialang-s3.julialang.org/bin/linux/x64/1.10/julia-1.10.1-linux-x86_64.tar.gz
- tar zxvf julia-1.10.1-linux-x86_64.tar.gz
- export PATH="$PATH:/path/to/<Julia directory>/bin"
2. Install Julia packages:
- In the Julia REPL package manager: 
-- add PyCall
-- add PATHSolver@1.1.1 (side note, only version 1.1.1 works when called from pyjulia)
3. Install pyjulia:
- python3 -m pip install julia
"""
from julia.api import Julia

def is_symbolic_expr(z):
    """True if z is a CasADi SX/MX expression that depends on symbols."""
    return isinstance(z, (ca.SX, ca.MX)) and len(ca.symvar(z)) > 0

def _resolve_julia_runtime():
    env_runtime = os.environ.get('JULIA_RUNTIME')
    if env_runtime and os.path.isfile(env_runtime):
        return env_runtime

    local_runtime = pathlib.Path.cwd() / 'julia-1.10.1' / 'bin' / 'julia'
    if local_runtime.is_file():
        return str(local_runtime)

    return shutil.which('julia')

jl = None
Main = None
_terminal_executor = None
_terminal_executor_workers = 0


def _ensure_julia():
    global jl, Main
    if jl is not None and Main is not None:
        return

    jl = Julia(runtime=_resolve_julia_runtime(), compiled_modules=False)
    from julia import Main as JuliaMain

    Main = JuliaMain
    jl.using("PyCall")
    jl.using("PATHSolver")


def _initialize_terminal_worker():
    """Load Julia and PATHSolver once when a persistent worker starts."""
    _ensure_julia()


def _terminal_worker_pid():
    """Return the PID after the worker initializer has completed."""
    return os.getpid()


def _get_terminal_executor(max_workers):
    """Return the process pool shared by all sampled-terminal solves."""
    global _terminal_executor, _terminal_executor_workers
    if _terminal_executor is None:
        _terminal_executor_workers = max_workers
        spawn_context = mp.get_context("spawn")
        _terminal_executor = ProcessPoolExecutor(
            max_workers=max_workers,
            mp_context=spawn_context,
            initializer=_initialize_terminal_worker,
        )
    return _terminal_executor


def initialize_pathsolver_runtime(max_workers=None):
    """Load PATHSolver once in the main process and each persistent worker.

    Call this once, before a simulation loop.  The returned process IDs are
    useful for confirming that later solves continue to use the same workers.
    """
    _ensure_julia()

    if max_workers is None:
        cpu_count = os.cpu_count() or 1
        max_workers = max(2, int(cpu_count * 0.20))
    max_workers = max(1, int(max_workers))

    if max_workers == 1:
        return (os.getpid(),)

    executor = _get_terminal_executor(max_workers)
    # Submitting the complete batch starts the pool now instead of during the
    # first sampled-terminal solve. Each process runs its initializer once.
    futures = [executor.submit(_terminal_worker_pid) for _ in range(max_workers)]
    worker_pids = {future.result() for future in futures}
    return (os.getpid(), *sorted(worker_pids))


def _shutdown_terminal_executor():
    global _terminal_executor, _terminal_executor_workers
    if _terminal_executor is not None:
        _terminal_executor.shutdown(wait=True)
        _terminal_executor = None
        _terminal_executor_workers = 0


atexit.register(_shutdown_terminal_executor)


def _solve_sampled_terminal_candidate(
    worker_solver,
    candidate_data,
    sample_index,
    t,
    x0,
    forced_alpha,
    u1_0,
    u2_0,
):
    """Process-pool entry point for one discrete terminal-state solve."""
    solver = copy.copy(worker_solver)
    solver.Solution = copy.deepcopy(worker_solver.Solution)
    solver.Solver = None
    try:
        solver._step_once(
            t,
            x0,
            forced_alpha=forced_alpha,
            u1_0=u1_0,
            u2_0=u2_0,
            terminal_learned_data=candidate_data,
        )
        if not solver.last_solve_success:
            return sample_index, None, None, None
        cost = solver._player1_cost(solver.Solution, candidate_data)
        return sample_index, cost, solver.Solution, None
    except Exception as exc:
        return sample_index, None, None, f"{type(exc).__name__}: {exc}"

class DGSolver:
    """Basic structure for a dynamic game solver."""

    def __init__(self, game: GameDynamics, xf, 
                       dt=0.1, horizon=7, 
                       alpha=0.5,
                       R1 = 0.04,
                       R2 = 0.04,
                       LearnedData = None, 
                       p_tol=1e-5, 
                       verbose = False, 
                       options=None,
                       constraint_mode="sampled_points"):
        if horizon <= 0:
            raise ValueError("horizon must be positive")

        self.game = game
        self.xf = xf
        self.N = int(horizon)
        self.dt = float(dt)
        if LearnedData is None or LearnedData.AnalyzedData.n_data == 0:
            self.LearnedData = None
        else:
            self.LearnedData = LearnedData
        valid_constraint_mode = {"convex_hull", "sampled_points"}
        if constraint_mode not in valid_constraint_mode:
            raise ValueError(
                "terminal_constraint_mode must be 'convex_hull' or 'sampled_points'"
            )
        self.constraint_mode = constraint_mode
        if self.LearnedData is None:
            self.alpha_vec = alpha * np.ones((self.N+1,1))
        else:
            self.alpha_vec = np.ones((self.N+1,1))
        
        self.options = options.copy() if options is not None else {}
        self.solver = None
        self.is_built = False
        
        self.Qk = np.diag([1.0, 1.0]) if self.game.is_single_integrator else np.diag([1.0, 1.0, 0.25, 0.25])
        self.R1 = R1
        self.R2 = R2
        self.p_tol = p_tol
        self.verbose = verbose
        self.nms = True
        
        self.proximity_Q = np.diag([1.0, 1.0, 1.0, 1.0]) if self.game.is_single_integrator else np.diag([1.0, 1.0, 1.0, 1.0, 10.0, 10.0, 10.0, 10.0])
        self.small_dx = 1/self.game.nx*np.array([1e-3, 1e-3, 1e-3, 1e-3]) if self.game.is_single_integrator else 1/self.game.nx*np.array([1e-3, 1e-3, 1e-3, 1e-3, 1e-4, 1e-4, 1e-4, 1e-4])
        self.large_dx = 1/self.game.nx*np.array([2e-1, 2e-1, 2e-1, 2e-1]) if self.game.is_single_integrator else 1/self.game.nx*np.array([2e-1, 2e-1, 2e-1, 2e-1, 2e-2, 2e-2, 2e-2, 2e-2])
        self.proximity_minval = np.array(ca.bilin(self.proximity_Q, self.small_dx)).flatten()[0]
        self.proximity_maxval = np.array(ca.bilin(self.proximity_Q, self.large_dx)).flatten()[0]
        
        self.Solution = SimpleNamespace()
        self.Solution.success = False
        self.last_solve_success = False

    def build_solver(self, LearnedData = None):
        """
        Build the dynamic game solver.

        This is a placeholder for constructing optimization variables,
        constraints, costs, and the numerical backend.
        """
                
        # Player 1 trajectory variables over the horizon.
        x1 = ca.SX.sym('x1',self.N+1, self.game.nx1)
        u1 = ca.SX.sym('u1',self.N, self.game.nu1)
        x1_0 = ca.SX.sym('x1_0',1, self.game.nx1)
        alpha_vec = ca.SX.sym('alpha_vec', self.N+1)
        c1_vec = ca.SX.sym('c1_i', 4)
        if LearnedData is not None:
            ai_xf = ca.SX.sym('ai_xf', LearnedData.AnalyzedData.state.shape[0])
        else:
            ai_xf = []

        A1, B1 = self._discrete_player_dynamics(self.game.nx1)

        # Store each player's equality constraints, private constraints, and multipliers.
        h_vec, mu_vec = [], []
        p_vec, lambda_vec = [], []
        sg_vec = []
        # Define The first player lagrangian:
        L1 = 0
        for k in range(self.N):
            L1 += ca.bilin(self.Qk, x1[k+1,:]-self.xf)
            L1 += ca.bilin(self.R1*np.eye(self.game.nu1), u1[k,:])
            
        if LearnedData is not None:
            L1 += ca.mtimes(LearnedData.AnalyzedData.Cost2Go.reshape(1,-1), ai_xf)
        
        # Player 1 Dynamics:
        h = []
        n_mu = 0
        for k in range(self.N+1):
            if k == 0:
                h.append(x1[k,:].T - x1_0.T)
            else:
                h.append(x1[k,:].T - A1@x1[k-1,:].T - B1@u1[k-1,:].T)
            n_mu += h[-1].shape[0]
                    
        # Final joint state is a convex combination of the smapled dataset
        if LearnedData is not None:
            h.append(ca.mtimes(LearnedData.AnalyzedData.state.T[:self.game.nx1,:], ai_xf) - x1[self.N,:].T)
            n_mu += h[-1].shape[0]
            h.append(1.0 - ca.sum1(ai_xf))
            n_mu += h[-1].shape[0]
        
        mu1 = ca.SX.sym(f'mu_1', n_mu)
        L1 += ca.dot(mu1, ca.vertcat(*h))
        h_vec.append(ca.vertcat(*h))
        mu_vec.append(mu1)
            
        # Player 1 Private Constraints:
        p1 = []
        for k in range(self.N + 1):
            px = x1[k, 0]
            py = x1[k, 1]

            # Position bounds: x_min < px < x_max, y_min < py < y_max.
            p1.extend(
                [
                    px - self.game.x_min,
                    self.game.x_max - px,
                    py - self.game.y_min,
                    self.game.y_max - py,
                ]
            )   

            if not self.game.is_single_integrator:
                vx = x1[k, 2]
                vy = x1[k, 3]

                # Velocity bounds: vx_min < vx < vx_max, vy_min < vy < vy_max.
                p1.extend(
                    [
                        vx - self.game.vx_min,
                        self.game.vx_max - vx,
                        vy - self.game.vy_min,
                        self.game.vy_max - vy,
                    ]
                )

            if k < self.N:
                ax = u1[k, 0]
                ay = u1[k, 1]

                # Input bounds for Player 1: velocity in single-integrator mode,
                # acceleration in double-integrator mode.
                p1.extend(
                    [
                        ax - self.game.u_min,
                        self.game.u_max - ax,
                        ay - self.game.u_min,
                        self.game.u_max - ay,
                    ]
                )
                
                if k==0:                    
                    p1.extend(
                        [
                            ax - c1_vec[0],
                            c1_vec[1] - ax,
                            ay - c1_vec[2],
                            c1_vec[3] - ay,
                        ])

        # Final joint state is a convex combination of the smapled dataset
        if LearnedData is not None:
            p1.append(1.0 - ai_xf)
            p1.append(ai_xf)

        p1_ph = ca.vertcat(*p1)
        lambda_1 = ca.SX.sym("lambda_1", p1_ph.shape[0])
        L1 -= ca.dot(lambda_1, p1_ph)
        p_vec.append(p1_ph)
        lambda_vec.append(lambda_1)

        # Player 2 trajectory variables over the horizon.
        x2 = ca.SX.sym('x2', self.N+1, self.game.nx2)
        u2 = ca.SX.sym('u2', self.N, self.game.nu2)
        x2_0 = ca.SX.sym('x2_0', 1, self.game.nx2)

        A2, B2 = self._discrete_player_dynamics(self.game.nx2)

        # Define the second player lagrangian using the same quadratic structure.
        L2 = 0
        for k in range(self.N):
            L2 += ca.bilin(self.Qk, x2[k+1, :] - self.xf)
            L2 += ca.bilin(self.R2 * np.eye(self.game.nu2), u2[k, :])

        # Player 2 dynamics are equality constraints enforced by mu_2.
        h = []
        n_mu = 0
        for k in range(self.N + 1):
            if k == 0:
                h.append(x2[k, :].T - x2_0.T)
            else:
                h.append(x2[k, :].T - A2 @ x2[k-1, :].T - B2 @ u2[k-1, :].T)
            n_mu += h[-1].shape[0]
            
        # The final state must be a convex combination of the sampled dataset
        if LearnedData is not None:
            h.append(ca.mtimes(LearnedData.AnalyzedData.state.T[self.game.nx1:,:], ai_xf) - x2[self.N,:].T)
            n_mu += h[-1].shape[0]

        mu2 = ca.SX.sym('mu_2', n_mu)
        L2 += ca.dot(mu2, ca.vertcat(*h))
        h_vec.append(ca.vertcat(*h))
        mu_vec.append(mu2)

        # Player 2 Private Constraints:
        # All inequalities are written in positive form p(x2, u2) > 0.
        # Player 2 state layout is [px, py] or [px, py, vx, vy].
        p2 = []
        for k in range(self.N + 1):
            px = x2[k, 0]
            py = x2[k, 1]

            # Position bounds: x_min < px < x_max, y_min < py < y_max.
            p2.extend(
                [
                    px - self.game.x_min,
                    self.game.x_max - px,
                    py - self.game.y_min,
                    self.game.y_max - py,
                ]
            )

            if not self.game.is_single_integrator:
                vx = x2[k, 2]
                vy = x2[k, 3]

                # Velocity bounds: vx_min < vx < vx_max, vy_min < vy < vy_max.
                p2.extend(
                    [
                        vx - self.game.vx_min,
                        self.game.vx_max - vx,
                        vy - self.game.vy_min,
                        self.game.vy_max - vy,
                    ]
                )

            if k < self.N:
                ax = u2[k, 0]
                ay = u2[k, 1]

                # Input bounds for Player 2: velocity in single-integrator mode,
                # acceleration in double-integrator mode.
                p2.extend(
                    [
                        ax - self.game.u_min,
                        self.game.u_max - ax,
                        ay - self.game.u_min,
                        self.game.u_max - ay,
                    ]
                )

        p2_ph = ca.vertcat(*p2)
        lambda_2 = ca.SX.sym("lambda_2", p2_ph.shape[0])

        # Positive-form constraints enter the lagrangian with nonnegative multipliers.
        L2 -= ca.dot(lambda_2, p2_ph)
        p_vec.append(p2_ph)
        lambda_vec.append(lambda_2)
        
        # Shared constranits:
        Sc = []
        n_ls = 0
        alpha_vec_k = []
        for k in range(self.N+1):
            if k<self.N:
                f_val_k = self.game.f_shared(ca.horzcat(x1[k,:], x2[k,:]), u1[k,:], u2[k,:])
            else:
                f_val_k = self.game.f_shared(ca.horzcat(x1[k,:], x2[k,:]), np.zeros_like(u1[0,:].shape), np.zeros_like(u2[0,:].shape))
            if len(f_val_k)>0:
                for f_k in f_val_k:
                    if is_symbolic_expr(f_k): 
                        Sc.append(f_k)
                        alpha_vec_k.append(alpha_vec[k])
        sg_vec = ca.vertcat(*Sc)
        alpha_vec_k = ca.vertcat(*alpha_vec_k)
        n_ls += sg_vec.shape[0]
        
        sigma_vec = ca.SX.sym('sigma', n_ls)
        L1 -= ca.dot(alpha_vec_k*sigma_vec, sg_vec)
        L2 -= ca.dot((1-alpha_vec_k)*sigma_vec, sg_vec)
        
        # Build Z vector and F and J functions:        
        Z_len = []
        Z = []
        z1 = ca.vertcat(x1[:], u1[:], ai_xf[:])
        Z.append(z1)
        Z_len.append([ca.vertcat(x1[:]).shape[0], ca.vertcat(u1[:]).shape[0], ca.vertcat(ai_xf[:]).shape[0]])
        z2 = ca.vertcat(x2[:], u2[:])
        Z.append(z2)
        Z_len.append([ca.vertcat(x2[:]).shape[0], ca.vertcat(u2[:]).shape[0]])
        Z.append(ca.vertcat(*mu_vec))
        Z_len.append(Z[-1].shape[0])
        Z.append(ca.vertcat(*lambda_vec))
        Z_len.append(Z[-1].shape[0])
        Z.append(ca.vertcat(sigma_vec[:]))
        Z_len.append(Z[-1].shape[0])
        Z = ca.vertcat(*Z)
        
        _Dxu_L = []
        _Dxu_L.append(ca.jacobian(L1, z1).T)
        _Dxu_L.append(ca.jacobian(L2, z2).T)
        
        F = ca.vertcat(*_Dxu_L, *h_vec, *p_vec, sg_vec)
        J = ca.jacobian(F, Z)

        self.A1 = A1
        self.B1 = B1
        self.A2 = A2
        self.B2 = B2
        
        # Expose the symbolic game components for the later PATHSolver backend.
        self.solver = SimpleNamespace()
        self.solver.params = {
            "nx": self.game.nx,
            "nu": self.game.nu,
            "horizon": self.N,
            "dynamics_type": self.game.dynamics_type,
            "options": self.options,
        }
        self.solver.params["lagrangians"] = [L1, L2]
        self.solver.params["equality_constraints"] = h_vec
        self.solver.params["equality_multipliers"] = mu_vec
        self.solver.params["private_constraints"] = p_vec
        self.solver.params["private_constraint_multipliers"] = lambda_vec
        self.solver.params["shared_constraints"] = sg_vec
        self.solver.params["shared_constraint_multipliers"] = sigma_vec
        self.solver.params["lagrangians"] = [ca.Function('L1',[Z, x1_0, x2_0, alpha_vec, c1_vec],[L1]), ca.Function('L1',[Z, x1_0, x2_0, alpha_vec, c1_vec],[L1])]
        self.solver.Z = Z
        self.solver.Z_len = Z_len
        self.solver.F = ca.Function('F', [Z, x1_0, x2_0, alpha_vec, c1_vec], [F])
        self.solver.J = ca.Function('J', [Z, x1_0, x2_0, alpha_vec, c1_vec], [J])
        self.solver.n_l_inf = sum(Z_len[0]) + sum(Z_len[1]) + Z_len[2]
        self.solver.n_u_inf = self.solver.n_l_inf + int(np.sum(Z_len[3:]))

        self.is_built = True
        return self.solver

    def _discrete_player_dynamics(self, nx):
        if self.game.is_single_integrator:
            return np.eye(nx), self.dt * np.eye(nx)

        A = np.eye(nx)
        A[0, 2] = self.dt
        A[1, 3] = self.dt
        B = np.array(
            [
                [0.5 * self.dt**2, 0.0],
                [0.0, 0.5 * self.dt**2],
                [self.dt, 0.0],
                [0.0, self.dt],
            ]
        )
        return A, B

    def step(self, t, x0, forced_alpha=None, u1_0=None, u2_0=None):
        """Solve one step using the configured learned terminal-state mode."""
        if self.constraint_mode == "sampled_points" and self.LearnedData is not None:
            return self._step_over_sampled_terminal_states(
                t, x0, forced_alpha=forced_alpha, u1_0=u1_0, u2_0=u2_0
            )
        return self._step_once(
            t, x0, forced_alpha=forced_alpha, u1_0=u1_0, u2_0=u2_0
        )

    def _step_over_sampled_terminal_states(
        self, t, x0, forced_alpha=None, u1_0=None, u2_0=None
    ):
        """Enumerate learned terminal points and keep the lowest-cost P1 solution."""
        analyzed = self.LearnedData.AnalyzedData
        states = np.asarray(analyzed.state)
        sample_times = np.asarray(analyzed.t)
        previous_solution = copy.deepcopy(self.Solution)
        previous_sample_time = getattr(previous_solution, "terminal_sample_time", -np.inf)
        candidate_indices = np.where(
            (sample_times > t)
            & (sample_times > previous_sample_time-2*self.dt)
            & (sample_times <= t + (1.5 * self.N) * self.dt)
        )[0]
        if candidate_indices.shape[0]==0:
            candidate_indices = np.where((states[:,0] == self.game.xf[0,0]) & (states[:,1] == self.game.xf[0,1]))[0]
        previous_solver = getattr(self, "Solver", None)
        best_solution = None
        best_solver = None
        best_cost = np.inf
        candidate_data_by_index = {}
        for sample_index in candidate_indices:
            candidate_data = copy.deepcopy(self.LearnedData)
            candidate = candidate_data.AnalyzedData
            for field in ("t", "c", "state", "Cost2Go"):
                values = np.asarray(getattr(analyzed, field))
                setattr(candidate, field, values[[sample_index]])
            candidate.n_data = 1
            candidate_data_by_index[int(sample_index)] = candidate_data

        cpu_count = os.cpu_count() or 1
        max_workers = min(len(candidate_indices), max(1, int(cpu_count * 0.20)))
        candidate_results = []
        if max_workers == 1:
            for sample_index, candidate_data in candidate_data_by_index.items():
                self.Solution = copy.deepcopy(previous_solution)
                self._step_once(
                    t,
                    x0,
                    forced_alpha=forced_alpha,
                    u1_0=u1_0,
                    u2_0=u2_0,
                    terminal_learned_data=candidate_data,
                )
                if self.last_solve_success:
                    candidate_results.append(
                        (
                            sample_index,
                            self._player1_cost(self.Solution, candidate_data),
                            copy.deepcopy(self.Solution),
                            self.Solver,
                        )
                    )
        elif max_workers > 1:
            worker_solver = copy.copy(self)
            worker_solver.Solution = copy.deepcopy(previous_solution)
            worker_solver.Solver = None
            worker_solver.solver = None
            worker_solver.is_built = False
            executor = _get_terminal_executor(
                max(2, max(1, int(cpu_count * 0.20)))
            )
            futures = {
                executor.submit(
                    _solve_sampled_terminal_candidate,
                    worker_solver,
                    candidate_data,
                    sample_index,
                    t,
                    x0,
                    forced_alpha,
                    u1_0,
                    u2_0,
                ): sample_index
                for sample_index, candidate_data in candidate_data_by_index.items()
            }
            for future in as_completed(futures):
                submitted_index = futures[future]
                try:
                    sample_index, cost, solution, error = future.result()
                except Exception as exc:
                    if self.verbose:
                        print(
                            f"Terminal sample {submitted_index} worker failed: "
                            f"{type(exc).__name__}: {exc}"
                        )
                    continue
                if error is not None:
                    if self.verbose:
                        print(f"Terminal sample {sample_index} failed: {error}")
                    continue
                if solution is not None:
                    candidate_results.append(
                        (sample_index, cost, solution, None)
                    )

        for sample_index, candidate_cost, candidate_solution, candidate_solver in candidate_results:
            if candidate_cost < best_cost:
                best_cost = candidate_cost
                best_solution = candidate_solution
                best_solver = candidate_solver
                best_solution.terminal_sample_index = sample_index
                best_solution.terminal_sample_time = float(sample_times[sample_index])
                best_solution.terminal_sample_state = states[sample_index].copy()
                best_solution.player1_cost = candidate_cost
                best_solution.terminal_workers = max_workers

        if best_solution is None:
            self.Solution = previous_solution
            self.Solver = previous_solver
            self.last_solve_success = False
            self.Solution.success = False
            if hasattr(self.Solution, "u1") and hasattr(self.Solution, "u2"):
                if u1_0 is None:
                    self.Solution.indx = min(self.Solution.indx + 1, self.N - 1)
                return np.concatenate(
                    (self.Solution.u1[self.Solution.indx], self.Solution.u2[self.Solution.indx])
                )
            return np.zeros(self.game.nu)

        self.Solution = best_solution
        if best_solver is None:
            best_solver = self.build_solver(
                LearnedData=candidate_data_by_index[
                    best_solution.terminal_sample_index
                ]
            )
        self.Solver = best_solver
        self.last_solve_success = True
        return np.concatenate((best_solution.u1[0], best_solution.u2[0]))

    def _player1_cost(self, solution, learned_data):
        """Evaluate the primal player-1 objective used to rank terminal samples."""
        cost = 0.0
        target = np.asarray(self.xf, dtype=float).reshape(-1)
        for k in range(self.N):
            dx = solution.x1[k + 1] - target
            cost += float(dx @ self.Qk @ dx)
            cost += float(self.R1 * (solution.u1[k] @ solution.u1[k]))
        cost_to_go = np.asarray(
            learned_data.AnalyzedData.Cost2Go, dtype=float
        ).reshape(-1)
        weights = np.asarray(solution.ai_xf_vec, dtype=float).reshape(-1)
        return cost + float(cost_to_go @ weights)

    def _step_once(
        self,
        t,
        x0,
        forced_alpha=None,
        u1_0=None,
        u2_0=None,
        terminal_learned_data=None,
    ):
        """
        Solve one game step and extract the planned trajectories from z.

        Args:
            t: Current simulation time.
            x0: Current state [x1, x2].
            z0: Optional warm start for PATHSolver.

        Returns:
            Tuple of (first control, success flag, residual, solver status).
        """
        
        a_set, proximity_factor = self.calc_a_set(x0)
        alpha_vec = self.alpha_vec
        if forced_alpha is not None:
            alpha_vec = forced_alpha * np.ones_like(self.alpha_vec)
        elif self.LearnedData is not None:
            alpha_vec[0] = np.clip(1.0-proximity_factor, 0.1, 1.0)
        
        if not self.is_built and self.LearnedData is None:
            self.Solver = self.build_solver(LearnedData=self.LearnedData)
        if terminal_learned_data is not None:
            LearnedData1 = terminal_learned_data
            self.Solver = self.build_solver(LearnedData=LearnedData1)
        elif self.LearnedData is not None:
            LearnedData1 = copy.deepcopy(self.LearnedData)
            future = np.where((np.array(LearnedData1.AnalyzedData.t) > t + (3*self.N / 4) * self.dt) &
                                    (np.array(LearnedData1.AnalyzedData.t) <= t + (1.25 * self.N) * self.dt))[0]
            if future.size > 0:
                LearnedData1.AnalyzedData.t = np.array(LearnedData1.AnalyzedData.t)[future]
                LearnedData1.AnalyzedData.c = np.array(LearnedData1.AnalyzedData.c)[future]
                LearnedData1.AnalyzedData.state = np.array(LearnedData1.AnalyzedData.state)[future]
                LearnedData1.AnalyzedData.Cost2Go = np.array(LearnedData1.AnalyzedData.Cost2Go)[future]
                LearnedData1.AnalyzedData.n_data = LearnedData1.AnalyzedData.t.shape[0]
            else:
                LearnedData1 = None
            self.Solver = self.build_solver(LearnedData=LearnedData1)
        
        if abs(np.sum(a_set)-1.0)< 1e-5:
            c1_vec = a_set @ self.LearnedData.AnalyzedData.c
        else:
            c1_vec = np.array([-10+self.game.u_min, 10+self.game.u_max, -10+self.game.u_min, 10+self.game.u_max,])

        _ensure_julia()
                    
        x0 = np.asarray(x0, dtype=float)
        if x0.shape != (self.game.nx,):
            raise ValueError(f"x0 must have shape ({self.game.nx},)")

        # Design initial guess for z vector
        x1_0 = x0[:self.game.nx1].reshape(1, self.game.nx1)
        x2_0 = x0[self.game.nx1:].reshape(1, self.game.nx2)

        n_z = int(self.Solver.Z.shape[0])
        if u1_0 is None:
            u1 = np.zeros((self.N, self.game.nu1))
        else:
            u1 = np.asarray(u1_0, dtype=float)
        if u1.shape != (self.N, self.game.nu1):
            raise ValueError(f"u1_0 must have shape ({self.N}, {self.game.nu1})")

        if u2_0 is None:
            u2 = np.zeros((self.N, self.game.nu2))
        else:
            u2 = np.asarray(u2_0, dtype=float)
        if u2.shape != (self.N, self.game.nu2):
            raise ValueError(f"u2_0 must have shape ({self.N}, {self.game.nu2})")

        x1_len, u1_len, ai_len = self.Solver.Z_len[0]
        x2_len, u2_len = self.Solver.Z_len[1]
        mu_len = self.Solver.Z_len[2]
        lambda_len = self.Solver.Z_len[3]
        sigma_len = self.Solver.Z_len[4]

        x1 = np.zeros((self.N + 1, self.game.nx1))+0.1
        x2 = np.zeros((self.N + 1, self.game.nx2))-0.1
        x1[0, :] = x1_0.ravel()
        x2[0, :] = x2_0.ravel()
        for k in range(self.N):
            x1[k + 1, :] = self.A1 @ x1[k, :].T + self.B1 @ u1[k, :].T
            x2[k + 1, :] = self.A2 @ x2[k, :].T + self.B2 @ u2[k, :].T
            
        ai_xf_vec = np.zeros((ai_len,1))

        z0 = np.concatenate(
            (
                x1.reshape(x1_len, order="F"),
                u1.reshape(u1_len, order="F"),
                ai_xf_vec.reshape(ai_len, order="F"),
                x2.reshape(x2_len, order="F"),
                u2.reshape(u2_len, order="F"),
                np.zeros(mu_len),
                np.zeros(lambda_len),
                np.zeros(sigma_len),
            )
        )

        if z0.shape != (n_z,):
            raise RuntimeError(f"initial guess has shape {z0.shape}, expected ({n_z},)")
        
        Main.z0 = z0
        Main.ub = np.inf*np.ones(self.Solver.n_u_inf)
        Main.lb = np.concatenate((-np.inf*np.ones(self.Solver.n_l_inf), np.zeros(self.Solver.n_u_inf-self.Solver.n_l_inf)))
        Main.nnz = self.Solver.J.numel_out(0)
        Main.F_py = lambda z: np.array(self.Solver.F(z, x1_0, x2_0, alpha_vec, c1_vec)).squeeze()
        Main.J_py = lambda z: np.array(self.Solver.J(z, x1_0, x2_0, alpha_vec, c1_vec))
        
        Main.tol = self.p_tol

        F_def = """
        function F(n::Cint, x::Vector{Cdouble}, f::Vector{Cdouble})
            @assert n == length(x)
            f .= F_py(x)
            return Cint(0)
        end
        return(F)
        """
        Main.F = jl.eval(F_def)

        J_def = """
        function J(
            n::Cint,
            nnz::Cint,
            x::Vector{Cdouble},
            col::Vector{Cint},
            len::Vector{Cint},
            row::Vector{Cint},
            data::Vector{Cdouble},
        )
            @assert n == length(x)  == length(col) == length(len)
            @assert nnz == length(row) == length(data)
            j = Array{Float64}(undef, n, n)
            j .= J_py(x)
            i = 1
            for c in 1:n
                col[c], len[c] = i, 0
                for r in 1:n
                    # if !iszero(j[r, c])
                    #     row[i], data[i] = r, j[r, c]
                    #     len[c] += 1
                    #     i += 1
                    # end
                    row[i], data[i] = r, j[r, c]
                    len[c] += 1
                    i += 1
                end
            end
            return Cint(0)
        end
        return(J)
        """
        Main.J = jl.eval(J_def)
        
        if self.verbose:
            output = 'yes'
        else:
            output = 'no'
            
        if self.nms:
            nms = 'yes'
        else:
            nms = 'no'

        solve = f"""
        PATHSolver.c_api_License_SetString("1259252040&Courtesy&&&USR&GEN2035&5_1_2026&1000&PATH&GEN&31_12_2035&0_0_0&6000&0_0")
        status, z, info = PATHSolver.solve_mcp(F, 
                                               J,
                                               lb,
                                               ub,
                                               z0,
                                               nnz=nnz,
                                               output="{output}",
                                               convergence_tolerance=tol,
                                               nms="{nms}",
                                               crash_nbchange_limit=50,
                                               major_iteration_limit=500,
                                               minor_iteration_limit=10000,
                                               cumulative_iteration_limit=100000,
                                               restart_limit=100)
        success = status == PATHSolver.MCP_Solved

        return z, success, info.residual, status
        """
        z, success, residual, status = jl.eval(solve)
        z = np.asarray(z, dtype=float).reshape(-1)
        self.last_solve_success = bool(success)
        
        if not success:
            print(f"Solver Not Converged: residual={residual:2.2}, status={status.__name__}")

        i = 0
        x1_len, u1_len, ai_len = self.Solver.Z_len[0]
        x2_len, u2_len = self.Solver.Z_len[1]
        mu_len = self.Solver.Z_len[2]
        lambda_len = self.Solver.Z_len[3]
        sigma_len = self.Solver.Z_len[4]

        x1 = z[i:i + x1_len].reshape(self.N + 1, self.game.nx1, order="F")
        i += x1_len
        u1 = z[i:i + u1_len].reshape(self.N, self.game.nu1, order="F")
        i += u1_len
        ai_xf_vec = np.zeros((0, 1))
        if ai_len > 0:
            ai_xf_vec = z[i:i + ai_len].reshape(ai_len, 1, order="F")
            i += ai_len
        x2 = z[i:i + x2_len].reshape(self.N + 1, self.game.nx2, order="F")
        i += x2_len
        u2 = z[i:i + u2_len].reshape(self.N, self.game.nu2, order="F")
        i += u2_len
        mu = z[i:i + mu_len]
        i += mu_len
        lambdas = z[i:i + lambda_len]
        i += lambda_len
        sigma = z[i:i + sigma_len]
        i += sigma_len

        if i != z.shape[0]:
            raise RuntimeError(f"unpacked {i} entries from z, expected {z.shape[0]}")

        
        if success:
            self.Solution = SimpleNamespace()
            self.Solution.success = bool(success)
            self.Solution.t = t
            self.Solution.z = z
            self.Solution.residual = float(residual)
            self.Solution.status = status.__name__
            self.Solution.x1 = x1
            self.Solution.u1 = u1
            self.Solution.ai_xf_vec = ai_xf_vec
            self.Solution.x2 = x2
            self.Solution.u2 = u2
            self.Solution.mu = mu
            self.Solution.lambdas = lambdas
            self.Solution.sigma = sigma
            self.Solution.a_set = a_set
            self.Solution.x0 = x0
            self.Solution.indx = 0
        elif u1_0 is None and hasattr(self.Solution, "indx"):
            self.Solution.success = bool(success)
            self.Solution.indx = min(self.Solution.indx + 1, self.N - 1)

        if not hasattr(self.Solution, "u1") or not hasattr(self.Solution, "u2"):
            return np.zeros(self.game.nu)

        u = np.concatenate((self.Solution.u1[self.Solution.indx], self.Solution.u2[self.Solution.indx]))

        return u

    def affine_lstsq_weights(self, Sx, x0, reg=1e-10):
        """
        Solve:
            min_a ||Sx.T @ a - x0||^2
            s.t. sum(a) = 1

        Sx: shape (M, d), rows are data points
        x0: shape (d,)
        """

        Sx = np.asarray(Sx, dtype=float)
        x0 = np.asarray(x0, dtype=float).reshape(-1)

        M, d = Sx.shape
        A = Sx.T  # shape (d, M)

        H = A.T @ A + reg * np.eye(M)
        f = A.T @ x0

        KKT = np.block([
            [H, np.ones((M, 1))],
            [np.ones((1, M)), np.zeros((1, 1))]
        ])

        rhs = np.concatenate([f, np.array([1.0])])

        sol = np.linalg.solve(KKT, rhs)

        a = sol[:M]

        x_rec = A @ a
        err = np.linalg.norm(x_rec - x0)

        return a, x_rec, err



    def convex_lstsq_weights(self, Sx, x0):
        """
        Solve:
            min_a ||Sx.T @ a - x0||^2
            s.t. sum(a) = 1
                a >= 0
        """

        Sx = np.asarray(Sx, dtype=float)
        x0 = np.asarray(x0, dtype=float).reshape(-1)

        M, d = Sx.shape
        A = Sx.T

        def cost(a):
            e = A @ a - x0
            return e @ e

        def grad(a):
            e = A @ a - x0
            return 2.0 * A.T @ e

        cons = {
            "type": "eq",
            "fun": lambda a: np.sum(a) - 1.0,
            "jac": lambda a: np.ones_like(a),
        }

        bounds = [(0.0, 1.0) for _ in range(M)]

        a0 = np.ones(M) / M

        res = minimize(
            cost,
            a0,
            jac=grad,
            bounds=bounds,
            constraints=[cons],
            method="SLSQP",
            options={"ftol": 1e-10, "maxiter": 500}
        )

        if not res.success:
            print("Optimization failed:", res.message)

        a = res.x
        a = np.clip(a, 0.0, 1.0)
        a = a / np.sum(a)

        x_rec = A @ a
        err = np.linalg.norm(x_rec - x0)

        return a, x_rec, err


    def calc_a_set(self, x0):
        eps = 1e-6
        if self.LearnedData is None or len(self.LearnedData.AnalyzedData.state) == 0:
            return 0, 0.0
        
        # Calculate the convex conbination factor of states in data set that best approximate xo:
        States = np.array(self.LearnedData.AnalyzedData.state)
        proximity_vec = np.zeros((len(self.LearnedData.AnalyzedData.state)))
        for i, state in enumerate(States):
            proximity_vec[i] = ca.bilin(self.proximity_Q, state-x0)
            # if proximity_vec[i] <= self.proximity_minval:
            #     a_vec = np.zeros((len(self.LearnedData.AnalyzedData.state)))
            #     a_vec[i] = 1
            #     return a_vec, 1.0-eps
        
        arg_sort = np.argsort(proximity_vec)
        arg_sort = arg_sort[proximity_vec[arg_sort] <= self.proximity_maxval]
        sorted_states = States[arg_sort]
        
        if  sorted_states.shape[0] <= self.game.nx:
            if sorted_states.shape[0] >= 1:
                # a_vec, x_rec, err = self.affine_lstsq_weights(sorted_states, x0)
                a_vec, x_rec, err = self.convex_lstsq_weights(sorted_states, x0)
            else:
                return 0, 0.0
        else:
            opti = ca.Opti()
            a_set = opti.variable(sorted_states.shape[0])
            # opti.minimize(ca.norm_1(ca.mtimes(sorted_states.T, a_set) - x0))
            cost = -ca.sumsqr(a_set)
            # for i in range(n_penalty):
            #     cost += 10*ca.sumsqr(a_set[-i])
            opti.minimize(cost)
            opti.subject_to(ca.sum1(a_set) == 1)
            opti.subject_to(ca.mtimes(sorted_states.T, a_set) == x0)
            opti.subject_to(a_set >= 0.0-eps)
            opti.subject_to(a_set <= 1.0+eps)
            
            # opts = {'ipopt.print_level': 1, 'print_time': 0, 'ipopt.max_iter': 250, "ipopt.mu_strategy": "adaptive"}
            opts = {'ipopt.print_level': 1, 'print_time': 0, 'ipopt.max_iter': 250, 'ipopt.tol': 1e-6}
            opti.solver('ipopt', opts)
            
            # find closest point:
            opti.set_initial(a_set, 0*np.ones(sorted_states.shape[0])/sorted_states.shape[0])
            try:
                sol = opti.solve()
            except RuntimeError as e:
                pass
                # print(f"\n[CasADi Opti FAILED]\nReason:\n{e}\n")
                # opti.debug.show_infeasibilities()
                # print("Last a_set value:", opti.debug.value(a_set))
            a_vec = opti.debug.value(a_set)
        a_vec = np.clip(a_vec, 0.0, 1.0)
        a_vec = a_vec / np.sum(a_vec)
        
        factor = np.array(ca.bilin(self.proximity_Q, ca.mtimes(sorted_states.T, a_vec)-x0)/self.proximity_minval)[0,0]
        if factor > 1.0:
            return 0, 0.0
            
        proximity_factor = a_vec@a_vec
        proximity_factor -= np.clip(10.0*(factor-1.0), 0.0, 1.0)
        
        a_vec1 = np.zeros((len(self.LearnedData.AnalyzedData.state)))
        for j, isort in enumerate(arg_sort[0:sorted_states.shape[0]]):
            a_vec1[isort] = a_vec[j]
            
        if self.constraint_mode == "sampled_points":
            if proximity_factor > 1.0-1e-8:
                return a_vec1, proximity_factor
            else:
                return 0.0, 0.0
        
        return a_vec1, proximity_factor
        
        

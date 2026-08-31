import numpy as np
from scipy.optimize import minimize

from Game import GameDynamics

import scipy as sp
import casadi as ca

import os
import pathlib
import copy
import pickle
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


def solution_has_no_interaction(solution, tolerance=1e-8):
    """Return whether all shared-constraint multipliers are effectively zero."""
    sigma = getattr(solution, "sigma", None)
    if sigma is None:
        return False
    sigma = np.asarray(sigma, dtype=float).reshape(-1)
    return sigma.size == 0 or np.all(np.abs(sigma) <= tolerance)


def select_nash_bargaining_result(candidate_results, disagreement_costs, cost_tol = 1e-1):
    """Return the individually rational candidate with largest Nash product.

    Candidate tuples use the internal layout ``(z_index, gamma, C1, C2, ...)``.
    ``None`` is returned when the individually rational set is empty.

    If one player cannot obtain a strictly positive improvement from any
    acceptable agreement, the other player chooses its minimum-cost agreement.
    This handles a zero Nash product without discarding useful cooperation.
    """
    baseline = np.asarray(disagreement_costs, dtype=float).reshape(-1)
    if baseline.shape != (2,) or not np.all(np.isfinite(baseline)):
        raise ValueError("disagreement_costs must be two finite costs (b1_t, b2_t)")
    acceptable = [
        result for result in candidate_results
        if result[2] <= baseline[0] + cost_tol
        and result[3] <= baseline[1] + cost_tol
    ]
    if not acceptable:
        return None

    improvement_tolerance = 0.0
    improvements = np.asarray(
        [
            (
                max(0.0, baseline[0] - result[2]),
                max(0.0, baseline[1] - result[3]),
            )
            for result in acceptable
        ]
    )
    player1_can_improve = np.max(improvements[:, 0]) > improvement_tolerance
    player2_can_improve = np.max(improvements[:, 1]) > improvement_tolerance

    if not player1_can_improve and player2_can_improve:
        return min(acceptable, key=lambda result: (result[3], result[2]))
    if not player2_can_improve and player1_can_improve:
        return min(acceptable, key=lambda result: (result[2], result[3]))
    if not player1_can_improve and not player2_can_improve:
        return min(acceptable, key=lambda result: (result[2] + result[3], result[2]))

    return max(
        acceptable,
        key=lambda result: (
            max(0.0, baseline[0] - result[2] + 1e-12)
            * max(0.0, baseline[1] - result[3] + 1e-12),
            max(0.0, baseline[0] - result[2] + 1e-12),
            + max(0.0, baseline[1] - result[3] + 1e-12),
            -result[2] - result[3],
        ),
    )


def select_convex_cost_result(candidate_results, cost_weights=(0.5, 0.5)):
    """Return the candidate minimizing a convex sum of the players' costs.

    Candidate tuples use the internal layout ``(z_index, gamma, C1, C2, ...)``.
    ``cost_weights`` must contain two nonnegative values that sum to one.
    """
    weights = np.asarray(cost_weights, dtype=float).reshape(-1)
    if (
        weights.shape != (2,)
        or not np.all(np.isfinite(weights))
        or np.any(weights < 0.0)
        or not np.isclose(np.sum(weights), 1.0)
    ):
        raise ValueError(
            "cost_weights must be two finite nonnegative values that sum to 1"
        )
    if not candidate_results:
        return None
    return min(
        candidate_results,
        key=lambda result: (
            weights[0] * result[2] + weights[1] * result[3],
            result[2] + result[3],
            result[2],
        ),
    )


def filter_monotonic_cost_candidates(
    candidate_results,
    executed_costs,
    previous_iteration_costs,
    tolerance=1e-1,
):
    """Keep candidates that do not worsen either player's prior total cost."""
    executed = np.asarray(executed_costs, dtype=float).reshape(-1)
    previous = np.asarray(previous_iteration_costs, dtype=float).reshape(-1)
    if executed.shape != (2,) or previous.shape != (2,):
        raise ValueError("executed and previous iteration costs must each contain two values")
    if not (np.all(np.isfinite(executed)) and np.all(np.isfinite(previous))):
        raise ValueError("monotonic cost limits require finite costs")
    return [
        result for result in candidate_results
        if executed[0] + result[2] <= previous[0] + tolerance
        and executed[1] + result[3] <= previous[1] + tolerance
    ]

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


def _casadi_pickle(value):
    """Serialize a process payload while CasADi symbolic pickling is enabled."""
    pickle_context = getattr(ca, "global_pickle_context", None)
    if pickle_context is None:
        raise RuntimeError(
            "parallel symbolic solves require a CasADi version that provides "
            "global_pickle_context"
        )
    with pickle_context():
        return pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)


def _casadi_unpickle(payload):
    """Deserialize a process payload while CasADi symbolic loading is enabled."""
    unpickle_context = getattr(ca, "global_unpickle_context", None)
    if unpickle_context is None:
        raise RuntimeError(
            "parallel symbolic solves require a CasADi version that provides "
            "global_unpickle_context"
        )
    with unpickle_context():
        return pickle.loads(payload)


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
    sample_number,
    sample_count,
    t,
    x0,
    forced_alpha,
    u1_0,
    u2_0,
    a_set,
    proximity_factor,
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
            precomputed_a_set=(a_set, proximity_factor),
            sample_number=sample_number,
            sample_count=sample_count,
        )
        if not solver.last_solve_success:
            return sample_index, forced_alpha, None, None, None, None
        cost1 = solver._player1_cost(solver.Solution, candidate_data)
        cost2 = solver._player2_cost(solver.Solution, candidate_data)
        return sample_index, forced_alpha, cost1, cost2, solver.Solution, None
    except Exception as exc:
        return sample_index, forced_alpha, None, None, None, f"{type(exc).__name__}: {exc}"


def _solve_sampled_terminal_gamma_sequence(
    worker_solver,
    candidate_data,
    sample_index,
    sample_number_start,
    sample_count,
    t,
    x0,
    gammas,
    u1_0,
    u2_0,
    a_set,
    proximity_factor,
):
    """Solve gamma values for one terminal state, stopping at zero interaction."""
    solver = copy.copy(worker_solver)
    initial_solution = copy.deepcopy(worker_solver.Solution)
    solver.Solver = None
    terminal_solver = None
    results = []
    for gamma_offset, gamma in enumerate(gammas):
        solver.Solution = copy.deepcopy(initial_solution)
        try:
            solver._step_once(
                t,
                x0,
                forced_alpha=gamma,
                u1_0=u1_0,
                u2_0=u2_0,
                terminal_learned_data=candidate_data,
                terminal_solver=terminal_solver,
                precomputed_a_set=(a_set, proximity_factor),
                sample_number=sample_number_start + gamma_offset,
                sample_count=sample_count,
            )
            if terminal_solver is None:
                terminal_solver = solver.Solver
            if solver.last_solve_success:
                result = (
                    sample_index,
                    gamma,
                    solver._player1_cost(solver.Solution, candidate_data),
                    solver._player2_cost(solver.Solution, candidate_data),
                    copy.deepcopy(solver.Solution),
                    None,
                )
            else:
                result = (sample_index, gamma, None, None, None, None)
        except Exception as exc:
            result = (
                sample_index, gamma, None, None, None,
                f"{type(exc).__name__}: {exc}",
            )
        results.append(result)
        solution = result[4]
        if solution is not None and solution_has_no_interaction(
            solution, worker_solver.sigma_zero_tolerance
        ):
            solution.gamma_independent = True
            solution.skipped_bargaining_gammas = np.asarray(
                gammas[gamma_offset + 1:], dtype=float
            )
            break
    return results


def _solve_sampled_terminal_gamma_sequence_serialized(payload):
    """Run a sampled-terminal task using CasADi-safe byte serialization."""
    arguments = _casadi_unpickle(payload)
    results = _solve_sampled_terminal_gamma_sequence(*arguments)
    return _casadi_pickle(results)

class DGSolver:
    """Basic structure for a dynamic game solver."""

    def __init__(self, game: GameDynamics, x1f, x2f, x3f=None,
                       dt=0.1, horizon=10, 
                       alpha=0.5,
                       R1 = 0.05,
                       R2 = 0.05,
                       LearnedData = None, 
                       p_tol=1e-4,
                       prev_best_cost=None,
                       max_workers = 1,
                       verbose = False, 
                       options=None,
                       constraint_mode="sampled_points",
                       cooperative=False,
                       bargaining_gammas=None,
                       cooperative_selection="nash_bargaining",
                       cooperative_cost_weights=None,
                       disagreement_costs=None,
                       sigma_zero_tolerance=1e-8):
        if horizon <= 0:
            raise ValueError("horizon must be positive")

        self.game = game
        self.x1f = x1f
        self.x2f = x2f
        self.x3f = x3f
        self.targets = [
            np.asarray(target, dtype=float).reshape(-1)
            for target in (x1f, x2f, x3f) if target is not None
        ]
        if len(self.targets) != game.n_players:
            raise ValueError("one target is required for every player")
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
        self.cooperative = bool(cooperative)
        valid_cooperative_selections = {"nash_bargaining", "weighted_sum"}
        if cooperative_selection not in valid_cooperative_selections:
            raise ValueError(
                "cooperative_selection must be 'nash_bargaining' or 'weighted_sum'"
            )
        self.cooperative_selection = cooperative_selection
        if cooperative_cost_weights is None:
            cooperative_cost_weights = np.full(
                self.game.n_players, 1.0 / self.game.n_players
            )
        self.cooperative_cost_weights = np.asarray(
            cooperative_cost_weights, dtype=float
        ).reshape(-1)
        if (
            self.cooperative_cost_weights.shape != (self.game.n_players,)
            or not np.all(np.isfinite(self.cooperative_cost_weights))
            or np.any(self.cooperative_cost_weights < 0.0)
            or not np.isclose(np.sum(self.cooperative_cost_weights), 1.0)
        ):
            raise ValueError(
                "cooperative_cost_weights must contain one finite nonnegative value per player "
                "that sum to 1"
            )
        if bargaining_gammas is None:
            bargaining_gammas = (
                np.array([[0.5, 0.25]]) if self.game.n_players == 3
                else np.linspace(0.1, 0.9, 9)
            )
        bargaining_gammas = np.asarray(bargaining_gammas, dtype=float)
        if self.game.n_players == 3:
            self.bargaining_gammas = bargaining_gammas.reshape(-1, 2)
            if (
                self.bargaining_gammas.shape[0] == 0
                or np.any(self.bargaining_gammas < 0.0)
                or np.any(np.sum(self.bargaining_gammas, axis=1) > 1.0 + 1e-12)
            ):
                raise ValueError(
                    "three-player bargaining weights must satisfy alpha1 >= 0, "
                    "alpha2 >= 0, and alpha1 + alpha2 <= 1"
                )
        else:
            self.bargaining_gammas = bargaining_gammas.reshape(-1)
            if self.bargaining_gammas.size == 0 or np.any(
                (self.bargaining_gammas < 0.0) | (self.bargaining_gammas > 1.0)
            ):
                raise ValueError("bargaining_gammas must contain values in [0, 1]")
        self.disagreement_costs = disagreement_costs
        self.sigma_zero_tolerance = float(sigma_zero_tolerance)
        if self.sigma_zero_tolerance < 0.0:
            raise ValueError("sigma_zero_tolerance must be nonnegative")
        alpha = np.asarray(alpha, dtype=float).reshape(-1)
        if self.game.n_players == 3:
            if alpha.size == 1:
                alpha = np.array([alpha[0], (1.0 - alpha[0]) / 2.0])
            if alpha.shape != (2,) or np.any(alpha < 0.0) or np.sum(alpha) > 1.0 + 1e-12:
                raise ValueError("alpha must contain valid (alpha1, alpha2) weights")
            self.alpha_vec = np.tile(alpha, (self.N + 1, 1))
        else:
            if alpha.shape != (1,) or not 0.0 <= alpha[0] <= 1.0:
                raise ValueError("alpha must be in [0, 1]")
            self.alpha_vec = alpha[0] * np.ones((self.N + 1, 1))
        self.max_workers = max_workers
        self.options = options.copy() if options is not None else {}
        self.solver = None
        self.is_built = False
        self._sampled_solver_cache = {}
        
        self.Qk = np.diag([1.0, 1.0]) if self.game.is_single_integrator else np.diag([1.0, 1.0, 0.1, 0.1])
        self.R1 = R1
        self.R2 = R2
        self.p_tol = p_tol
        self.verbose = verbose
        self.nms = True
        self.use_slack = False
        self.cost_tol = 1e-1
        
        per_player_proximity = ([1.0, 1.0] if self.game.is_single_integrator
                                else [1.0, 1.0, 5.0, 5.0])
        self.proximity_Q = (1 / self.game.nx) * np.diag(
            per_player_proximity * self.game.n_players
        )
        per_player_dx = ([1e-3, 1e-3] if self.game.is_single_integrator
                         else [1e-2, 1e-2, 1e-3, 1e-3])
        self.small_dx = np.asarray(per_player_dx * self.game.n_players)
        self.large_dx = 20 * self.small_dx
        self.proximity_minval = np.array(ca.bilin(self.proximity_Q, self.small_dx)).flatten()[0]
        self.proximity_maxval = np.array(ca.bilin(self.proximity_Q, self.large_dx)).flatten()[0]
        
        x1 = ca.SX.sym('x1', self.game.nx1)
        x2 = ca.SX.sym('x2', self.game.nx2)
        u1 = ca.SX.sym('u1', self.game.nu1)
        u2 = ca.SX.sym('u2', self.game.nu2)

        time1_to_target = ca.if_else(ca.bilin(self.Qk, x1-self.x1f.T) <= self.proximity_minval, 0.0, 1.0)
        time2_to_target = ca.if_else(ca.bilin(self.Qk, x2-self.x2f.T) <= self.proximity_minval, 0.0, 1.0)
        
        self.l1 = ca.Function('l1', [x1, u1, x2, u2], [ca.bilin(self.Qk, x1-self.x1f.T) + ca.bilin(self.R1*np.eye(self.game.nu1), u1)+time1_to_target - 0.0*(ca.bilin(self.Qk, x2-self.x2f.T) - ca.bilin(self.R2*np.eye(self.game.nu2), u2)-time2_to_target)])
        self.l2 = ca.Function('l2', [x2, u2, x1, u1], [ca.bilin(self.Qk, x2-self.x2f.T) + ca.bilin(self.R2*np.eye(self.game.nu2), u2)+time2_to_target - 0.0*(ca.bilin(self.Qk, x1-self.x1f.T) - ca.bilin(self.R1*np.eye(self.game.nu1), u1)-time1_to_target)])

        self.stage_costs = []
        for player, target in enumerate(self.targets):
            xp = ca.SX.sym(f'cost_x{player + 1}', self.game.nx1)
            up = ca.SX.sym(f'cost_u{player + 1}', self.game.nu1)
            time_to_target = ca.if_else(
                ca.bilin(self.Qk, xp - target) <= self.proximity_minval,
                0.0, 1.0,
            )
            self.stage_costs.append(ca.Function(
                f'player_{player + 1}_stage_cost', [xp, up],
                [ca.bilin(self.Qk, xp - target)
                 + ca.bilin((self.R1 if player != 1 else self.R2)
                            * np.eye(self.game.nu1), up)
                 + time_to_target],
            ))
        if self.game.n_players == 3:
            x3 = ca.SX.sym('x3', self.game.nx1)
            u3 = ca.SX.sym('u3', self.game.nu1)
            self.l3 = ca.Function(
                'l3', [x3, u3], [self.stage_costs[2](x3, u3)]
            )

        
        self.Solution = SimpleNamespace()
        self.Solution.success = False
        self.Solution.terminal_sample_state = None
        self.Solution.player1_predicted_cost = prev_best_cost
        self.last_solve_success = False
        
        if game.iteration > 1 and LearnedData is not None:
            self.backup_controller_init()

    def _learned_player2_action(self, a_set):
        """Return the saved player-2 action only in non-cooperative mode."""
        if self.cooperative or self.LearnedData is None:
            return None
        if abs(np.sum(a_set) - 1.0) < 1e-5:
            return a_set @ self.LearnedData.AnalyzedData.u2
        return None

    def build_solver(self, u2_0 = None, Terminal_Safe_Set = None):
        """
        Build the dynamic game solver.

        This is a placeholder for constructing optimization variables,
        constraints, costs, and the numerical backend.
        """
        if self.game.n_players == 3:
            return self._build_three_player_solver(Terminal_Safe_Set)

        # In cooperative mode both controls are chosen by Solver1.  A player-2
        # action stored with the learned safe set must therefore never pin the
        # first action of the new trajectory.
        if self.cooperative:
            u2_0 = None
                
        # Player 1 trajectory variables over the horizon.
        x1 = ca.SX.sym('x1',self.N+1, self.game.nx1)
        u1 = ca.SX.sym('u1',self.N, self.game.nu1)
        x1_0 = ca.SX.sym('x1_0',1, self.game.nx1)
        
        # Player 2 trajectory variables over the horizon.
        x2 = ca.SX.sym('x2', self.N+1, self.game.nx2)
        u2 = ca.SX.sym('u2', self.N, self.game.nu2)
        x2_0 = ca.SX.sym('x2_0', 1, self.game.nx2)
        
        alpha_vec = ca.SX.sym('alpha_vec', self.N+1)
        # The terminal state is a convex combination of the sampled dataset, with weights ai_xf.
        if self.use_slack and Terminal_Safe_Set is not None and Terminal_Safe_Set.state.shape[0]:
            ai_xf = ca.SX.sym('ai_xf', Terminal_Safe_Set.state.shape[0])
            x1f_slack = ca.SX.sym('x1f_slack', self.game.nx1, 1)
            x2f_slack = ca.SX.sym('x2f_slack', self.game.nx2, 1)
        else:
            ai_xf = []
            x1f_slack = []
            x2f_slack = []

        A1, B1 = self._discrete_player_dynamics(self.game.nx1)

        # Store each player's equality constraints, private constraints, and multipliers.
        h_vec, mu_vec = [], []
        p_vec, lambda_vec = [], []
        sg_vec = []
        # Define The first player lagrangian:
        L1 = 0
        for k in range(self.N):
            L1 += self.l1(x1[k,:], u1[k,:], x2[k,:], u2[k,:])
            
        if Terminal_Safe_Set is not None:
            if Terminal_Safe_Set.state.shape[0] > 1:
                L1 += ca.mtimes(Terminal_Safe_Set.Cost2Go.reshape(1,-1), ai_xf)
            else:
                L1 += Terminal_Safe_Set.Cost2Go
        else:
            L1 += self.l1(x1[self.N,:], np.zeros_like(u1[0,:].shape), x2[self.N,:], np.zeros_like(u2[0,:].shape))
        # L1 += 1e8*ca.sumsqr(x1f_slack)
            
        # Player 1 Dynamics:
        h = []
        n_mu = 0
        for k in range(self.N+1):
            if k == 0:
                h.append(x1[k,:].T - x1_0.T)
            else:
                h.append(x1[k,:].T - A1@x1[k-1,:].T - B1@u1[k-1,:].T)
            n_mu += h[-1].shape[0]

        # Final joint state is a convex combination of the sampled dataset
        if Terminal_Safe_Set is not None:
            if Terminal_Safe_Set.state.shape[0] > 1:
                h.append(1.0 - ca.sum1(ai_xf))
                n_mu += h[-1].shape[0]
                h.append(ca.mtimes(Terminal_Safe_Set.state.T[:self.game.nx1,:], ai_xf) - x1[self.N,:].T)
            elif self.use_slack:
                h.append(Terminal_Safe_Set.state.T[:self.game.nx1,:] - x1[self.N,:].T + x1f_slack)
            else:
                h.append(Terminal_Safe_Set.state.T[:self.game.nx1,:] - x1[self.N,:].T)
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
                    ])
        if self.use_slack and Terminal_Safe_Set is not None:
            p1.extend([1.0e-8 - x1f_slack**2])
                
        # Final joint state is a convex combination of the smapled dataset
        if Terminal_Safe_Set is not None:
            if Terminal_Safe_Set.state.shape[0] > 1:
                p1.append(1.0 - ai_xf)
                p1.append(ai_xf)

        p1_ph = ca.vertcat(*p1)
        lambda_1 = ca.SX.sym("lambda_1", p1_ph.shape[0])
        L1 -= ca.dot(lambda_1, p1_ph)
        p_vec.append(p1_ph)
        lambda_vec.append(lambda_1)

        # Second Player Lagrangian and Constraints:
        A2, B2 = self._discrete_player_dynamics(self.game.nx2)
        
        # Define the second player lagrangian using the same quadratic structure.
        L2 = 0
        for k in range(self.N):
            L2 += self.l2(x2[k, :], u2[k, :], x1[k, :], u1[k, :])
        if Terminal_Safe_Set is not None and hasattr(Terminal_Safe_Set, "Cost2Go2"):
            terminal_cost2 = np.asarray(
                Terminal_Safe_Set.Cost2Go2, dtype=float
            ).reshape(-1)
            if terminal_cost2.size > 1:
                L2 += ca.mtimes(terminal_cost2.reshape(1, -1), ai_xf)
            elif terminal_cost2.size == 1:
                L2 += float(terminal_cost2[0])
        else:
            L2 += self.l2(x2[self.N, :], np.zeros_like(u2[0, :].shape), x1[self.N, :], np.zeros_like(u1[0, :].shape))
        # L2 += 1e8*ca.sumsqr(x2f_slack)

        # Player 2 dynamics are equality constraints enforced by mu_2.
        h = []
        n_mu = 0
        for k in range(self.N + 1):
            if k == 0:
                h.append(x2[k, :].T - x2_0.T)
            else:
                h.append(x2[k, :].T - A2 @ x2[k-1, :].T - B2 @ u2[k-1, :].T)
            n_mu += h[-1].shape[0]
            
        if u2_0 is not None:
            h.append(u2[0,:].T - u2_0.reshape(-1,1))
            n_mu += h[-1].shape[0]
            
        # The final state must be a convex combination of the sampled dataset
        if Terminal_Safe_Set is not None:
            if Terminal_Safe_Set.state.shape[0] > 1:
                h.append(ca.mtimes(Terminal_Safe_Set.state.T[self.game.nx1:,:], ai_xf) - x2[self.N,:].T)
            elif self.use_slack:
                h.append(Terminal_Safe_Set.state.T[self.game.nx1:,:] - x2[self.N,:].T + x2f_slack)
            else:
                h.append(Terminal_Safe_Set.state.T[self.game.nx1:,:] - x2[self.N,:].T)
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
        if self.use_slack and Terminal_Safe_Set is not None:
            p2.extend([1.0e-8 - x2f_slack**2])
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
            elif Terminal_Safe_Set is None:
                f_val_k = self.game.f_shared(ca.horzcat(x1[k,:], x2[k,:]), np.zeros_like(u1[0,:].shape), np.zeros_like(u2[0,:].shape))
            if not isinstance(f_val_k, tuple):
                f_val_k = [f_val_k]
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
        z1 = ca.vertcat(x1[:], u1[:], ai_xf[:], x1f_slack[:])
        Z.append(z1)
        Z_len.append([ca.vertcat(x1[:]).shape[0], ca.vertcat(u1[:]).shape[0], ca.vertcat(ai_xf[:]).shape[0], ca.vertcat(x1f_slack[:]).shape[0]])
        z2 = ca.vertcat(x2[:], u2[:], x2f_slack[:])
        Z.append(z2)
        Z_len.append([ca.vertcat(x2[:]).shape[0], ca.vertcat(u2[:]).shape[0], ca.vertcat(x2f_slack[:]).shape[0]])
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
        self.solver.params["lagrangians"] = [ca.Function('L1',[Z, x1_0, x2_0, alpha_vec],[L1]), ca.Function('L2',[Z, x1_0, x2_0, alpha_vec],[L2])]
        self.solver.Z = Z
        self.solver.Z_len = Z_len
        self.solver.F = ca.Function('F', [Z, x1_0, x2_0, alpha_vec], [F])
        self.solver.J = ca.Function('J', [Z, x1_0, x2_0, alpha_vec], [J])
        self.solver.n_l_inf = sum(Z_len[0]) + sum(Z_len[1]) + Z_len[2]
        self.solver.n_u_inf = self.solver.n_l_inf + int(np.sum(Z_len[3:]))

        self.is_built = True
        return self.solver

    def _build_three_player_solver(self, terminal_safe_set=None):
        """Build the same KKT/MCP formulation with a third player block."""
        player_count = self.game.n_players
        xs = [ca.SX.sym(f'x{p + 1}', self.N + 1, self.game.nx1)
              for p in range(player_count)]
        us = [ca.SX.sym(f'u{p + 1}', self.N, self.game.nu1)
              for p in range(player_count)]
        x0s = [ca.SX.sym(f'x{p + 1}_0', 1, self.game.nx1)
               for p in range(player_count)]
        alpha_vec = ca.SX.sym('alpha_vec', self.N + 1, 2)

        sample_count = (
            terminal_safe_set.state.shape[0]
            if terminal_safe_set is not None else 0
        )
        ai_xf = ca.SX.sym('ai_xf', sample_count) if sample_count > 1 else []
        slacks = [[] for _ in range(player_count)]
        if self.use_slack and sample_count:
            slacks = [ca.SX.sym(f'x{p + 1}f_slack', self.game.nx1, 1)
                      for p in range(player_count)]

        h_vec, mu_vec, p_vec, lambda_vec = [], [], [], []
        lagrangians, player_z, player_z_lengths = [], [], []
        A, B = self._discrete_player_dynamics(self.game.nx1)
        self.A_players = [A] * player_count
        self.B_players = [B] * player_count

        for player in range(player_count):
            Lp = sum(
                self.stage_costs[player](xs[player][k, :], us[player][k, :])
                for k in range(self.N)
            )
            cost_field = 'Cost2Go' if player == 0 else f'Cost2Go{player + 1}'
            if terminal_safe_set is not None and hasattr(terminal_safe_set, cost_field):
                terminal_cost = np.asarray(
                    getattr(terminal_safe_set, cost_field), dtype=float
                ).reshape(-1)
                if terminal_cost.size > 1:
                    Lp += ca.mtimes(terminal_cost.reshape(1, -1), ai_xf)
                elif terminal_cost.size == 1:
                    Lp += float(terminal_cost[0])
            else:
                Lp += self.stage_costs[player](
                    xs[player][self.N, :], ca.DM.zeros(self.game.nu1)
                )

            h = []
            for k in range(self.N + 1):
                if k == 0:
                    h.append(xs[player][k, :].T - x0s[player].T)
                else:
                    h.append(xs[player][k, :].T - A @ xs[player][k - 1, :].T
                             - B @ us[player][k - 1, :].T)
            if terminal_safe_set is not None:
                start = player * self.game.nx1
                terminal_states = terminal_safe_set.state.T[
                    start:start + self.game.nx1, :
                ]
                if sample_count > 1:
                    if player == 0:
                        h.append(1.0 - ca.sum1(ai_xf))
                    h.append(ca.mtimes(terminal_states, ai_xf)
                             - xs[player][self.N, :].T)
                elif self.use_slack:
                    h.append(terminal_states - xs[player][self.N, :].T
                             + slacks[player])
                else:
                    h.append(terminal_states - xs[player][self.N, :].T)
            hp = ca.vertcat(*h)
            mu = ca.SX.sym(f'mu_{player + 1}', hp.shape[0])
            Lp += ca.dot(mu, hp)

            private = []
            for k in range(self.N + 1):
                xp = xs[player][k, :]
                private.extend([
                    xp[0] - self.game.x_min, self.game.x_max - xp[0],
                    xp[1] - self.game.y_min, self.game.y_max - xp[1],
                ])
                if not self.game.is_single_integrator:
                    private.extend([
                        xp[2] - self.game.vx_min, self.game.vx_max - xp[2],
                        xp[3] - self.game.vy_min, self.game.vy_max - xp[3],
                    ])
                if k < self.N:
                    up = us[player][k, :]
                    private.extend([
                        up[0] - self.game.u_min, self.game.u_max - up[0],
                        up[1] - self.game.u_min, self.game.u_max - up[1],
                    ])
            if self.use_slack and terminal_safe_set is not None:
                private.extend([1e-8 - slacks[player] ** 2])
            if player == 0 and sample_count > 1:
                private.extend([1.0 - ai_xf, ai_xf])
            pp = ca.vertcat(*private)
            lam = ca.SX.sym(f'lambda_{player + 1}', pp.shape[0])
            Lp -= ca.dot(lam, pp)

            x_flat = ca.vec(xs[player])
            u_flat = ca.vec(us[player])
            z_components = [x_flat, u_flat]
            ai_length = 0
            if player == 0 and sample_count > 1:
                z_components.append(ca.vec(ai_xf))
                ai_length = int(ai_xf.shape[0])
            slack_length = 0
            if self.use_slack and sample_count:
                z_components.append(ca.vec(slacks[player]))
                slack_length = int(ca.vertcat(slacks[player][:]).shape[0])
            zp = ca.vertcat(*z_components)
            player_z.append(zp)
            player_z_lengths.append([
                x_flat.shape[0],
                u_flat.shape[0],
                ai_length,
                slack_length,
            ])
            h_vec.append(hp)
            mu_vec.append(mu)
            p_vec.append(pp)
            lambda_vec.append(lam)
            lagrangians.append(Lp)

        shared, shared_stages = [], []
        for k in range(self.N + 1):
            controls = ([us[p][k, :] for p in range(player_count)]
                        if k < self.N else
                        [ca.DM.zeros(self.game.nu1) for _ in range(player_count)])
            values = self.game.f_shared(ca.horzcat(*[x[k, :] for x in xs]), *controls)
            if not isinstance(values, tuple):
                values = (values,)
            for value in values:
                if is_symbolic_expr(value):
                    shared.append(value)
                    shared_stages.append(k)
        sg_vec = ca.vertcat(*shared)
        alpha1_k = ca.vertcat(*[alpha_vec[k, 0] for k in shared_stages])
        alpha2_k = ca.vertcat(*[alpha_vec[k, 1] for k in shared_stages])
        alpha3_k = 1.0 - alpha1_k - alpha2_k
        sigma = ca.SX.sym('sigma', sg_vec.shape[0])
        alpha_weights = (alpha1_k, alpha2_k, alpha3_k)
        for player, alpha_weight in enumerate(alpha_weights):
            lagrangians[player] -= ca.dot(alpha_weight * sigma, sg_vec)

        mu_all = ca.vertcat(*mu_vec)
        lambda_all = ca.vertcat(*lambda_vec)
        z_parts = [*player_z, mu_all, lambda_all, sigma]
        Z = ca.vertcat(*z_parts)
        Z_len = [*player_z_lengths, mu_all.shape[0],
                 lambda_all.shape[0], sigma.shape[0]]
        stationarity = [
            ca.jacobian(lagrangians[p], player_z[p]).T
            for p in range(player_count)
        ]
        F_expr = ca.vertcat(*stationarity, *h_vec, *p_vec, sg_vec)
        J_expr = ca.jacobian(F_expr, Z)
        function_args = [Z, *x0s, alpha_vec]

        self.solver = SimpleNamespace()
        self.solver.params = {
            'nx': self.game.nx, 'nu': self.game.nu, 'horizon': self.N,
            'dynamics_type': self.game.dynamics_type, 'options': self.options,
            'lagrangians': lagrangians,
        }
        self.solver.Z = Z
        self.solver.Z_len = Z_len
        self.solver.F = ca.Function('F3', function_args, [F_expr])
        self.solver.J = ca.Function('J3', function_args, [J_expr])
        self.solver.n_l_inf = sum(sum(lengths) for lengths in player_z_lengths) + int(mu_all.shape[0])
        self.solver.n_u_inf = self.solver.n_l_inf + int(lambda_all.shape[0]) + int(sigma.shape[0])
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

    def step(self, t, x0, current_cost1=0.0, current_cost2=0.0,
             current_cost3=0.0,
             forced_alpha=None, u1_0=None, u2_0=None,
             last_attempted_solution=False, use_all_terminal_points=False,
             disagreement_costs=None, previous_iteration_costs=None):
        """Solve one step using the configured learned terminal-state mode."""
        if self.constraint_mode == "sampled_points" and self.LearnedData is not None:
            return self._step_over_sampled_terminal_states(
                t, x0, current_cost1=current_cost1, current_cost2=current_cost2,
                current_cost3=current_cost3,
                forced_alpha=forced_alpha, u1_0=u1_0, u2_0=u2_0,
                last_attempted_solution=last_attempted_solution,
                use_all_terminal_points=use_all_terminal_points,
                disagreement_costs=disagreement_costs,
                previous_iteration_costs=previous_iteration_costs,
            )
        control = self._step_once(
            t,
            x0,
            forced_alpha=forced_alpha,
            u1_0=u1_0,
            u2_0=u2_0,
            last_attempted_solution=last_attempted_solution,
        )
        if not self.last_solve_success and hasattr(self, "backup"):
            return self.backup_controller(x0)
        return control

    def _step_over_sampled_terminal_states(
        self, t, x0, current_cost1=0.0, current_cost2=0.0,
        current_cost3=0.0,
        forced_alpha=None, u1_0=None, u2_0=None,
        last_attempted_solution=False, use_all_terminal_points=False,
        disagreement_costs=None, previous_iteration_costs=None,
    ):
        """Enumerate safe-set states and, in cooperative mode, bargaining weights."""
        analyzed = self.LearnedData.AnalyzedData
        states = np.asarray(analyzed.state)
        Cost2Go = np.asarray(analyzed.Cost2Go)
        Cost2Go2 = np.asarray(analyzed.Cost2Go2)
        sample_times = np.asarray(analyzed.t)
        previous_solution = copy.deepcopy(self.Solution)
        terminal_sample_index = getattr(previous_solution, "terminal_sample_index", -1)
        prev_cost2go = Cost2Go[terminal_sample_index]+10.0 if terminal_sample_index >= 0 else np.inf
        prev_cost2go2 = Cost2Go2[terminal_sample_index]+10.0 if terminal_sample_index >= 0 else np.inf
        a_set, proximity_factor = self.calc_a_set(x0)
        previous_sample_time = getattr(previous_solution, "terminal_sample_time", 0.0)
        distance_to_terminal = np.linalg.norm(states[:,:2] - x0[:2], axis=1)
        if not use_all_terminal_points:
            if self.cooperative:
                cost_filter = (Cost2Go <= prev_cost2go + self.cost_tol) & (Cost2Go2 <= prev_cost2go2 + self.cost_tol)
            else:
                cost_filter = (Cost2Go <= prev_cost2go + self.cost_tol) 
                
            candidate_indices = np.where(
                cost_filter
                & (sample_times <= previous_sample_time + (1.5 * self.N) * self.dt)
                & (distance_to_terminal <= np.sqrt(2) * self.game.vx_max * self.N * self.dt)
                & (sample_times > t + (self.N-1) * self.dt - 1e-5)
                
            )[0]
        else:
            candidate_indices = np.where( (sample_times >= t-self.N*self.dt) )[0]
        if candidate_indices.shape[0]==0:
            candidate_indices = np.where((states[:,0] == self.game.x1f[0,0]) & (states[:,1] == self.game.x2f[0,1]))[0]
        candidate_indices = np.asarray(candidate_indices, dtype=int)
        candidate_terminal_states = states[candidate_indices].copy()
        previous_solver = getattr(self, "Solver", None)
        best_solution = None
        best_solver = None
        best_cost = np.inf
        candidate_data_by_index = {}
        for sample_index in candidate_indices:
            candidate_data = copy.deepcopy(self.LearnedData)
            candidate = candidate_data.AnalyzedData
            fields = ("t", "state", "Cost2Go", "Cost2Go2", "Cost2Go3")
            if not self.cooperative:
                fields += ("u2",)
            for field in fields:
                if not hasattr(analyzed, field):
                    continue
                values = np.asarray(getattr(analyzed, field))
                setattr(candidate, field, values[[sample_index]])
            if self.cooperative and hasattr(candidate, "u2"):
                candidate.u2 = None
            candidate.n_data = 1
            candidate_data_by_index[int(sample_index)] = candidate_data

        candidate_results = []
        if forced_alpha is not None:
            forced = np.asarray(forced_alpha, dtype=float).reshape(-1)
            gammas = [forced if self.game.n_players == 3 else float(forced[0])]
        elif self.cooperative:
            gammas = list(self.bargaining_gammas)
        else:
            gammas = [None]
        gammas = [
            (None if gamma is None else
             np.asarray(gamma, dtype=float).reshape(2)
             if self.game.n_players == 3 else float(gamma))
            for gamma in gammas
        ]
        sample_count = len(candidate_data_by_index) * len(gammas)
        if self.max_workers == 1:
            sample_number = 0
            for sample_index, candidate_data in candidate_data_by_index.items():
                for gamma_offset, gamma in enumerate(gammas):
                    sample_number += 1
                    self.Solution = copy.deepcopy(previous_solution)
                    cache_key = int(sample_index)
                    candidate_solver = self._sampled_solver_cache.get(cache_key)
                    self._step_once(
                        t,
                        x0,
                        forced_alpha=gamma,
                        u1_0=u1_0,
                        u2_0=u2_0,
                        last_attempted_solution=last_attempted_solution,
                        terminal_learned_data=candidate_data,
                        terminal_solver=candidate_solver,
                        precomputed_a_set=(a_set, proximity_factor),
                        sample_number=sample_number,
                        sample_count=sample_count,
                    )
                    if candidate_solver is None:
                        self._sampled_solver_cache[cache_key] = self.Solver
                    if not self.last_solve_success:
                        continue

                    candidate_solution = copy.deepcopy(self.Solution)
                    no_interaction = solution_has_no_interaction(
                        candidate_solution, self.sigma_zero_tolerance
                    )
                    if no_interaction:
                        candidate_solution.gamma_independent = True
                        candidate_solution.skipped_bargaining_gammas = np.asarray(
                            gammas[gamma_offset + 1:], dtype=float
                        )
                    candidate_results.append(
                        (
                            sample_index,
                            gamma,
                            self._player1_cost(candidate_solution, candidate_data),
                            self._player2_cost(candidate_solution, candidate_data),
                            candidate_solution,
                            self.Solver,
                        )
                    )
                    if no_interaction:
                        sample_number += len(gammas) - gamma_offset - 1
                        break
        elif self.max_workers > 1:
            worker_solver = copy.copy(self)
            worker_solver.Solution = copy.deepcopy(previous_solution)
            worker_solver.Solver = None
            worker_solver.solver = None
            worker_solver.is_built = False
            executor = _get_terminal_executor(self.max_workers)
            futures = {}
            for sample_offset, (sample_index, candidate_data) in enumerate(
                candidate_data_by_index.items()
            ):
                payload = _casadi_pickle(
                    (
                        worker_solver,
                        candidate_data,
                        sample_index,
                        sample_offset * len(gammas) + 1,
                        sample_count,
                        t,
                        x0,
                        gammas,
                        u1_0,
                        u2_0,
                        a_set,
                        proximity_factor,
                    )
                )
                future = executor.submit(
                    _solve_sampled_terminal_gamma_sequence_serialized,
                    payload,
                )
                futures[future] = sample_index
            for future in as_completed(futures):
                submitted_index = futures[future]
                try:
                    results = _casadi_unpickle(future.result())
                except Exception as exc:
                    if self.verbose:
                        print(
                            f"Terminal sample {submitted_index} worker failed: "
                            f"{type(exc).__name__}: {exc}"
                        )
                    continue
                for sample_index, gamma, cost1, cost2, solution, error in results:
                    if error is not None:
                        if self.verbose:
                            print(
                                f"Terminal sample {sample_index}, gamma={gamma} failed: {error}"
                            )
                        continue
                    if solution is not None:
                        candidate_results.append(
                            (sample_index, gamma, cost1, cost2, solution, None)
                        )

        monotonic_limits = None
        if self.cooperative and previous_iteration_costs is not None:
            monotonic_limits = np.asarray(
                previous_iteration_costs, dtype=float
            ).reshape(-1)
            if self.game.n_players == 2:
                candidate_results = filter_monotonic_cost_candidates(
                    candidate_results,
                    (current_cost1, current_cost2),
                    monotonic_limits,
                )

        baseline = None
        if self.cooperative and candidate_results:
            if self.game.n_players == 3:
                for result in candidate_results:
                    result[4].player3_cost = self._player3_cost(
                        result[4], candidate_data_by_index[int(result[0])]
                    )
                if monotonic_limits is not None:
                    executed = np.array([current_cost1, current_cost2, current_cost3])
                    if monotonic_limits.shape != (3,):
                        raise ValueError("previous_iteration_costs must contain three costs")
                    candidate_results = [
                        result for result in candidate_results
                        if np.all(executed + np.array([
                            result[2], result[3], result[4].player3_cost
                        ]) <= monotonic_limits + self.cost_tol)
                    ]
                    if not candidate_results:
                        selected = None
                        baseline = None
                        # Continue through the common rejection path.
                if not candidate_results:
                    selected = None
                elif self.cooperative_selection == "nash_bargaining":
                    baseline = disagreement_costs if disagreement_costs is not None else self.disagreement_costs
                    if baseline is None:
                        baseline = np.max([
                            [r[2], r[3], r[4].player3_cost]
                            for r in candidate_results
                        ], axis=0)
                    baseline = np.asarray(baseline, dtype=float).reshape(-1)
                    if baseline.shape != (3,) or not np.all(np.isfinite(baseline)):
                        raise ValueError("disagreement_costs must contain three finite costs")
                    acceptable = [
                        r for r in candidate_results
                        if np.all(np.array([r[2], r[3], r[4].player3_cost]) <= baseline + self.cost_tol)
                    ]
                    selected = max(
                        acceptable,
                        key=lambda r: np.prod(np.maximum(
                            baseline - np.array([r[2], r[3], r[4].player3_cost]), 0.0
                        )),
                    ) if acceptable else None
                elif candidate_results:
                    selected = min(
                        candidate_results,
                        key=lambda r: self.cooperative_cost_weights @ np.array(
                            [r[2], r[3], r[4].player3_cost]
                        ),
                    )
            elif self.cooperative_selection == "nash_bargaining":
                baseline = (
                    disagreement_costs
                    if disagreement_costs is not None
                    else self.disagreement_costs
                )
                if baseline is None:
                    # Conservative default when no policy-specific disagreement
                    # point is provided: the componentwise worst feasible outcome.
                    baseline = (
                        max(result[2] for result in candidate_results),
                        max(result[3] for result in candidate_results),
                    )
                baseline = np.asarray(baseline, dtype=float).reshape(-1)
                if baseline.shape != (2,) or not np.all(np.isfinite(baseline)):
                    raise ValueError(
                        "disagreement_costs must be two finite costs (b1_t, b2_t)"
                    )
                selected = select_nash_bargaining_result(candidate_results, baseline)
            else:
                selected = select_convex_cost_result(
                    candidate_results, self.cooperative_cost_weights
                )
            candidate_results = [] if selected is None else [selected]

        for sample_index, gamma, cost1, cost2, candidate_solution, candidate_solver in candidate_results:
            if self.cooperative or cost1 < best_cost:
                best_cost = cost1
                best_solution = candidate_solution
                best_solver = candidate_solver
                best_solution.terminal_sample_index = sample_index
                best_solution.terminal_sample_time = float(sample_times[sample_index])
                best_solution.terminal_sample_state = states[sample_index].copy()
                best_solution.player1_cost = cost1
                best_solution.player2_cost = cost2
                best_solution.player1_predicted_cost = current_cost1 + cost1
                best_solution.player2_predicted_cost = current_cost2 + cost2
                if self.game.n_players == 3:
                    cost3 = self._player3_cost(
                        candidate_solution,
                        candidate_data_by_index[int(sample_index)],
                    )
                    best_solution.player3_cost = cost3
                    best_solution.player3_predicted_cost = cost3
                if monotonic_limits is not None:
                    best_solution.previous_iteration_costs = monotonic_limits.copy()
                    predicted_costs = [
                        best_solution.player1_predicted_cost,
                        best_solution.player2_predicted_cost,
                    ]
                    if self.game.n_players == 3:
                        predicted_costs.append(current_cost3 + best_solution.player3_cost)
                    best_solution.monotonic_cost_margins = (
                        monotonic_limits - np.array(predicted_costs)
                    )
                if self.cooperative:
                    best_solution.bargaining_gamma = gamma
                    best_solution.cooperative_selection = self.cooperative_selection
                if baseline is not None:
                    best_solution.disagreement_costs = baseline.copy()
                    bargaining_costs = [cost1, cost2]
                    if self.game.n_players == 3:
                        bargaining_costs.append(best_solution.player3_cost)
                    best_solution.bargaining_improvements = np.maximum(
                        baseline - np.array(bargaining_costs), 0.0
                    )
                    best_solution.nash_product = float(np.prod(best_solution.bargaining_improvements))
                elif self.cooperative:
                    best_solution.cooperative_cost_weights = (
                        self.cooperative_cost_weights.copy()
                    )
                    selected_costs = [cost1, cost2]
                    if self.game.n_players == 3:
                        selected_costs.append(best_solution.player3_cost)
                    best_solution.cooperative_objective = float(
                        self.cooperative_cost_weights @ np.array(selected_costs)
                    )
                best_solution.terminal_workers = self.max_workers
                if self.cooperative:
                    break

        if best_solution is None:
            self.Solution = previous_solution
            self.Solver = previous_solver
            self.last_solve_success = False
            self.Solution.success = False
            if monotonic_limits is not None:
                self.Solution.monotonic_rejection = True
                self.Solution.previous_iteration_costs = monotonic_limits.copy()
            self.Solution.candidate_indices = candidate_indices.copy()
            self.Solution.candidate_terminal_states = candidate_terminal_states.copy()
            if hasattr(self, "backup"):
                return self.backup_controller(x0)
            if all(hasattr(self.Solution, f"u{p + 1}") for p in range(self.game.n_players)):
                t_vec = np.arange(self.N) * self.dt+self.Solution.t
                indx = np.argmin(np.abs(t_vec - t))
                self.Solution.indx = max(self.Solution.indx+1, indx)
                if self.Solution.indx >= self.N:
                    return np.zeros(self.game.nu)
                return np.concatenate([
                    getattr(self.Solution, f"u{p + 1}")[self.Solution.indx]
                    for p in range(self.game.n_players)
                ])
            return np.zeros(self.game.nu)

        best_solution.candidate_indices = candidate_indices.copy()
        best_solution.candidate_terminal_states = candidate_terminal_states.copy()
        self.Solution = best_solution
        if best_solver is None:
            best_solver = self.build_solver(u2_0=None, 
                Terminal_Safe_Set=candidate_data_by_index[best_solution.terminal_sample_index].AnalyzedData)
        self.Solver = best_solver
        self.last_solve_success = True
        
        self.backup_controller_update(self.Solution)
        return np.concatenate([
            getattr(best_solution, f"u{p + 1}")[0]
            for p in range(self.game.n_players)
        ])

    def _player1_cost(self, solution, learned_data):
        """Evaluate the primal player-1 objective used to rank terminal samples."""
        cost = 0.0
        target = np.asarray(self.x1f, dtype=float).reshape(-1)
        for k in range(self.N):
            cost += float(self.l1(solution.x1[k], solution.u1[k], solution.x2[k], solution.u2[k]))
        cost_to_go = np.asarray(learned_data.AnalyzedData.Cost2Go, dtype=float).reshape(-1)
        weights = np.asarray(solution.ai_xf_vec, dtype=float).reshape(-1) if solution.ai_xf_vec.shape[0]>1 else np.asarray(1, dtype=float).reshape(-1)
        return cost + float(cost_to_go @ weights)

    def _player2_cost(self, solution, learned_data):
        """Evaluate player 2's primal cost for a candidate agreement."""
        cost = 0.0
        for k in range(self.N):
            cost += float(
                self.l2(
                    solution.x2[k], solution.u2[k],
                    solution.x1[k], solution.u1[k],
                )
            )
        analyzed = learned_data.AnalyzedData
        if hasattr(analyzed, "Cost2Go2"):
            cost_to_go = np.asarray(analyzed.Cost2Go2, dtype=float).reshape(-1)
            if cost_to_go.size:
                weights = (
                    np.asarray(solution.ai_xf_vec, dtype=float).reshape(-1)
                    if solution.ai_xf_vec.shape[0] > 1 else np.ones(1)
                )
                cost += float(cost_to_go @ weights)
        return cost

    def _player3_cost(self, solution, learned_data):
        """Evaluate player 3's primal cost for a terminal candidate."""
        cost = sum(
            float(self.stage_costs[2](solution.x3[k], solution.u3[k]))
            for k in range(self.N)
        )
        terminal_costs = np.asarray(
            getattr(learned_data.AnalyzedData, "Cost2Go3", []), dtype=float
        ).reshape(-1)
        if terminal_costs.size:
            weights = (
                np.asarray(solution.ai_xf_vec, dtype=float).reshape(-1)
                if solution.ai_xf_vec.shape[0] > 1 else np.ones(1)
            )
            cost += float(terminal_costs @ weights)
        return cost

    def _step_once(
        self,
        t,
        x0,
        forced_alpha=None,
        u1_0=None,
        u2_0=None,
        last_attempted_solution=False,
        terminal_learned_data=None,
        terminal_solver=None,
        precomputed_a_set=None,
        sample_number=None,
        sample_count=None,
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
        if self.game.n_players == 3:
            return self._step_once_three_player(
                t, x0, forced_alpha=forced_alpha,
                terminal_learned_data=terminal_learned_data,
                terminal_solver=terminal_solver,
                precomputed_a_set=precomputed_a_set,
                sample_number=sample_number, sample_count=sample_count,
            )
        
        if precomputed_a_set is not None:
            a_set, proximity_factor = precomputed_a_set
        else:
            a_set, proximity_factor = self.calc_a_set(x0)
        
        u2_Learned = self._learned_player2_action(a_set)
        
        alpha_vec = self.alpha_vec
        if forced_alpha is not None:
            alpha_vec = forced_alpha * np.ones_like(self.alpha_vec)
        elif self.LearnedData is not None:
            alpha_vec[0] = np.clip(1.0-proximity_factor, 0.1, 1.0)
        
        if not self.is_built and self.LearnedData is None and u2_Learned is None:
            self.Solver = self.build_solver(u2_0 = None, Terminal_Safe_Set=None)
        if terminal_learned_data is not None:
            LearnedData1 = terminal_learned_data
            if terminal_solver is None:
                self.Solver = self.build_solver(u2_0 = u2_Learned, Terminal_Safe_Set=LearnedData1.AnalyzedData)
            else:
                self.Solver = terminal_solver
        elif self.LearnedData is not None:
            LearnedData1 = copy.deepcopy(self.LearnedData)
            future = np.where((np.array(LearnedData1.AnalyzedData.t) > t + (3*self.N / 4) * self.dt) &
                                    (np.array(LearnedData1.AnalyzedData.t) <= t + (1.25 * self.N) * self.dt))[0]
            if future.size > 0:
                LearnedData1.AnalyzedData.t = np.array(LearnedData1.AnalyzedData.t)[future]
                # LearnedData1.AnalyzedData.c = np.array(LearnedData1.AnalyzedData.c)[future]
                LearnedData1.AnalyzedData.state = np.array(LearnedData1.AnalyzedData.state)[future]
                LearnedData1.AnalyzedData.Cost2Go = np.array(LearnedData1.AnalyzedData.Cost2Go)[future]
                if hasattr(LearnedData1.AnalyzedData, "Cost2Go2"):
                    LearnedData1.AnalyzedData.Cost2Go2 = np.array(
                        LearnedData1.AnalyzedData.Cost2Go2
                    )[future]
                if hasattr(LearnedData1.AnalyzedData, "Cost2Go3"):
                    LearnedData1.AnalyzedData.Cost2Go3 = np.array(
                        LearnedData1.AnalyzedData.Cost2Go3
                    )[future]
                LearnedData1.AnalyzedData.n_data = LearnedData1.AnalyzedData.t.shape[0]
            else:
                LearnedData1 = None
            self.Solver = self.build_solver(u2_0 = u2_Learned, Terminal_Safe_Set=LearnedData1.AnalyzedData)
        

        _ensure_julia()
                    
        x0 = np.asarray(x0, dtype=float)
        if x0.shape != (self.game.nx,):
            raise ValueError(f"x0 must have shape ({self.game.nx},)")

        # Design initial guess for z vector
        x1_0 = x0[:self.game.nx1].reshape(1, self.game.nx1)
        x2_0 = x0[self.game.nx1:].reshape(1, self.game.nx2)

        n_z = int(self.Solver.Z.shape[0])
        if u1_0 is None:
            u1 = np.ones((self.N, self.game.nu1))
            u1[:,0] *= -self.game.u_max*0.01
            u1[:,1] *= -self.game.u_max*0.01
        else:
            u1 = np.asarray(u1_0, dtype=float)
            u1 = u1[:self.N, :]
        if u1.shape != (self.N, self.game.nu1):
            raise ValueError(f"u1_0 must have shape ({self.N}, {self.game.nu1})")

        if u2_0 is None:
            u2 = np.ones((self.N, self.game.nu2))
            u2[:,0] *= self.game.u_max*0.01
            u2[:,1] *= self.game.u_max*0.01
        else:
            u2 = np.asarray(u2_0, dtype=float)
            u2 = u2[:self.N, :]
        if u2.shape != (self.N, self.game.nu2):
            raise ValueError(f"u2_0 must have shape ({self.N}, {self.game.nu2})")

        x1_len, u1_len, ai_len, x1f_slack_len = self.Solver.Z_len[0]
        x2_len, u2_len, x2f_slack_len = self.Solver.Z_len[1]
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
        x1f_slack = np.zeros((x1f_slack_len,1))
        x2f_slack = np.zeros((x2f_slack_len,1))

        z0 = np.concatenate(
            (
                x1.reshape(x1_len, order="F"),
                u1.reshape(u1_len, order="F"),
                ai_xf_vec.reshape(ai_len, order="F"),
                x1f_slack.reshape(x1f_slack.shape[0], order="F"),
                x2.reshape(x2_len, order="F"),
                u2.reshape(u2_len, order="F"),
                x2f_slack.reshape(x2f_slack.shape[0], order="F"),
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
        Main.nnz = self.Solver.J.nnz_out(0) # Uses CasADi's structural non-zero count
        Main.F_py = lambda z: np.array(self.Solver.F(z, x1_0, x2_0, alpha_vec)).squeeze()
        Main.J_py = lambda z: np.array(self.Solver.J(z, x1_0, x2_0, alpha_vec))
        
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
                    if !iszero(j[r, c])
                        row[i], data[i] = r, j[r, c]
                        len[c] += 1
                        i += 1
                    end
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
            sample_progress = (
                f", sample={sample_number}/{sample_count}"
                if sample_number is not None and sample_count is not None
                else ""
            )
            print(
                f"Solver Not Converged: residual={residual:2.2}, "
                f"status={status.__name__}{sample_progress}"
            )

        i = 0
        x1_len, u1_len, ai_len, x1f_slack_len = self.Solver.Z_len[0]
        x2_len, u2_len, x2f_slack_len = self.Solver.Z_len[1]
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
        x1f_slack = z[i:i + x1f_slack_len].reshape(x1f_slack_len, 1, order="F")
        i += x1f_slack_len
        x2 = z[i:i + x2_len].reshape(self.N + 1, self.game.nx2, order="F")
        i += x2_len
        u2 = z[i:i + u2_len].reshape(self.N, self.game.nu2, order="F")
        i += u2_len
        x2f_slack = z[i:i + x2f_slack_len].reshape(x2f_slack_len, 1, order="F")
        i += x2f_slack_len
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
            self.Solution.x1f_slack = x1f_slack
            self.Solution.x2f_slack = x2f_slack
        elif hasattr(self.Solution, "indx"):
            self.Solution.success = bool(success)
            if last_attempted_solution:
                self.Solution.indx = min(self.Solution.indx + 1, self.N - 1)

        if not hasattr(self.Solution, "u1") or not hasattr(self.Solution, "u2"):
            return np.zeros(self.game.nu)

        u = np.concatenate((self.Solution.u1[self.Solution.indx], self.Solution.u2[self.Solution.indx]))

        return u

    def _step_once_three_player(
        self, t, x0, forced_alpha=None, terminal_learned_data=None,
        terminal_solver=None, precomputed_a_set=None,
        sample_number=None, sample_count=None,
    ):
        """Solve and unpack the three-player MCP using the standard backend."""
        if precomputed_a_set is not None:
            a_set, proximity_factor = precomputed_a_set
        else:
            a_set, proximity_factor = self.calc_a_set(x0)
        alpha_vec = self.alpha_vec.copy()
        if forced_alpha is not None:
            alpha_vec[:] = forced_alpha
        elif self.LearnedData is not None:
            alpha_vec[0] = np.clip(1.0 - proximity_factor, 0.1, 1.0)

        terminal = None
        if terminal_learned_data is not None:
            terminal = terminal_learned_data.AnalyzedData
        elif self.LearnedData is not None:
            terminal = self.LearnedData.AnalyzedData
        if terminal_solver is not None:
            self.Solver = terminal_solver
        elif not self.is_built or terminal is not None:
            self.Solver = self.build_solver(Terminal_Safe_Set=terminal)

        _ensure_julia()
        x0 = np.asarray(x0, dtype=float)
        if x0.shape != (self.game.nx,):
            raise ValueError(f"x0 must have shape ({self.game.nx},)")
        x0s = [x0[p * self.game.nx1:(p + 1) * self.game.nx1].reshape(1, -1)
                for p in range(self.game.n_players)]

        initial_parts = []
        trajectories = []
        controls = []
        for player, lengths in enumerate(self.Solver.Z_len[:self.game.n_players]):
            x_len, u_len, ai_len, slack_len = lengths
            sign = -1.0 if player == 0 else 1.0
            up = np.full((self.N, self.game.nu1), sign * self.game.u_max * 0.01)
            xp = np.zeros((self.N + 1, self.game.nx1))
            xp[0] = x0s[player]
            A, B = self.A_players[player], self.B_players[player]
            for k in range(self.N):
                xp[k + 1] = A @ xp[k] + B @ up[k]
            trajectories.append(xp)
            controls.append(up)
            initial_parts.extend([
                xp.reshape(x_len, order='F'), up.reshape(u_len, order='F'),
                np.zeros(ai_len), np.zeros(slack_len),
            ])
        mu_len, lambda_len, sigma_len = self.Solver.Z_len[-3:]
        z0 = np.concatenate([*initial_parts, np.zeros(mu_len),
                             np.zeros(lambda_len), np.zeros(sigma_len)])

        Main.z0 = z0
        Main.ub = np.inf * np.ones(self.Solver.n_u_inf)
        Main.lb = np.concatenate((
            -np.inf * np.ones(self.Solver.n_l_inf),
            np.zeros(self.Solver.n_u_inf - self.Solver.n_l_inf),
        ))
        Main.nnz = self.Solver.J.nnz_out(0)
        function_args = [*x0s, alpha_vec]
        Main.F_py = lambda z: np.array(self.Solver.F(z, *function_args)).squeeze()
        Main.J_py = lambda z: np.array(self.Solver.J(z, *function_args))
        Main.tol = self.p_tol
        Main.F = jl.eval("""
        function F(n::Cint, x::Vector{Cdouble}, f::Vector{Cdouble})
            f .= F_py(x); return Cint(0)
        end
        return(F)
        """)
        Main.J = jl.eval("""
        function J(n::Cint, nnz::Cint, x::Vector{Cdouble}, col::Vector{Cint},
                   len::Vector{Cint}, row::Vector{Cint}, data::Vector{Cdouble})
            j = J_py(x); i = 1
            for c in 1:n
                col[c], len[c] = i, 0
                for r in 1:n
                    if !iszero(j[r,c])
                        row[i], data[i] = r, j[r,c]; len[c] += 1; i += 1
                    end
                end
            end
            return Cint(0)
        end
        return(J)
        """)
        output = 'yes' if self.verbose else 'no'
        nms = 'yes' if self.nms else 'no'
        z, success, residual, status = jl.eval(f"""
        PATHSolver.c_api_License_SetString("1259252040&Courtesy&&&USR&GEN2035&5_1_2026&1000&PATH&GEN&31_12_2035&0_0_0&6000&0_0")
        status, z, info = PATHSolver.solve_mcp(F, J, lb, ub, z0,
            nnz=nnz, output="{output}", convergence_tolerance=tol,
            nms="{nms}", crash_nbchange_limit=50, major_iteration_limit=500,
            minor_iteration_limit=10000, cumulative_iteration_limit=100000,
            restart_limit=100)
        return z, status == PATHSolver.MCP_Solved, info.residual, status
        """)
        z = np.asarray(z, dtype=float).reshape(-1)
        self.last_solve_success = bool(success)

        offset = 0
        solved_x, solved_u, slacks = [], [], []
        ai_xf_vec = np.zeros((0, 1))
        for player, lengths in enumerate(self.Solver.Z_len[:self.game.n_players]):
            x_len, u_len, ai_len, slack_len = lengths
            solved_x.append(z[offset:offset + x_len].reshape(
                self.N + 1, self.game.nx1, order='F'))
            offset += x_len
            solved_u.append(z[offset:offset + u_len].reshape(
                self.N, self.game.nu1, order='F'))
            offset += u_len
            if ai_len:
                ai_xf_vec = z[offset:offset + ai_len].reshape(-1, 1)
            offset += ai_len
            slacks.append(z[offset:offset + slack_len].reshape(-1, 1))
            offset += slack_len
        offset += mu_len + lambda_len
        sigma = z[offset:offset + sigma_len]

        if success:
            self.Solution = SimpleNamespace(
                success=True, t=t, z=z, residual=float(residual),
                status=status.__name__, ai_xf_vec=ai_xf_vec, sigma=sigma,
                a_set=a_set, x0=x0, indx=0,
            )
            for player in range(self.game.n_players):
                setattr(self.Solution, f'x{player + 1}', solved_x[player])
                setattr(self.Solution, f'u{player + 1}', solved_u[player])
                setattr(self.Solution, f'x{player + 1}f_slack', slacks[player])
        elif hasattr(self.Solution, 'indx'):
            self.Solution.success = False
        if not all(hasattr(self.Solution, f'u{p + 1}')
                   for p in range(self.game.n_players)):
            return np.zeros(self.game.nu)
        return np.concatenate([
            getattr(self.Solution, f'u{p + 1}')[self.Solution.indx]
            for p in range(self.game.n_players)
        ])

    def update_alpha_vec(self, new_alpha):
        """Update the alpha vector used for shared constraints."""
        new_alpha = np.asarray(new_alpha, dtype=float).reshape(-1)
        if self.game.n_players == 3:
            if new_alpha.shape != (2,) or np.any(new_alpha < 0.0) or np.sum(new_alpha) > 1.0 + 1e-12:
                raise ValueError("new_alpha must be a valid (alpha1, alpha2) pair")
            self.alpha_vec[:] = new_alpha
        else:
            if new_alpha.shape != (1,) or not 0.0 <= new_alpha[0] <= 1.0:
                raise ValueError("new_alpha must be in [0, 1]")
            self.alpha_vec[:] = new_alpha[0]

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
            
        proximity_factor = np.sqrt(a_vec@a_vec)
        proximity_factor -= np.clip(10.0*(factor-1.0), 0.0, 1.0)
        
        a_vec1 = np.zeros((len(self.LearnedData.AnalyzedData.state)))
        for j, isort in enumerate(arg_sort[0:sorted_states.shape[0]]):
            a_vec1[isort] = a_vec[j]
            
        if self.constraint_mode == "sampled_points":
            if proximity_factor > 1.0-1e-4:
                return a_vec1, proximity_factor
            else:
                return 0.0, 0.0
        
        return a_vec1, proximity_factor
        
        
    def backup_controller_init(self):
        self.backup = SimpleNamespace()
        self.backup.time = np.asarray(
            self.LearnedData.RawData[-1].t, dtype=float
        ).copy()
        self.backup.x = np.asarray(
            self.LearnedData.RawData[-1].x, dtype=float
        ).copy()
        self.backup.u = np.asarray(
            self.LearnedData.RawData[-1].u, dtype=float
        ).copy()
        self.backup.cost1 = self.LearnedData.RawData[-1].p1_total_cost
        self.backup.cost2 = self.LearnedData.RawData[-1].p2_total_cost
        self.backup.indx = 0
        
    def backup_controller_update(self, Solution):
        """Replace the backup with ``Solution`` followed by a learned suffix.

        The optimized trajectory ends at a state sampled from a previous safe
        trajectory.  Locate that state in ``LearnedData.RawData`` and splice
        the remainder of the saved trajectory onto the optimized prefix.  A
        state is stored once at the splice, while its saved control is retained
        because it drives the system from the terminal state toward the next
        saved state.
        """
        if self.LearnedData is None or not self.LearnedData.RawData:
            raise ValueError("cannot update backup without learned raw data")

        player_count = getattr(self.game, "n_players", 2)
        required_fields = tuple(
            [f"x{p + 1}" for p in range(player_count)]
            + [f"u{p + 1}" for p in range(player_count)]
            + ["t"]
        )
        missing_fields = [
            field for field in required_fields if not hasattr(Solution, field)
        ]
        if missing_fields:
            raise ValueError(
                "Solution is missing required backup fields: "
                + ", ".join(missing_fields)
            )

        player_states = [np.asarray(getattr(Solution, f"x{p + 1}"), dtype=float)
                         for p in range(player_count)]
        player_controls = [np.asarray(getattr(Solution, f"u{p + 1}"), dtype=float)
                           for p in range(player_count)]
        if any(value.ndim != 2 or value.shape[0] != self.N + 1
               for value in player_states):
            raise ValueError("all solution state trajectories must have N + 1 rows")
        if any(value.ndim != 2 or value.shape[0] != self.N
               for value in player_controls):
            raise ValueError("all solution control trajectories must have N rows")
        if player_states[0].shape[0] != self.N + 1 or player_controls[0].shape[0] != self.N:
            raise ValueError(
                f"the solution must contain {self.N + 1} states and "
                f"{self.N} controls"
            )

        solution_states = np.concatenate(player_states, axis=1)
        solution_controls = np.concatenate(player_controls, axis=1)
        if solution_states.shape[1] != self.game.nx:
            raise ValueError(f"solution states must have {self.game.nx} elements")
        if solution_controls.shape[1] != self.game.nu:
            raise ValueError(f"solution controls must have {self.game.nu} elements")

        terminal_state = getattr(Solution, "terminal_sample_state", None)
        if terminal_state is None:
            terminal_state = solution_states[-1]
        terminal_state = np.asarray(terminal_state, dtype=float).reshape(-1)
        if terminal_state.shape != (self.game.nx,):
            raise ValueError(
                f"Solution.terminal_sample_state must have {self.game.nx} elements"
            )

        terminal_sample_time = float(
            getattr(Solution, "terminal_sample_time", np.nan)
        )
        matches = []
        for raw_iteration, raw_data in reversed(
            list(enumerate(self.LearnedData.RawData))
        ):
            raw_states = np.asarray(raw_data.x, dtype=float)
            raw_times = np.asarray(raw_data.t, dtype=float).reshape(-1)
            raw_controls = np.asarray(raw_data.u, dtype=float)
            if (
                raw_states.ndim != 2
                or raw_states.shape[1] != self.game.nx
                or raw_controls.ndim != 2
                or raw_controls.shape != (raw_states.shape[0], self.game.nu)
                or raw_times.shape[0] != raw_states.shape[0]
            ):
                continue

            state_matches = np.flatnonzero(
                np.all(
                    np.isclose(
                        raw_states,
                        terminal_state,
                        rtol=1e-7,
                        atol=1e-9,
                    ),
                    axis=1,
                )
            )
            for raw_index in state_matches:
                time_error = (
                    abs(raw_times[raw_index] - terminal_sample_time)
                    if np.isfinite(terminal_sample_time)
                    else 0.0
                )
                matches.append(
                    (time_error, -raw_iteration, int(raw_index), raw_data)
                )

        if not matches:
            raise ValueError(
                "the solution terminal safe state was not found in LearnedData.RawData"
            )

        _, _, raw_index, raw_data = min(matches, key=lambda match: match[:3])
        raw_states = np.asarray(raw_data.x, dtype=float)
        raw_controls = np.asarray(raw_data.u, dtype=float)
        raw_times = np.asarray(raw_data.t, dtype=float).reshape(-1)

        solution_start_time = float(Solution.t)
        solution_control_times = (
            solution_start_time
            + self.dt * np.arange(solution_controls.shape[0], dtype=float)
        )
        solution_terminal_time = (
            solution_start_time + self.dt * solution_controls.shape[0]
        )
        learned_suffix_times = (
            solution_terminal_time
            + raw_times[raw_index:]
            - raw_times[raw_index]
        )

        backup = SimpleNamespace()
        backup.time = np.concatenate(
            (solution_control_times, learned_suffix_times)
        )
        backup.x = np.concatenate(
            (solution_states, raw_states[raw_index + 1:]), axis=0
        )
        backup.u = np.concatenate(
            (solution_controls, raw_controls[raw_index:]), axis=0
        )
        backup.cost1 = float(
            getattr(
                Solution,
                "player1_cost",
                getattr(raw_data, "p1_total_cost", np.nan),
            )
        )
        backup.cost2 = float(
            getattr(
                Solution,
                "player2_cost",
                getattr(raw_data, "p2_total_cost", np.nan),
            )
        )

        if not (backup.time.shape[0] == backup.x.shape[0] == backup.u.shape[0]):
            raise RuntimeError(
                "backup time, state, and control trajectories are misaligned"
            )
        backup.indx = 0
        self.backup = backup
    
    def backup_controller(self, x):
        """Return the safe stored control closest to the current state.

        Only the unexecuted portion of the trajectory is searched.  This
        prevents a self-intersecting backup trajectory from moving its index
        backward and replaying controls that have already been applied.
        """
        if not hasattr(self, "backup"):
            raise RuntimeError("backup controller has not been initialized")
        player_count = getattr(self.game, "n_players", 2)

        state = np.asarray(x, dtype=float).reshape(-1)
        states = np.asarray(self.backup.x, dtype=float)
        controls = np.asarray(self.backup.u, dtype=float)
        times = np.asarray(self.backup.time, dtype=float).reshape(-1)
        if state.shape != (self.game.nx,):
            raise ValueError(f"x must have shape ({self.game.nx},)")
        if (
            states.ndim != 2
            or states.shape[1] != self.game.nx
            or controls.ndim != 2
            or controls.shape != (states.shape[0], self.game.nu)
            or times.shape[0] != states.shape[0]
            or states.shape[0] == 0
        ):
            raise RuntimeError("stored backup trajectory is invalid")

        first_index = int(
            np.clip(
                getattr(self.backup, "indx", 0),
                0,
                states.shape[0] - 1,
            )
        )
        errors = states[first_index:] - state
        state_weight = np.asarray(
            getattr(self, "proximity_Q", np.eye(self.game.nx)), dtype=float
        )
        if state_weight.shape != (self.game.nx, self.game.nx):
            state_weight = np.eye(self.game.nx)
        distances = np.einsum("ij,jk,ik->i", errors, state_weight, errors)
        backup_index = first_index + int(np.argmin(distances))

        self.backup.indx = backup_index
        if hasattr(self, "Solution"):
            state_stop = min(backup_index + self.N + 1, states.shape[0])
            control_stop = min(backup_index + self.N, controls.shape[0])
            remaining_states = states[backup_index:state_stop].copy()
            remaining_controls = controls[backup_index:control_stop].copy()

            if remaining_states.shape[0] < self.N + 1:
                state_padding = np.repeat(
                    states[-1][None, :],
                    self.N + 1 - remaining_states.shape[0],
                    axis=0,
                )
                remaining_states = np.concatenate(
                    (remaining_states, state_padding), axis=0
                )
            if remaining_controls.shape[0] < self.N:
                control_padding = np.repeat(
                    controls[-1][None, :],
                    self.N - remaining_controls.shape[0],
                    axis=0,
                )
                remaining_controls = np.concatenate(
                    (remaining_controls, control_padding), axis=0
                )

            self.Solution.t = float(times[backup_index])
            self.Solution.x0 = state.copy()
            for player in range(player_count):
                xs = slice(player * self.game.nx1, (player + 1) * self.game.nx1)
                us = slice(player * self.game.nu1, (player + 1) * self.game.nu1)
                setattr(self.Solution, f"x{player + 1}", remaining_states[:, xs])
                setattr(self.Solution, f"u{player + 1}", remaining_controls[:, us])
            self.Solution.indx = 0
            self.Solution.success = False
            self.Solution.status = "backup_controller"
            self.Solution.is_backup = True
            self.Solution.used_backup_controller = True
            self.Solution.backup_index = backup_index
            self.Solution.terminal_sample_state = remaining_states[-1].copy()
            self.Solution.terminal_sample_time = float(
                times[backup_index] + self.N * self.dt
            )
        return controls[backup_index].copy()

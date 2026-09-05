import numpy as np
import casadi as ca

class GameDynamics:
    """Multi-player planar dynamics, including a nonlinear unicycle model."""

    # Return codes for one integration cycle.
    INPUT_OUTSIDE_BOUNDS = 1
    POSITION_OUTSIDE_BOUNDS = 2
    VELOCITY_OUTSIDE_BOUNDS = 3
    SHARED_CONSTRAINT_VIOLATED = 4
    STEP_OK = 0
    
    eps = 5e-3

    def __init__(
        self,
        dt,
        x0,
        x1f,
        x2f,
        x3f=None,
        u_min=-2,
        u_max=2,
        L=20.0,
        W=2,
        vx_min=-2,
        vx_max=2,
        vy_min=-2,
        vy_max=2,
        v_min=0.1,
        v_max=2.0,
        a_max=2.0,
        psi_max=np.pi,
        d_sep=0.3,
        dynamics_type=3,
        MaxIterations=50,
    ):
        if dt <= 0:
            raise ValueError("dt must be positive")
        if dynamics_type not in (1, 2, 3):
            raise ValueError(
                "dynamics_type must be 1 (single integrator), 2 (double "
                "integrator), or 3 (unicycle)"
            )

        self.dt = float(dt)
        self.dynamics_type = int(dynamics_type)
        self.nx1 = {1: 2, 2: 4, 3: 3}[self.dynamics_type]
        self.nu1 = 2
        self.targets = [
            np.asarray(target, dtype=float).reshape(-1)
            for target in (x1f, x2f, x3f) if target is not None
        ]
        self.n_players = len(self.targets)
        if self.n_players not in (2, 3):
            raise ValueError("GameDynamics supports two or three players")
        if any(target.shape != (self.nx1,) for target in self.targets):
            raise ValueError(f"each target must contain {self.nx1} state values")
        self.nx2 = self.nx1
        self.nu2 = self.nu1
        self.nx = self.n_players * self.nx1
        self.nu = self.n_players * self.nu1
        self.iteration = 0
        self.Max_Iterations = MaxIterations

        # Input bounds apply to [a1x, a1y, a2x, a2y].
        self.u_min = u_min
        self.u_max = u_max

        # Position bounds apply to each player's x/y coordinates.
        self.x_min = float(-L/2)
        self.x_max = float(L/2)
        self.y_min = float(-W/2)
        self.y_max = float(W/2)

        # Velocity bounds apply to velocity states for double-integrator mode.
        self.vx_min = float(vx_min)
        self.vx_max = float(vx_max)
        self.vy_min = float(vy_min)
        self.vy_max = float(vy_max)
        self.v_min = float(v_min)
        self.v_max = float(v_max)
        self.a_max = float(a_max)
        self.psi_max = float(psi_max)
        if min(self.v_max, self.a_max, self.psi_max) <= 0:
            raise ValueError("unicycle limits must be positive")
        if self.v_min < 0 or self.v_min >= self.v_max:
            raise ValueError("v_min must satisfy 0 <= v_min < v_max")
        
        # shared constranits data
        shared_f_limit = 1.25
        self.u_max_shared = self.u_max*shared_f_limit
        self.u_min_shared = self.u_min*shared_f_limit
        
        self.d_sep = d_sep
                
        self.x0 = x0
        self.x1f = x1f
        self.x2f = x2f
        self.x3f = x3f

        # Define shared constranits function:
        x_sym = ca.SX.sym('x_sym', self.nx)
        u_syms = [ca.SX.sym(f'u{player + 1}_sym', self.nu1)
                  for player in range(self.n_players)]
        u1_sym, u2_sym = u_syms[:2]

        # self.f_shared = ca.Function('f_shared', [x_sym, u1_sym, u2_sym], [u1_sym[0]+u2_sym[0]-self.u_min_shared, self.u_max_shared-u1_sym[0]-u2_sym[0], u1_sym[1]+u2_sym[1]-self.u_min_shared, self.u_max_shared-u2_sym[1]-u1_sym[1]])
        x1_sym = x_sym[:self.nx1]
        x2_sym = x_sym[self.nx1:self.nx1+self.nx2]
        if self.is_single_integrator:
            v1_sym = u1_sym
            v2_sym = u2_sym
        elif not self.is_unicycle:
            v1_sym = x_sym[2:4]
            v2_sym = x_sym[self.nx1+2:self.nx1+4]
        
        # Player Private Constranits:
        if self.is_single_integrator:
            self.f_private = ca.Function('f_private', [x1_sym, u1_sym], [u1_sym[0]-self.u_min, 
                                                                     self.u_max-u1_sym[0], 
                                                                     u1_sym[1]-self.u_min, 
                                                                     self.u_max-u1_sym[1], 
                                                                     x1_sym[0]-self.x_min, 
                                                                     self.x_max-x1_sym[0], 
                                                                     x1_sym[1]-self.y_min, 
                                                                     self.y_max-x1_sym[1]])
        elif self.is_unicycle:
            self.f_private = ca.Function(
                'f_private', [x1_sym, u1_sym], [
                    x1_sym[0] - self.x_min,
                    self.x_max - x1_sym[0],
                    x1_sym[1] - self.y_min,
                    self.y_max - x1_sym[1],
                    x1_sym[2] - self.v_min,
                    self.v_max - x1_sym[2],
                    u1_sym[0] + self.a_max,
                    self.a_max - u1_sym[0],
                    u1_sym[1] + self.psi_max,
                    self.psi_max - u1_sym[1],
                ],
            )
        else:
            self.f_private = ca.Function('f_private', [x1_sym, u1_sym], [u1_sym[0]-self.u_min, 
                                                                        self.u_max-u1_sym[0], 
                                                                        u1_sym[1]-self.u_min, 
                                                                        self.u_max-u1_sym[1], 
                                                                        x1_sym[0]-self.x_min, 
                                                                        self.x_max-x1_sym[0], 
                                                                        x1_sym[1]-self.y_min, 
                                                                        self.y_max-x1_sym[1],
                                                                        v1_sym[0]-self.vx_min,
                                                                        self.vx_max-v1_sym[0],
                                                                        v1_sym[1]-self.vy_min,
                                                                        self.vy_max-v1_sym[1]])
        
        shared_constraints = []
        if not self.is_unicycle:
            velocities = []
            for player, u_sym in enumerate(u_syms):
                offset = player * self.nx1
                velocities.append(u_sym if self.is_single_integrator
                                  else x_sym[offset + 2:offset + 4])
            shared_constraints.append(
                4.5 * self.vy_max**2 - sum(ca.sumsqr(v) for v in velocities)
            )
        for first in range(self.n_players):
            for second in range(first + 1, self.n_players):
                i = first * self.nx1
                j = second * self.nx1
                shared_constraints.append(
                    ca.sumsqr(x_sym[i:i + 2] - x_sym[j:j + 2]) - self.d_sep**2
                )
        self.f_shared = ca.Function(
            'f_shared', [x_sym, *u_syms], shared_constraints
        )
        # self.f_shared = ca.Function('f_shared', [x_sym, u1_sym, u2_sym], [ca.sumsqr(x_sym[:2]-x_sym[self.nx1:self.nx1+2]) - self.d_sep**2])

        # Dynamics Function:
        # Internal state is [p1x, p1y, p2x, p2y] for single-integrator mode,
        # [px, py, vx, vy] per player for double-integrator mode,
        # or [px, py, v] per player for unicycle mode.
        k1 = self.dynamics(x1_sym, u1_sym)
        k2 = self.dynamics(x1_sym + 0.5 * dt *k1, u1_sym)
        k3 = self.dynamics(x1_sym + 0.5 * dt * k2, u1_sym)
        k4 = self.dynamics(x1_sym + dt * k3, u1_sym)

        self.xkp1 = x1_sym + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        self.dynamics_fun = ca.Function('fynamics_fun', [x1_sym, u1_sym], [self.xkp1])
        
        
        return

    @property
    def is_single_integrator(self):
        return self.dynamics_type == 1

    @property
    def is_unicycle(self):
        return self.dynamics_type == 3

    @staticmethod
    def _as_bounds(value, size, name):
        """Convert a scalar bound or vector bound into a fixed-size array."""
        bounds = np.asarray(value, dtype=float)
        if bounds.shape == ():
            return np.full(size, bounds, dtype=float)
        if bounds.shape != (size,):
            raise ValueError(f"{name} must be a scalar or have shape ({size},)")
        return bounds.copy()

    def set_state(self, x):
        """Set the internal state."""
        x = np.asarray(x, dtype=float)
        if x.shape != (self.nx,):
            raise ValueError(f"x must have shape ({self.nx},)")
        self.x = x.copy()

    def get_state(self):
        """Return a copy of the internal state."""
        return self.x.copy()

    def reset_history(self):
        """Clear the simulation log and record the current state at the current time."""
        self.history = {
            "t": [],
            "x": [],
            "u": [],
            "status": [],
        }
        self._log_history(status=self.STEP_OK)

    def _log_history(self, u = None, status = STEP_OK):
        """Record one time, state, input, and step status sample."""
        self.history["t"].append(float(self.t))
        self.history["x"].append(self.x.copy())
        if u is not None:
            self.history["u"].append(np.asarray(u, dtype=float).copy())
        self.history["status"].append(int(status))

    def get_history(self):
        """Return the simulation log as numpy arrays."""
        return {
            "t": np.asarray(self.history["t"], dtype=float),
            "x": np.asarray(self.history["x"], dtype=float),
            "u": np.asarray(self.history["u"], dtype=float),
            "status": np.asarray(self.history["status"], dtype=int),
        }

    def save_history(self, path):
        """Save the simulation log to a compressed NumPy archive."""
        history = self.get_history()
        np.savez(path, **history)

    def dynamics(self, x, u):
        """
        Continuous-time dynamics for NumPy or CasADi vectors.

        ``x`` and ``u`` may describe either one player or the full game. The
        return value is a NumPy array for NumPy inputs and a CasADi column
        vector when either input is a CasADi ``SX``, ``MX``, or ``DM`` value.

        Single integrator:
            State x = [p1x, p1y, p2x, p2y]
            Input u = [v1x, v1y, v2x, v2y]

        Double integrator:
            State x = [p1x, p1y, v1x, v1y, p2x, p2y, v2x, v2y]
            Input u = [a1x, a1y, a2x, a2y]

        Unicycle (per player):
            State x = [x, y, v]
            Input u = [a, psi], with heading psi in radians
            x_dot = [v*cos(psi), v*sin(psi), a]
        """
        casadi_types = (ca.SX, ca.MX, ca.DM)
        use_casadi = isinstance(x, casadi_types) or isinstance(u, casadi_types)

        def prepare_vector(value, name):
            if isinstance(value, casadi_types):
                if not value.is_vector():
                    raise ValueError(f"{name} must be a vector")
                return ca.reshape(value, value.numel(), 1), value.numel()

            value = np.asarray(value, dtype=float)
            if value.ndim != 1:
                raise ValueError(f"{name} must be a one-dimensional array")
            if use_casadi:
                return ca.DM(value), value.size
            return value, value.size

        x, x_size = prepare_vector(x, "x")
        u, u_size = prepare_vector(u, "u")

        if x_size == self.nx1:
            expected_u_size = self.nu1
            player_count = 1
        elif x_size == self.nx:
            expected_u_size = self.nu
            player_count = self.n_players
        else:
            raise ValueError(
                f"x must contain one player ({self.nx1} elements) or "
                f"the full game ({self.nx} elements)"
            )
        if u_size != expected_u_size:
            raise ValueError(
                f"u must have {expected_u_size} elements when x has "
                f"{x_size} elements"
            )

        if self.is_single_integrator:
            components = [u[index] for index in range(u_size)]
        elif self.is_unicycle:
            components = []
            for player in range(player_count):
                x_offset = player * self.nx1
                u_offset = player * self.nu1
                speed = x[x_offset + 2]
                heading = u[u_offset + 1]
                components.extend([
                    speed * ca.cos(heading) if use_casadi else speed * np.cos(heading),
                    speed * ca.sin(heading) if use_casadi else speed * np.sin(heading),
                    u[u_offset],
                ])
        else:
            components = []
            for player in range(player_count):
                x_offset = player * self.nx1
                u_offset = player * self.nu1
                components.extend(
                    [
                        x[x_offset + 2],
                        x[x_offset + 3],
                        u[u_offset],
                        u[u_offset + 1],
                    ]
                )

        if use_casadi:
            return ca.vertcat(*components)
        return np.asarray(components, dtype=float)

    def step(self, u):
        """
        Advance the internal state one time step using RK4 integration.

        Returns:
            0: step went ok
            1: input outside bounds
            2: position outside bounds after integration
            3: velocity outside bounds after integration
        """
        u = np.asarray(u, dtype=float)
        if u.shape != (self.nu,):
            raise ValueError(f"u must have shape ({self.nu},)")
        self.u = u

        # Reject invalid controls before changing the internal state.
        if self.is_unicycle:
            accelerations = u[0::2]
            headings = u[1::2]
            invalid_input = (
                np.any(np.abs(accelerations) > self.a_max + self.eps)
                or np.any(np.abs(headings) > self.psi_max + self.eps)
            )
        else:
            invalid_input = np.any(u < self.u_min-self.eps) or np.any(u > self.u_max+self.eps)
        if invalid_input:
            self._log_history(u, self.INPUT_OUTSIDE_BOUNDS)
            return self.INPUT_OUTSIDE_BOUNDS

        x = self.x
        dt = self.dt
        # Classical RK4 integration supports both linear and nonlinear modes.
        k1 = self.dynamics(x, u)
        k2 = self.dynamics(x + 0.5 * dt * k1, u)
        k3 = self.dynamics(x + 0.5 * dt * k2, u)
        k4 = self.dynamics(x + dt * k3, u)
        x_next = x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        self.x = x_next
        self.t += dt

        # Check axis-aligned position bounds for both players.
        xs = self.x[0::self.nx1]
        ys = self.x[1::self.nx1]

        if (
            np.any(xs < self.x_min-self.eps)
            or np.any(xs > self.x_max+self.eps)
            or np.any(ys < self.y_min-self.eps)
            or np.any(ys > self.y_max+self.eps)
        ):
            self._log_history(u, self.POSITION_OUTSIDE_BOUNDS)
            return self.POSITION_OUTSIDE_BOUNDS

        if self.is_unicycle:
            speeds = self.x[2::self.nx1]
            if (
                np.any(speeds < self.v_min - self.eps)
                or np.any(speeds > self.v_max + self.eps)
            ):
                self._log_history(u, self.VELOCITY_OUTSIDE_BOUNDS)
                return self.VELOCITY_OUTSIDE_BOUNDS
        elif not self.is_single_integrator:
            vxs = self.x[2::self.nx1]
            vys = self.x[3::self.nx1]
            if (
                np.any(vxs < self.vx_min-self.eps)
                or np.any(vxs > self.vx_max+self.eps)
                or np.any(vys < self.vy_min-self.eps)
                or np.any(vys > self.vy_max+self.eps)
            ):
                self._log_history(u, self.VELOCITY_OUTSIDE_BOUNDS)
                return self.VELOCITY_OUTSIDE_BOUNDS
        
        controls = [
            self.u[p * self.nu1:(p + 1) * self.nu1]
            for p in range(self.n_players)
        ]
        f_shared = self.f_shared(self.x, *controls)
        if not isinstance(f_shared, tuple):
            f_shared = (f_shared,)
        for f in f_shared:
            if f < -self.eps:
                return self.SHARED_CONSTRAINT_VIOLATED
        
        self._log_history(u, self.STEP_OK)
        return self.STEP_OK

    def reset_game(self):
        self.t = 0.0
        self.x = np.zeros(self.nx, dtype=float)
        if self.x0 is not None:
            self.set_state(self.x0)
        self.reset_history()
        self.iteration += 1

    def _unicycle_goal_controller(
        self, player, target, position_gain=0.75,
        speed_gain=3.0,
    ):
        """Bounded point-tracking controller for the unicycle bootstrap."""
        offset = player * self.nx1
        state = self.x[offset:offset + self.nx1]
        error = np.asarray(target, dtype=float).reshape(-1)[:2] - state[:2]
        distance = np.linalg.norm(error)
        if distance > 1e-8:
            desired_heading = np.arctan2(error[1], error[0])
        else:
            desired_heading = 0.0
        desired_speed = max(self.v_min, min(self.v_max / 2, position_gain * distance))
        acceleration = np.clip(
            speed_gain * (desired_speed - state[2]), -self.a_max*0.6, self.a_max*0.6
        )
        heading = np.clip(desired_heading, -self.psi_max, self.psi_max)
        return np.array([acceleration, heading])

    def SimpleController1(self, position_gain=2.0, velocity_gain=5.0, max_velocity=1.0):
        """Return a bounded, goal-tracking control for player 1.

        The single-integrator controller commands velocity proportional to the
        position error.  The double-integrator controller uses position and
        velocity feedback to command acceleration.  In both cases the result
        respects player 1's input bounds.
        """
        
        if (
            self.t < 1.5
            and (
                self.is_single_integrator
                or self.is_unicycle
                or (
                    abs(self.x[3]) < self.vy_max - 0.5
                    and abs(self.x[2]) < self.vx_max - 0.5
                )
            )
        ):
            target = np.asarray(self.x1f, dtype=float).reshape(-1).copy()
            target[0] -= 2.0
        else:
            target = np.asarray(self.x1f, dtype=float).reshape(-1)
        
        if self.is_unicycle:
            return self._unicycle_goal_controller(0, target)

        if target.shape != (self.nx1,):
            raise ValueError(
                f"x1f must contain one player state with shape ({self.nx1},)"
            )

        # add damp if too close to player 2:
        dist = np.linalg.norm(self.x[:2] - self.x[self.nx1:self.nx1 + 2])
        if dist < 2*self.d_sep:
            velocity_gain = 2 * velocity_gain
        if np.linalg.norm(self.x2f[0,:2] - self.x[self.nx1:self.nx1 + 2]) < 3*self.d_sep:
            position_gain = 4 * position_gain

        position_error = target[:2] - self.x[:2]
        if self.is_single_integrator:
            control = position_gain * position_error
        else:
            velocity_error = -self.x[2:4]
            control = (
                position_gain * position_error
                + velocity_gain * velocity_error
            )
            
            if np.linalg.norm(self.x[2:4]) > self.vx_max-1.0:
                control = control -1. * self.x[2:4] / np.linalg.norm(self.x[2:4])
                if np.dot(control, self.x[2:4]) > 0:
                    control = control - np.dot(control, self.x[2:4]) * self.x[2:4] / np.linalg.norm(self.x[2:4])**2
                
            # if np.linalg.norm(self.x[2:4]) > max_velocity and np.dot(control, self.x[2:4]) > 0:
            #     control = control - np.dot(control, self.x[2:4]) * self.x[2:4] / np.linalg.norm(self.x[2:4])**2

        u_min = self._as_bounds(self.u_min, self.nu, "u_min")[:self.nu1]
        u_max = self._as_bounds(self.u_max, self.nu, "u_max")[:self.nu1]
        return np.clip(control, u_min, u_max)

    def SimpleController2(self, position_gain=2.0, velocity_gain=5.0, max_velocity=1.0):
        """Return a bounded, goal-tracking control for player 2.

        This is the player-2-symmetric counterpart of ``SimpleController1``.
        It uses player 2's state and target, mirrors the initial x waypoint,
        and returns only player 2's two control components.
        """
        if self.t < 2.3:
            target = np.asarray(self.x2f, dtype=float).reshape(-1).copy()
            target[0] += 2.0
        else:
            target = np.asarray(self.x2f, dtype=float).reshape(-1)        
        
        if self.is_unicycle:
            return self._unicycle_goal_controller(1, target)
        p2 = self.nx1

        if target.shape != (self.nx2,):
            raise ValueError(
                f"x2f must contain one player state with shape ({self.nx2},)"
            )

        distance = np.linalg.norm(self.x[:2] - self.x[p2:p2 + 2])
        if distance < 2 * self.d_sep:
            velocity_gain = 2 * velocity_gain
        if np.linalg.norm(self.x1f[0, :2] - self.x[:2]) < 3*self.d_sep:
            position_gain = 4 * position_gain

        position_error = target[:2] - self.x[p2:p2 + 2]
        if self.is_single_integrator:
            control = position_gain * position_error
        else:
            velocity = self.x[p2 + 2:p2 + 4]
            velocity_error = -velocity
            control = position_gain * position_error + velocity_gain * velocity_error

            if np.linalg.norm(velocity) > self.vx_max - 1.0 and self.t >= 0.8:
                control = control - velocity / np.linalg.norm(velocity)
                if np.dot(control, velocity) > 0:
                    control = (
                        control
                        - np.dot(control, velocity)
                        * velocity
                        / np.linalg.norm(velocity) ** 2
                    )

        u_min = self._as_bounds(self.u_min, self.nu, "u_min")[self.nu1:2 * self.nu1]
        u_max = self._as_bounds(self.u_max, self.nu, "u_max")[self.nu1:2 * self.nu1]
        return np.clip(control, u_min, u_max)

    def SimpleController3(self, position_gain=2.0, velocity_gain=5.0):
        """Return the same bounded goal-tracking controller for player 3."""
        target = np.asarray(self.x3f, dtype=float).reshape(-1).copy()
        if (
            self.t < 2.0
            and (
                self.is_single_integrator
                or self.is_unicycle
                or (
                    abs(self.x[3]) < self.vy_max - 0.5
                    and abs(self.x[2]) < self.vx_max - 0.5
                )
            )
        ):
            target[1] += 2.5
        
        if self.n_players < 3:
            raise ValueError("player 3 is not part of this game")
        if self.is_unicycle:
            return self._unicycle_goal_controller(2, target)
        offset = 2 * self.nx1

        
        position_error = target[:2] - self.x[offset:offset + 2]
        if np.linalg.norm(position_error) < 3*self.d_sep:
            position_gain = 4 * position_gain
        if self.is_single_integrator:
            control = position_gain * position_error
        else:
            velocity = self.x[offset + 2:offset + 4]
            control = position_gain * position_error - velocity_gain * velocity
            

            
        # limit max velocity:
        if np.linalg.norm(velocity) > self.vx_max - 1.0 and self.t <= 3.0:
            control = control - velocity / np.linalg.norm(velocity)
            if np.dot(control, velocity) > 0:
                control = (control - np.dot(control, velocity)* velocity/ np.linalg.norm(velocity) ** 2)
            
        return np.clip(control, self.u_min, self.u_max)

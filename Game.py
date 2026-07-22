import numpy as np
import casadi as ca

class GameDynamics:
    """Two-player 2D single- or double-integrator dynamics with RK4 integration."""

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
        u_min=-2,
        u_max=2,
        L=20.0,
        W=2,
        vx_min=-2,
        vx_max=2,
        vy_min=-2,
        vy_max=2,
        d_sep=0.3,
        dynamics_type=2,
        MaxIterations=50,
    ):
        if dt <= 0:
            raise ValueError("dt must be positive")
        if dynamics_type not in (1, 2):
            raise ValueError("dynamics_type must be 1 for single integrator or 2 for double integrator")

        self.dt = float(dt)
        self.dynamics_type = int(dynamics_type)
        self.nx1 = 2 if self.is_single_integrator else 4
        self.nx2 = self.nx1
        self.nu1 = 2
        self.nu2 = 2
        self.nx = self.nx1 + self.nx2
        self.nu = self.nu1 + self.nu2
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
        
        # shared constranits data
        shared_f_limit = 1.25
        self.u_max_shared = self.u_max*shared_f_limit
        self.u_min_shared = self.u_min*shared_f_limit
        
        self.d_sep = d_sep
                
        self.x0 = x0
        self.x1f = x1f
        self.x2f = x2f

        # Define shared constranits function:
        x_sym = ca.SX.sym('x_sym', self.nx)
        u1_sym = ca.SX.sym('u1_sym', self.nu1)
        u2_sym = ca.SX.sym('u2_sym', self.nu2)

        # self.f_shared = ca.Function('f_shared', [x_sym, u1_sym, u2_sym], [u1_sym[0]+u2_sym[0]-self.u_min_shared, self.u_max_shared-u1_sym[0]-u2_sym[0], u1_sym[1]+u2_sym[1]-self.u_min_shared, self.u_max_shared-u2_sym[1]-u1_sym[1]])
        if self.is_single_integrator:
            v1_sym = u1_sym
            v2_sym = u2_sym
        else:
            v1_sym = x_sym[2:4]
            v2_sym = x_sym[self.nx1+2:self.nx1+4]
        # self.f_shared = ca.Function('f_shared', [x_sym, u1_sym, u2_sym], [self.vy_max**2+self.vx_max**2 - ca.sumsqr(v1_sym) - ca.sumsqr(v2_sym), ca.sumsqr(x_sym[0]-x_sym[self.nx1]) + ca.sumsqr(x_sym[1]-x_sym[self.nx1+1]) - self.d_sep**2])
        # self.f_shared = ca.Function('f_shared', [x_sym, u1_sym, u2_sym], [ca.sumsqr(x_sym[0]-x_sym[self.nx1]) + ca.sumsqr(x_sym[1]-x_sym[self.nx1+1]) - self.d_sep**2])
        self.f_shared = ca.Function('f_shared', [x_sym, u1_sym, u2_sym], [ca.sumsqr(x_sym[0]-x_sym[self.nx1]) + ca.sumsqr(x_sym[1]-x_sym[self.nx1+1]) - self.d_sep**2, x_sym[1]-x_sym[self.nx1+1]])

        # Internal state is [p1x, p1y, p2x, p2y] for single-integrator mode,
        # or [p1x, p1y, v1x, v1y, p2x, p2y, v2x, v2y] for double-integrator mode.
        
        return

    @property
    def is_single_integrator(self):
        return self.dynamics_type == 1

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
        Continuous-time dynamics.

        Single integrator:
            State x = [p1x, p1y, p2x, p2y]
            Input u = [v1x, v1y, v2x, v2y]

        Double integrator:
            State x = [p1x, p1y, v1x, v1y, p2x, p2y, v2x, v2y]
            Input u = [a1x, a1y, a2x, a2y]
        """
        x = np.asarray(x, dtype=float)
        u = np.asarray(u, dtype=float)
        if x.shape != (self.nx,):
            raise ValueError(f"x must have shape ({self.nx},)")
        if u.shape != (self.nu,):
            raise ValueError(f"u must have shape ({self.nu},)")

        if self.is_single_integrator:
            return np.array([u[0], u[1], u[2], u[3]], dtype=float)

        return np.array(
            [
                x[2],
                x[3],
                u[0],
                u[1],
                x[6],
                x[7],
                u[2],
                u[3],
            ],
            dtype=float,
        )

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
        if np.any(u < self.u_min-self.eps) or np.any(u > self.u_max+self.eps):
            self._log_history(u, self.INPUT_OUTSIDE_BOUNDS)
            return self.INPUT_OUTSIDE_BOUNDS

        x = self.x
        dt = self.dt

        # Classical fourth-order Runge-Kutta integration with constant input u.
        k1 = self.dynamics(x, u)
        k2 = self.dynamics(x + 0.5 * dt * k1, u)
        k3 = self.dynamics(x + 0.5 * dt * k2, u)
        k4 = self.dynamics(x + dt * k3, u)

        self.x = x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        self.t += dt

        # Check axis-aligned position bounds for both players.
        xs = np.array([self.x[0], self.x[self.nx1]])
        ys = np.array([self.x[1], self.x[self.nx1 + 1]])

        if (
            np.any(xs < self.x_min-self.eps)
            or np.any(xs > self.x_max+self.eps)
            or np.any(ys < self.y_min-self.eps)
            or np.any(ys > self.y_max+self.eps)
        ):
            self._log_history(u, self.POSITION_OUTSIDE_BOUNDS)
            return self.POSITION_OUTSIDE_BOUNDS

        if not self.is_single_integrator:
            vxs = np.array([self.x[2], self.x[self.nx1 + 2]])
            vys = np.array([self.x[3], self.x[self.nx1 + 3]])
            if (
                np.any(vxs < self.vx_min-self.eps)
                or np.any(vxs > self.vx_max+self.eps)
                or np.any(vys < self.vy_min-self.eps)
                or np.any(vys > self.vy_max+self.eps)
            ):
                self._log_history(u, self.VELOCITY_OUTSIDE_BOUNDS)
                return self.VELOCITY_OUTSIDE_BOUNDS
        
        f_shared = self.f_shared(self.x, self.u[:self.nu1], self.u[self.nu2:])
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

    def SimpleController(self, position_gain=2.0, velocity_gain=5.0, max_velocity=1.0):
        """Return a bounded, goal-tracking control for player 1.

        The single-integrator controller commands velocity proportional to the
        position error.  The double-integrator controller uses position and
        velocity feedback to command acceleration.  In both cases the result
        respects player 1's input bounds.
        """
        if self.t < 0.9 and self.x[3]<self.vy_max-0.02:
            target = np.asarray([-2.0,2,0,0], dtype=float).reshape(-1)
            velocity_gain = 0.0
            position_gain=10.0
        else:
            target = np.asarray(self.x1f, dtype=float).reshape(-1)
        if target.shape != (self.nx1,):
            raise ValueError(
                f"x1f must contain one player state with shape ({self.nx1},)"
            )

        # add damp if too close to player 2:
        dist = np.linalg.norm(self.x[:2] - self.x[self.nx1:self.nx1 + 2])
        if dist < 2*self.d_sep:
            velocity_gain = 2 * velocity_gain
        if np.linalg.norm(self.x2f[0,:2] - self.x[self.nx1:self.nx1 + 2]) < self.d_sep:
            position_gain = 4 * position_gain

        position_error = target[:2] - self.x[:2]
        if self.is_single_integrator:
            control = position_gain * position_error
        else:
            velocity_error = target[2:4] - self.x[2:4]
            control = (
                position_gain * position_error
                + velocity_gain * velocity_error
            )
            
            if np.linalg.norm(self.x[2:4]) > self.vx_max-1.0 and self.t>=0.9:
                control = control -1. * self.x[2:4] / np.linalg.norm(self.x[2:4])
                if np.dot(control, self.x[2:4]) > 0:
                    control = control - np.dot(control, self.x[2:4]) * self.x[2:4] / np.linalg.norm(self.x[2:4])**2
                
            # if np.linalg.norm(self.x[2:4]) > max_velocity and np.dot(control, self.x[2:4]) > 0:
            #     control = control - np.dot(control, self.x[2:4]) * self.x[2:4] / np.linalg.norm(self.x[2:4])**2

        u_min = self._as_bounds(self.u_min, self.nu, "u_min")[:self.nu1]
        u_max = self._as_bounds(self.u_max, self.nu, "u_max")[:self.nu1]
        return np.clip(control, u_min, u_max)

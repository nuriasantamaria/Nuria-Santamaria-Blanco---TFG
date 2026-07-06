import numpy as np
import gymnasium as gym
from gymnasium import spaces
import params as p
from dynamics       import rk4_step, quat_to_dcm, quat_normalize
from orbital        import propagate_mean_elements, kepler_to_cartesian
from magnetic_field import dipole_field_body, bdot_numerical
from perturbations  import compute_C_BV, total_perturbation_torque
from sensors        import fused_magnetometer

_B_MAX    = 3e-5    
_BDOT_MAX = 4.5e-6 
_OBS_DIM  = 12


class UPMSat2Env(gym.Env):

    metadata = {"render_modes": ["human"], "render_fps": 1}

    def __init__(self,
                 dt: float               = 1.0,
                 max_steps: int          = 15000,
                 omega_init_max: float   = 0.1,
                 omega_max: float        = 0.5,
                 n_success: int          = 500,
                 w_detumble: float       = 1.0,
                 omega_detumble_thresh: float = 0.0025,
                 lambda_m: float         = 0.15,
                 lambda_smooth: float    = 0.15,
                 R_success: float        = 5.0,
                 gaussian_amplitude: float = 0.15,
                 gaussian_sigma: float   = 0.002,
                 render_mode             = None,
                 seed: int               = 42,
                 T0_JD: float            = 2451545.0,
                 inertia: np.ndarray | None = None,
                 ):

        super().__init__()

        self.dt              = dt
        self.max_steps       = max_steps
        self.omega_init_max  = omega_init_max
        self.omega_max       = omega_max
        self.n_success       = n_success
        self.w_detumble      = w_detumble
        self.lambda_m        = lambda_m
        self.lambda_smooth   = lambda_smooth
        self.R_success       = R_success
        self.gaussian_amplitude = gaussian_amplitude
        self.gaussian_sigma  = gaussian_sigma
        self.render_mode     = render_mode
        self._rng            = np.random.default_rng(seed)
        self.T0_JD           = T0_JD

        self._omega_detumble_thresh = omega_detumble_thresh   
        self.omega_target           = self._omega_detumble_thresh

        self._orb_a         = p.A_ORB
        self._orb_e         = p.ECC
        self._orb_inc       = p.INC
        self._orb_raan      = p.RAAN0
        self._orb_aop       = p.AOP
        self._orb_ta0       = p.TA0
        self._orb_raan_rate = p.RAAN_RATE

        self._inertia_fixed = inertia
        if self._inertia_fixed is not None:
            self._inertia_base = self._validate_inertia(self._inertia_fixed, "inertia_fixed")
        else:
            self._inertia_base = self._validate_inertia(p.INERTIA, "INERTIA")
        self._inertia  = self._inertia_base.copy()
        self._r_cm_b   = np.zeros(3)

        
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(_OBS_DIM,), dtype=np.float32)
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

        
        self._state          = None   
        self._t              = 0.0
        self._step_count     = 0
        self._success_count  = 0
        self._B_b_prev       = None   
        self._B_b_meas_cache = None   
        self._B_dot_cache    = None   
        self._r_I            = None
        self._v_I            = None
        self._last_m         = np.zeros(3)
        self._last_action    = np.zeros(3, dtype=np.float32)
        self._prev_action    = np.zeros(3, dtype=np.float32)
        self._last_r_bonus   = 0.0

    def _validate_inertia(self,
                          inertia: np.ndarray,
                          name: str = "inertia") -> np.ndarray:
        I = np.array(inertia, dtype=float)
        if I.shape != (3, 3):
            raise ValueError(f"{name} debe tener shape (3,3), recibido: {I.shape}")
        I = 0.5 * (I + I.T)  
        eigvals = np.linalg.eigvalsh(I)
        if np.min(eigvals) <= 1e-10:
            raise ValueError(
                f"{name} debe ser definido positivo; autovalores={eigvals}")
        return I

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self._inertia = self._inertia_base.copy()

        self._orb_a         = p.A_ORB
        self._orb_e         = p.ECC
        self._orb_inc       = p.INC
        self._orb_raan      = p.RAAN0
        self._orb_aop       = p.AOP
        self._orb_ta0       = p.TA0
        self._orb_raan_rate = p.RAAN_RATE

        omega0 = self._rng.uniform(-self.omega_init_max, self.omega_init_max, size=3)

        
        u1, u2, u3 = self._rng.uniform(size=3)
        q0 = quat_normalize(np.array([
            np.sqrt(1 - u1) * np.sin(2 * np.pi * u2),
            np.sqrt(1 - u1) * np.cos(2 * np.pi * u2),
            np.sqrt(u1)     * np.sin(2 * np.pi * u3),
            np.sqrt(u1)     * np.cos(2 * np.pi * u3),
        ]))

        self._state         = np.concatenate([q0, omega0])
        self._t             = 0.0
        self._step_count    = 0
        self._success_count = 0
        self._last_m        = np.zeros(3)
        self._last_action   = np.zeros(3, dtype=np.float32)
        self._prev_action   = np.zeros(3, dtype=np.float32)

        elems = propagate_mean_elements(
            0.0,
            a=self._orb_a, e=self._orb_e, inc=self._orb_inc,
            raan0=self._orb_raan, aop=self._orb_aop, ta0=self._orb_ta0,
            raan_rate=self._orb_raan_rate,
        )
        self._r_I, self._v_I = kepler_to_cartesian(elems)

        C_BI           = quat_to_dcm(self._state[:4])
        B_b_true_0     = dipole_field_body(self._r_I, C_BI, self._t)
        self._B_b_prev = fused_magnetometer(B_b_true_0, rng=self._rng)

        self._B_b_meas_cache = self._B_b_prev.copy()   
        self._B_dot_cache    = np.zeros(3)

        return self._get_observation(), self._get_info()

    def step(self, action: np.ndarray):
        self._prev_action = self._last_action.copy()
        self._last_action = np.clip(action, -1.0, 1.0).astype(np.float32)

        m, torque_control = self._action_to_torque(action)
        self._last_m = m.copy()

        B_b_prev_meas = self._B_b_meas_cache.copy()   

        self._t += self.dt
        elems = propagate_mean_elements(
            self._t,
            a=self._orb_a, e=self._orb_e, inc=self._orb_inc,
            raan0=self._orb_raan, aop=self._orb_aop, ta0=self._orb_ta0,
            raan_rate=self._orb_raan_rate,
        )
        self._r_I, self._v_I = kepler_to_cartesian(elems)

        C_BI        = quat_to_dcm(self._state[:4])
        B_b         = dipole_field_body(self._r_I, C_BI, self._t)
        C_BV        = compute_C_BV(self._r_I, self._v_I, C_BI)
        torque_pert = total_perturbation_torque(
            self._r_I, self._v_I, C_BI, B_b, self._t, self.T0_JD, C_BV,
            inertia=self._inertia, r_cm_b=self._r_cm_b)

        self._state = rk4_step(self._state, torque_control, torque_pert, self.dt)
        self._step_count += 1

        C_BI_new     = quat_to_dcm(self._state[:4])
        B_b_true_new = dipole_field_body(self._r_I, C_BI_new, self._t)
        B_b_meas_new = fused_magnetometer(B_b_true_new, rng=self._rng)

        self._B_b_prev       = B_b_prev_meas   
        self._B_b_meas_cache = B_b_meas_new

        
        self._B_dot_cache = bdot_numerical(
            self._B_b_prev, self._B_b_meas_cache, max(self.dt, 1e-9))

        obs                   = self._get_observation()
        reward                = self._compute_reward(m)
        terminated, truncated = self._check_done()
        info                  = self._get_info()

        return obs, reward, terminated, truncated, info

    def _action_to_torque(self, action: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """m = clip(action,-1,1)·M_MAX [A·m²]; T = m × B_b [N·m]."""
        m    = np.clip(action, -1.0, 1.0) * p.M_MAX_MTQ
        C_BI = quat_to_dcm(self._state[:4])
        B_b  = dipole_field_body(self._r_I, C_BI, self._t)
        T    = np.cross(m, B_b)
        return m, T

    def _get_observation(self) -> np.ndarray:
        """obs 12D hardware-realizable: [B_curr, B_prev, Ḃ, a_{k-1}] (3+3+3+3), todo normalizado."""
        return np.concatenate([
            np.clip(self._B_b_meas_cache / _B_MAX,   -1.0, 1.0),  
            np.clip(self._B_b_prev       / _B_MAX,   -1.0, 1.0),  
            np.clip(self._B_dot_cache    / _BDOT_MAX, -1.0, 1.0),  
            np.clip(self._last_action,               -1.0, 1.0),  
        ]).astype(np.float32)

    def _compute_reward(self, m: np.ndarray) -> float:
        omega_real = np.linalg.norm(self._state[4:])  

        r_det = -np.clip(omega_real / self.omega_init_max, 0.0, 2.0)

        r_energy = -self.lambda_m * (np.linalg.norm(m) / p.M_MAX_MTQ) ** 2

        delta_a  = np.linalg.norm(self._last_action - self._prev_action)
        r_smooth = -self.lambda_smooth * delta_a ** 2

        r_bonus = 0.0
        thresh  = self._omega_detumble_thresh

        if omega_real < thresh:
            self._success_count += 1
            depth   = 1.0 - (omega_real / thresh)   
            sat     = min(self._success_count / self.n_success, 1.0)
            r_bonus = self.R_success * depth * (1.0 + sat)
        else:
            self._success_count = 0
        self._last_r_bonus = r_bonus

        r_gaussian = self.gaussian_amplitude * np.exp(-(omega_real ** 2) / (2 * self.gaussian_sigma ** 2))

        return float(self.w_detumble * r_det + r_energy + r_smooth + r_bonus + r_gaussian)

    def _check_done(self) -> tuple[bool, bool]:
        omega_norm = np.linalg.norm(self._state[4:])
        truncated  = bool(
            self._step_count >= self.max_steps or omega_norm > self.omega_max)
        return False, truncated

    def _get_info(self) -> dict:
        omega      = self._state[4:]
        omega_norm = float(np.linalg.norm(omega))
        C_BI       = quat_to_dcm(self._state[:4])
        B_b        = dipole_field_body(self._r_I, C_BI, self._t)

        B_dot  = self._B_dot_cache if self._B_dot_cache is not None else np.zeros(3)

        h_I    = np.cross(self._r_I, self._v_I)
        h_norm = np.linalg.norm(h_I) + 1e-12
        h_b    = C_BI @ (h_I / h_norm)
        alignment = float(np.dot(h_b, [0.0, 0.0, 1.0]))

        return {
            "t":                    self._t,
            "omega_norm":           omega_norm,
            "omega_deg_s":          float(np.rad2deg(omega_norm)),
            "omega_x":              float(omega[0]),
            "omega_y":              float(omega[1]),
            "omega_z":              float(omega[2]),
            "m_norm":               float(np.linalg.norm(self._last_m)),
            "q":                    self._state[:4].copy(),
            "B_x":                  float(B_b[0]),
            "B_y":                  float(B_b[1]),
            "B_z":                  float(B_b[2]),
            "bdot":                 B_dot.copy(),
            "success_steps":        self._success_count,
            "r_bonus":              self._last_r_bonus,
            "alignment":            alignment,
        }

    def render(self):
        if self.render_mode == "human":
            omega_norm  = np.linalg.norm(self._state[4:])
            print(f"  t={self._t:7.1f}s  "
                  f"|ω|_real={np.rad2deg(omega_norm):7.3f} °/s  "
                  f"|m|={np.linalg.norm(self._last_m):5.2f} A·m²")

    def close(self):
        pass
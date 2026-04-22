"""
LiDAR Sensor Simulation Module
==============================
Professional-grade 2D LiDAR simulation for grid-map environments.

Design Principles
-----------------
1. **ROS-Aligned Data Model**: Output follows ``sensor_msgs/LaserScan`` semantics
   (ranges, intensities, angle_min/max/increment, time_increment, etc.).
2. **Resolution-Adaptive Ray-Casting**: Step size is automatically clamped to
   sub-cell scale so that thin obstacles are never skipped.
3. **Physics-Informed Noise**: Distance-dependent Gaussian noise, angular jitter,
   systematic bias, and random drop-outs model real-world RPLIDAR A1 behaviour.
4. **Backwards Compatible**: ``scan()`` returns a ``LaserScan`` object that can
   still be unpacked as ``ranges, angles = lidar.scan(...)``.

Reference (RPLIDAR A1 Spec)
---------------------------
* Range: 0.15 m ~ 12.0 m
* Scan frequency: 5.5 Hz (typical)
* Angular resolution: ~0.5° @ 5.5 Hz  (360° / 720 beams)
* Distance accuracy: ±2 % (≤ 3 m), ±3 % (3 m ~ 12 m)
* Angle accuracy: < 0.5°
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Protocol, Tuple, Iterator

import numpy as np


# --------------------------------------------------------------------------- #
#  Data Structures
# --------------------------------------------------------------------------- #

@dataclass
class LidarConfig:
    """Complete parameter set for a 2D spinning LiDAR (default = RPLIDAR A1)."""

    # -- Geometric limits -----------------------------------------------------
    range_min: float = 0.15          # [m]
    range_max: float = 12.0          # [m]
    angle_min: float = -np.pi        # [rad]
    angle_max: float = np.pi         # [rad]
    angle_increment: float = np.deg2rad(0.5)  # [rad]

    # -- Temporal parameters --------------------------------------------------
    scan_frequency: float = 5.5      # [Hz]
    frame_id: str = "lidar_link"

    # -- Distance noise (relative, from data-sheet) ---------------------------
    noise_ratio_near: float = 0.02   # ≤ 3 m  →  σ = 2 % · d
    noise_ratio_far: float = 0.03    # > 3 m  →  σ = 3 % · d
    noise_transition_dist: float = 3.0  # [m]

    # -- Systematic errors ----------------------------------------------------
    range_bias_mean: float = 0.0     # permanent firmware calibration bias [m]
    range_bias_std: float = 0.005    # unit-to-unit variance [m]
    angle_noise_std: float = np.deg2rad(0.25)  # encoder / motor jitter [rad]

    # -- Data integrity -------------------------------------------------------
    dropout_rate: float = 0.005      # probability of invalid return [0~1]

    # -- Intensity model ------------------------------------------------------
    intensity_scale: float = 255.0
    intensity_decay_factor: float = 0.15  # exponential attenuation [1/m]
    intensity_sigma: float = 8.0     # read-out noise [grey-level]

    def __post_init__(self):
        if self.scan_frequency <= 0:
            raise ValueError("scan_frequency must be positive")
        if self.range_max <= self.range_min:
            raise ValueError("range_max must be greater than range_min")


@dataclass
class LaserScan:
    """Standard 2D laser scan frame (ROS ``sensor_msgs/LaserScan`` compatible).

    Attributes
    ----------
    stamp : float
        Simulation time when the scan was triggered [s].
    frame_id : str
        Coordinate frame name.
    angle_min / angle_max : float
        First / last beam angle in the *sensor* frame [rad].
    angle_increment : float
        Nominal angular step between beams [rad].
    time_increment : float
        Time between individual range measurements [s].
    scan_time : float
        Time to complete one full 360° sweep [s].
    range_min / range_max : float
        Sensor limits [m].
    ranges : np.ndarray, shape (N,)
        Measured distances [m].  ``inf``  = no obstacle, ``nan`` = dropout.
    intensities : np.ndarray, shape (N,)
        Return intensities [0, 255].
    angles : np.ndarray, shape (N,)
        *Actual* beam angles including angular noise [rad].  This is kept in
        the sensor frame so that ``global_angle = angles + pose[2]``.
    """

    stamp: float
    frame_id: str
    angle_min: float
    angle_max: float
    angle_increment: float
    time_increment: float
    scan_time: float
    range_min: float
    range_max: float
    ranges: np.ndarray
    intensities: np.ndarray
    angles: np.ndarray

    # ---- Backwards-compatibility helpers -----------------------------------

    def __iter__(self) -> Iterator[np.ndarray]:
        """Allow ``ranges, angles = lidar.scan(...)`` unpacking."""
        yield self.ranges
        yield self.angles

    def __len__(self) -> int:
        return len(self.ranges)


class BaseLidar(Protocol):
    """Protocol / abstract interface for 2D LiDAR sensors."""

    @property
    def config(self) -> LidarConfig: ...

    @property
    def range_min(self) -> float: ...

    @property
    def range_max(self) -> float: ...

    def ready(self, sim_time: float) -> bool: ...

    def scan(self, robot_pose: np.ndarray, env) -> LaserScan: ...


# --------------------------------------------------------------------------- #
#  RPLIDAR A1 Implementation
# --------------------------------------------------------------------------- #

class LidarA1:
    """思岚 (Slamtec) RPLIDAR A1 simulation model.

    The sensor performs **resolution-adaptive vectorised ray-casting** on a
    grid-map environment.  The step size is chosen automatically from
    ``env.res`` so that no occupied cell thinner than one pixel can be missed.

    Noise is injected according to the official data-sheet:

    * distance : Gaussian whose σ is 2 % of ``d`` (≤ 3 m) or 3 % (> 3 m)
    * angle    : Gaussian jitter σ = 0.25°
    * dropout  : Bernoulli trial with ``p = dropout_rate``

    Parameters
    ----------
    config : LidarConfig, optional
        Override any default parameter.  If omitted, A1 defaults are used.
    """

    def __init__(self, config: Optional[LidarConfig] = None):
        self.cfg = config if config is not None else LidarConfig()

        # temporal state
        self._scan_period = 1.0 / self.cfg.scan_frequency
        self._last_scan_time = -self._scan_period

        # pre-compute ideal (noise-free) local angles — saves work every frame
        num_beams = max(
            0,
            int(
                np.ceil(
                    (self.cfg.angle_max - self.cfg.angle_min)
                    / self.cfg.angle_increment
                )
            ),
        )
        self._num_beams = num_beams
        self._ideal_angles = np.linspace(
            self.cfg.angle_min,
            self.cfg.angle_min + num_beams * self.cfg.angle_increment,
            num_beams,
            endpoint=False,
            dtype=np.float32,
        )

        # systematic range bias drawn once per sensor instance
        # (models factory calibration variance)
        self._system_range_bias = float(
            np.random.normal(self.cfg.range_bias_mean, self.cfg.range_bias_std)
        )

    # -- Public properties ----------------------------------------------------

    @property
    def config(self) -> LidarConfig:
        return self.cfg

    @property
    def range_min(self) -> float:
        """Shortest measurable distance [m]."""
        return self.cfg.range_min

    @property
    def range_max(self) -> float:
        """Longest measurable distance [m]."""
        return self.cfg.range_max

    # -- Scheduling -----------------------------------------------------------

    def ready(self, sim_time: float) -> bool:
        """Return ``True`` when ``sim_time`` has crossed the next 5.5 Hz tick.

        The internal latch is updated only on ``True`` returns, guaranteeing
        a stable fixed-rate trigger.
        """
        if sim_time - self._last_scan_time >= self._scan_period:
            self._last_scan_time = sim_time
            return True
        return False

    # -- Noise generators -----------------------------------------------------

    def _distance_noise(self, distances: np.ndarray) -> np.ndarray:
        """Return distance-dependent Gaussian noise (vectorised)."""
        ratios = np.where(
            distances <= self.cfg.noise_transition_dist,
            self.cfg.noise_ratio_near,
            self.cfg.noise_ratio_far,
        )
        sigmas = distances * ratios
        return np.random.normal(0.0, sigmas, size=distances.shape).astype(np.float32)

    def _angular_noise(self, n: int) -> np.ndarray:
        """Return angular jitter for *n* beams."""
        if self.cfg.angle_noise_std <= 0.0:
            return np.zeros(n, dtype=np.float32)
        return np.random.normal(
            0.0, self.cfg.angle_noise_std, size=n
        ).astype(np.float32)

    # -- Core ray-casting -----------------------------------------------------

    def _raycast(
        self, rx: float, ry: float, rtheta: float, env
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Vectorised grid-map ray-casting.

        Returns
        -------
        ranges : np.ndarray
            Measured distances (``inf`` = no hit).
        intensities : np.ndarray
            Simulated return intensity.
        local_angles : np.ndarray
            Actual beam angles in sensor frame (includes angular noise).
        """
        # ---- 1. Angle preparation -------------------------------------------
        local_angles = self._ideal_angles + self._angular_noise(self._num_beams)
        global_angles = local_angles + float(rtheta)

        cos_a = np.cos(global_angles)
        sin_a = np.sin(global_angles)

        # ---- 2. Adaptive step size ------------------------------------------
        # Grid cell size in metres.  Step must be ≤ cell size so that every
        # pixel on the ray is visited.  We use 0.5×cell as a safety margin.
        cell_size = 1.0 / float(getattr(env, "res", 50.0))
        step = cell_size * 0.5
        max_steps = int(np.ceil(self.cfg.range_max / step))

        # ---- 3. Buffers ------------------------------------------------------
        ranges = np.full(self._num_beams, np.inf, dtype=np.float32)
        intensities = np.zeros(self._num_beams, dtype=np.float32)
        finished = np.zeros(self._num_beams, dtype=bool)

        # Pre-calculate origin in pixel space (only for documentation; not used
        # in the hot loop because we operate in metric space then quantise).
        for i in range(1, max_steps + 1):
            active = ~finished
            if not np.any(active):
                break

            d = i * step
            if d < self.cfg.range_min:
                continue

            # Metric coordinates of *active* beams at this depth
            tx = rx + d * cos_a[active]
            ty = ry + d * sin_a[active]

            # Quantise to pixel indices (round-to-nearest)
            px = np.floor(tx * env.res + 0.5).astype(np.int32)
            py = np.floor(ty * env.res + 0.5).astype(np.int32)

            # Boundary mask
            in_bounds = (
                (px >= 0)
                & (px < env.size_px)
                & (py >= 0)
                & (py < env.size_px)
            )
            if not np.any(in_bounds):
                continue

            # Map active-array indices → global beam indices
            active_idx = np.nonzero(active)[0]
            valid_idx = active_idx[in_bounds]
            valid_px = px[in_bounds]
            valid_py = py[in_bounds]

            # Occupancy test
            hit = env.grid_map[valid_py, valid_px] > 0
            if not np.any(hit):
                continue

            hit_idx = valid_idx[hit]

            # ---- Measurement noise ----------------------------------------
            noise = self._distance_noise(
                np.full(len(hit_idx), d, dtype=np.float32)
            )
            measured = d + noise + self._system_range_bias
            measured = np.clip(
                measured, self.cfg.range_min, self.cfg.range_max
            )
            ranges[hit_idx] = measured

            # ---- Intensity model ------------------------------------------
            # Real LiDAR intensity falls off with distance (inverse-square-like
            # + medium absorption).  We use an exponential decay + read noise.
            base_intensity = self.cfg.intensity_scale * math.exp(
                -self.cfg.intensity_decay_factor * d
            )
            intensity_noise = np.random.normal(
                0.0, self.cfg.intensity_sigma, size=len(hit_idx)
            )
            intensities[hit_idx] = np.clip(
                base_intensity + intensity_noise, 0.0, self.cfg.intensity_scale
            ).astype(np.float32)

            finished[hit_idx] = True

        return ranges, intensities, local_angles

    # -- Public API -----------------------------------------------------------

    def scan(self, robot_pose: np.ndarray, env) -> LaserScan:
        """Execute one full 360° sweep and return a ``LaserScan`` frame.

        Parameters
        ----------
        robot_pose : np.ndarray, shape (3,)
            ``[x, y, theta]`` in world frame [m, m, rad].
        env : MapEnvironment
            Grid-map container with attributes ``res``, ``size_px``,
            ``grid_map``.

        Returns
        -------
        LaserScan
            Standardised scan object.  Can be unpacked as
            ``ranges, angles = scan`` for legacy code.
        """
        rx, ry, rtheta = robot_pose
        ranges, intensities, local_angles = self._raycast(rx, ry, rtheta, env)

        # ---- Drop-out simulation (occlusion / low reflectivity) -------------
        if self.cfg.dropout_rate > 0.0:
            mask = np.random.rand(self._num_beams) < self.cfg.dropout_rate
            ranges[mask] = np.nan
            intensities[mask] = 0.0

        # ---- Timing metadata ------------------------------------------------
        time_increment = (
            self._scan_period / self._num_beams if self._num_beams > 0 else 0.0
        )

        return LaserScan(
            stamp=self._last_scan_time,
            frame_id=self.cfg.frame_id,
            angle_min=self.cfg.angle_min,
            angle_max=self.cfg.angle_max,
            angle_increment=self.cfg.angle_increment,
            time_increment=time_increment,
            scan_time=self._scan_period,
            range_min=self.cfg.range_min,
            range_max=self.cfg.range_max,
            ranges=ranges,
            intensities=intensities,
            angles=local_angles,
        )

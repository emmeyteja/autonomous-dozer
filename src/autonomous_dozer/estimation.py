"""Bounded cone/ridge inference from partial observations, never Terrain truth.

Intervals are conditional linearized/ensemble diagnostics, not calibrated
confidence bounds. Unsupported geometry and weak constraints request rescanning.
"""
from dataclasses import dataclass, field
import math
import numpy as np
from numpy.typing import NDArray
from .domain import Rect, positive
from .observations import Observation


@dataclass(frozen=True)
class GroundReference:
    elevation_m: float = 0.0
    std_m: float = 0.0

    def __post_init__(self):
        if not math.isfinite(self.elevation_m) or not math.isfinite(self.std_m) or self.std_m < 0:
            raise ValueError("Flat-ground reference must be finite with nonnegative uncertainty")


@dataclass(frozen=True)
class MassPrior:
    mass_kg: float
    bulk_density_kg_m3: float
    relative_volume_std: float = 0.1

    def __post_init__(self):
        for name in self.__dict__:
            positive(name, getattr(self, name))
        if self.relative_volume_std >= 1:
            raise ValueError("Relative volume uncertainty must be < 1")
        positive("derived volume", self.volume_m3)
        positive("derived volume standard deviation", self.std_m3)

    @property
    def volume_m3(self):
        return self.mass_kg/self.bulk_density_kg_m3

    @property
    def std_m3(self):
        return self.volume_m3*self.relative_volume_std


@dataclass(frozen=True)
class FitConfig:
    starts_per_model: int = 6
    max_nfev: int = 250
    max_samples: int = 1200
    max_grid_cells: int = 250_000
    min_elevated_samples: int = 12
    noise_floor_m: float = 0.02
    max_pile_rmse_m: float = 0.08
    max_hidden_fraction: float = 0.65
    max_relative_interval_width: float = 0.5
    plausible_score_delta: float = 0.1
    seed: int = 17

    def __post_init__(self):
        for name in ("starts_per_model", "max_nfev", "max_samples", "max_grid_cells", "min_elevated_samples"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.starts_per_model < 2 or self.min_elevated_samples < 8:
            raise ValueError("Require >=2 starts per model and >=8 elevated samples")
        if self.max_samples < 2*self.min_elevated_samples:
            raise ValueError("Sample budget must accommodate elevated and free-ground samples")
        for name in ("noise_floor_m", "max_pile_rmse_m", "max_relative_interval_width", "plausible_score_delta"):
            positive(name, getattr(self, name))
        if not math.isfinite(self.max_hidden_fraction) or not 0 <= self.max_hidden_fraction <= 1:
            raise ValueError("Hidden-fraction limit must lie in [0, 1]")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("Fit seed must be a nonnegative integer")


@dataclass(frozen=True)
class PileEstimate:
    status: str
    reasons: tuple[str, ...]
    source_version: int
    stamp_s: float
    frame_id: str
    selected_model: str | None = None
    parameters: dict = field(default_factory=dict)
    diagnostics: dict = field(default_factory=dict)
    candidates: tuple[dict, ...] = ()
    predicted_height_m: NDArray[np.float64] | None = None  # Excess above ground, not site elevation.
    observed_mask: NDArray[np.bool_] | None = None

    def __post_init__(self):
        for name in ("predicted_height_m", "observed_mask"):
            if getattr(self, name) is not None:
                a = np.array(getattr(self, name), copy=True)
                a.flags.writeable = False
                object.__setattr__(self, name, a)

    @property
    def rescan_required(self):
        return self.status != "estimated"


def _height(x, y, p):
    # cone: cx,cy,radius,slope; ridge: same + half_length,yaw
    cx, cy, b, slope = p[:4]
    a, yaw = p[4:] if len(p) == 6 else (0.0, 0.0)
    u = math.cos(yaw)*(x-cx)+math.sin(yaw)*(y-cy)
    v = -math.sin(yaw)*(x-cx)+math.cos(yaw)*(y-cy)
    return slope*np.maximum(b-np.hypot(np.maximum(np.abs(u)-a, 0), v), 0)


def _volume(p):
    b, slope = p[2:4]
    a = p[4] if len(p) == 6 else 0.0
    return float(slope*(math.pi*b**3/3+2*a*b**2))


def _margins(p, area):
    a, yaw = p[4:] if len(p) == 6 else (0.0, 0.0)
    hx, hy = a*abs(math.cos(yaw))+p[2], a*abs(math.sin(yaw))+p[2]
    return np.array([p[0]-hx-area.xmin_m, area.xmax_m-p[0]-hx,
                     p[1]-hy-area.ymin_m, area.ymax_m-p[1]-hy])


def _huber_residual(r):
    # Robustify only data residuals; physical/mass priors remain quadratic.
    a = np.abs(r)
    return np.sign(r)*np.sqrt(np.where(a <= 1, r*r, 2*a-1))


def fit_pile(observation: Observation, work_area: Rect, *, ground=GroundReference(),
             mass: MassPrior | None = None, config=FitConfig()) -> PileEstimate:
    """Fit unclipped cone/stadium-ridge excess heights in a known work area.

    Slope prior is 32 +/- 7.5 degrees, bounded to 25..40 degrees. Ground is a
    horizontal reference in the same datum as observed heights. No camera,
    previous-terrain differencing, multiple piles or capped plateau model yet.
    """
    if not isinstance(observation, Observation):
        raise TypeError("fit_pile consumes Observation, never a ground-truth Terrain")
    obs, g = observation, observation.grid
    if not g.contains(work_area):
        raise ValueError("Known work area must be inside the observation grid")
    if g.nx*g.ny > config.max_grid_cells:
        raise ValueError("Estimator grid-cell budget exceeded")
    if min(work_area.xmax_m-work_area.xmin_m, work_area.ymax_m-work_area.ymin_m) <= g.resolution_m:
        raise ValueError("Work area is too small for the model")
    x, y = g.centers()
    known = obs.observed & g.mask(work_area)
    z = obs.height_m-ground.elevation_m
    if not np.all(np.isfinite(z[known])):
        raise ValueError("Ground-relative observations must be finite")
    sigma = np.hypot(np.hypot(obs.height_std_m, ground.std_m), config.noise_floor_m)
    elevated = known & (z > 3*sigma)
    free = known & (np.abs(z) <= 3*sigma)
    common = dict(source_version=obs.source_version, stamp_s=obs.stamp_s, frame_id=g.frame_id)
    counts = {"observed_cells_in_work_area": int(known.sum()), "elevated_samples": int(elevated.sum()),
              "free_ground_samples": int(free.sum())}
    if elevated.sum() < config.min_elevated_samples:
        return PileEstimate("insufficient_data", ("insufficient_elevated_observations",),
                            diagnostics=counts, **common)
    # Fixed budgets and deterministic sampling; unknown cells never enter residuals.
    rng = np.random.default_rng(config.seed)
    elevated_idx, other_idx = np.flatnonzero(elevated), np.flatnonzero(known & ~elevated)
    n_high = min(len(elevated_idx), config.max_samples//2)
    high = rng.choice(elevated_idx, n_high, replace=False)
    other = rng.choice(other_idx, min(len(other_idx), config.max_samples-n_high), replace=False)
    indices = np.concatenate([high, other])
    xx, yy, zz, ss = (a.ravel()[indices] for a in (x, y, z, sigma))
    free_sample = free.ravel()[indices]
    width, depth = work_area.xmax_m-work_area.xmin_m, work_area.ymax_m-work_area.ymin_m
    radius_max = min(width, depth)/2
    low = np.array([work_area.xmin_m, work_area.ymin_m, g.resolution_m/2, math.tan(math.radians(25))])
    upper = np.array([work_area.xmax_m, work_area.ymax_m, radius_max, math.tan(math.radians(40))])
    pts = np.column_stack([x[elevated], y[elevated]])
    weights = np.maximum(z[elevated], 0)
    center = np.average(pts, axis=0, weights=weights)
    eigenvalues, eigenvectors = np.linalg.eigh(np.cov(pts.T))
    axis = eigenvectors[:, -1]
    yaw0 = math.atan2(axis[1], axis[0])
    yaw0 = (yaw0+math.pi/2) % math.pi-math.pi/2
    peak = float(np.percentile(z[elevated], 98))
    radius0 = np.clip(peak/math.tan(math.radians(32)), low[2]*1.1, radius_max*0.9)
    crest = pts[int(np.argmax(z[elevated]))]
    from scipy.optimize import least_squares
    fits = []
    for name in ("cone", "ridge"):
        lo, hi = low.copy(), upper.copy()
        if name == "ridge":
            lo, hi = np.r_[lo, 0, -math.pi/2], np.r_[hi, math.hypot(width, depth)/2, math.pi/2]
        def residual(p):
            predicted = _height(xx, yy, p)
            data = _huber_residual((predicted-zz)/ss)
            # Only measured, approximately flat cells contribute free-ground constraints.
            free_penalty = np.maximum(predicted[free_sample]-3*ss[free_sample], 0)/ss[free_sample]
            constraints = np.minimum(_margins(p, work_area), 0)/(config.noise_floor_m*0.1)
            slope_prior = [(math.atan(p[3])-math.radians(32))/math.radians(7.5)]
            mass_residual = [( _volume(p)-mass.volume_m3)/mass.std_m3] if mass else []
            return np.r_[data, free_penalty, constraints, slope_prior, mass_residual]
        for start in range(config.starts_per_model):
            c0 = crest if start == 0 else center+rng.normal(0, radius0*0.3, 2)
            b0 = radius0*(1 if start == 0 else rng.uniform(0.8, 1.3))
            p0 = np.r_[c0, b0, math.tan(math.radians(32))]
            if name == "ridge":
                a0 = max(g.resolution_m, math.sqrt(max(eigenvalues[-1], 0))-radius0/2)
                yaw = yaw0 if start == 0 else rng.uniform(-math.pi/2, math.pi/2)
                if start == 1:
                    # Probe an extended ridge away from the sensor. A visible end
                    # cap can otherwise look like a well-constrained small cone.
                    away = center-np.array([obs.sensor_pose.x_m, obs.sensor_pose.y_m])
                    yaw = (math.atan2(away[1], away[0])+math.pi/2) % math.pi-math.pi/2
                    a0 = max(a0, radius0)
                    direction = np.array([math.cos(yaw), math.sin(yaw)])
                    if direction@away < 0:
                        direction = -direction
                    p0[:2] = crest+a0*direction
                p0 = np.r_[p0, a0, yaw]
            p0 = np.clip(p0, lo+1e-7, hi-1e-7)
            result = least_squares(residual, p0, bounds=(lo, hi), method="trf", loss="linear",
                                   x_scale="jac", max_nfev=config.max_nfev)
            if not result.success or not np.all(np.isfinite(result.x)):
                continue
            p = result.x
            # Evaluate against ALL known cells, not only sampled optimizer data.
            prediction = _height(x, y, p)
            relevant = known & ((z > 3*sigma) | (prediction > 3*sigma))
            rmse = float(np.sqrt(np.mean((prediction[relevant]-z[relevant])**2)))
            raw_residual = (prediction[known]-z[known])/sigma[known]
            data_cost = float(np.mean(_huber_residual(raw_residual)**2))
            mass_z = abs(_volume(p)-mass.volume_m3)/mass.std_m3 if mass else None
            constraint_cost = float(np.sum((np.minimum(_margins(p, work_area), 0)/(config.noise_floor_m*0.1))**2))
            prior_cost = ((math.atan(p[3])-math.radians(32))/math.radians(7.5))**2
            score = data_cost+(len(p)*math.log(int(known.sum()))+constraint_cost+prior_cost
                              +(mass_z**2 if mass_z is not None else 0))/int(known.sum())
            fits.append(dict(model=name, p=p, result=result, score=score, pile_rmse_m=rmse,
                             analytic_volume_m3=_volume(p), mass_z=mass_z))
    if not fits:
        return PileEstimate("optimization_failed", ("no_converged_multistart_fit",), diagnostics=counts, **common)
    # Prefer simpler model for near-equivalent fits; score includes parameter count.
    fits.sort(key=lambda f: f["score"])
    best = fits[0]
    p, solution = best["p"], best["result"]
    predicted = _height(x, y, p)
    total = float(predicted.sum()*g.area_m2)
    visible = float(predicted[obs.observed].sum()*g.area_m2)
    hidden = total-visible
    # Conditional local linear uncertainty plus plausible alternative-fit envelope.
    singular = np.linalg.svd(solution.jac, compute_uv=False)
    rank = int(np.sum(singular > singular[0]*1e-7)) if singular.size and singular[0] else 0
    local_std = None
    if rank == len(p):
        covariance = np.linalg.pinv(solution.jac.T@solution.jac)
        gradient = np.zeros(len(p))
        for i in range(len(p)):
            delta = 1e-5*max(1, abs(p[i]))
            plus, minus = p.copy(), p.copy(); plus[i] += delta; minus[i] -= delta
            gradient[i] = (_volume(plus)-_volume(minus))/(2*delta)
        local_std = float(math.sqrt(max(0, gradient@covariance@gradient)))
    footprint_area = math.pi*p[2]**2+(4*p[4]*p[2] if len(p) == 6 else 0)
    # Ground datum error is shared by every cell, not independent noise that
    # vanishes with sample count. Retain a conservative common-mode sensitivity.
    ground_volume_floor = float(footprint_area*ground.std_m)
    if local_std is not None:
        local_std = math.hypot(local_std, ground_volume_floor)
    plausible = [f for f in fits if f["score"] <= best["score"]+config.plausible_score_delta]
    volumes = [f["analytic_volume_m3"] for f in plausible]
    interval = [max(0, min(volumes)), max(volumes)]
    if local_std is not None:
        interval = [max(0, min(interval[0], _volume(p)-1.96*local_std)),
                    max(interval[1], _volume(p)+1.96*local_std)]
    reasons = []
    if int(free.sum()) < config.min_elevated_samples:
        reasons.append("insufficient_observed_free_ground")
    if best["pile_rmse_m"] > max(config.max_pile_rmse_m, 3*float(np.median(sigma[known]))):
        reasons.append("shape_model_mismatch")
    if rank < len(p):
        reasons.append("locally_unidentifiable_parameters")
    if min(_margins(p, work_area)) < g.resolution_m*0.1:
        reasons.append("footprint_near_or_outside_work_boundary")
    if min(p[3]-low[3], upper[3]-p[3]) < 0.01*(upper[3]-low[3]):
        reasons.append("slope_at_prior_bound")
    if total > 0 and hidden/total > config.max_hidden_fraction:
        reasons.append("excessive_hidden_fraction")
    if (interval[1]-interval[0])/max(_volume(p), 1e-12) > config.max_relative_interval_width:
        reasons.append("ambiguous_volume")
    if mass and best["mass_z"] > 3:
        reasons.append("mass_prior_conflict")
    status = "inconsistent_mass" if "mass_prior_conflict" in reasons else ("needs_rescan" if reasons else "estimated")
    parameters = dict(cx_m=float(p[0]), cy_m=float(p[1]), radius_m=float(p[2]),
                      slope_angle_deg=math.degrees(math.atan(p[3])), peak_height_m=float(p[2]*p[3]),
                      half_length_m=float(p[4]) if len(p) == 6 else 0.0,
                      yaw_rad=float(p[5]) if len(p) == 6 else 0.0)
    summaries = tuple({k: f[k] for k in ("model", "score", "pile_rmse_m", "analytic_volume_m3", "mass_z")} for f in fits)
    diagnostics = {**counts, "sampled_cells": len(indices), "converged_starts": len(fits),
                   "requested_starts": config.starts_per_model*2,
                   "pile_rmse_m": best["pile_rmse_m"], "total_volume_m3": total,
                   "visible_volume_m3": visible, "hidden_volume_m3": hidden,
                   "analytic_volume_m3": _volume(p), "mass_residual_std": best["mass_z"],
                   "jacobian_rank": rank, "parameter_count": len(p),
                   "conditional_local_volume_std_m3": local_std,
                   "common_ground_volume_std_floor_m3": ground_volume_floor,
                   "conditional_volume_interval_m3": interval,
                   "interval_method": "local_linear_1.96_sigma_plus_near_fit_envelope_not_calibrated",
                   "predicted_height_reference": "excess_above_supplied_flat_ground",
                   "estimated_cells_are_not_observations": True,
                   "volume_usable_without_rescan": not bool(reasons)}
    return PileEstimate(status, tuple(reasons), selected_model=best["model"], parameters=parameters,
                        diagnostics=diagnostics, candidates=summaries, predicted_height_m=predicted,
                        observed_mask=obs.observed, **common)

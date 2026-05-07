"""
ODE utility functions for dynamic 4D Gaussian Splatting.

Implements the closed-form / matrix-exponential solvers described in:
  "Dynamic 3D Gaussian Splatting with Explicit Real-Valued ODE-Based
   Mean and Covariance Evolution"

Mean evolution (3D real-valued ODE):
  dx/dtau = A*x + b,   A in R^{3x3}, b in R^3, x(0) = x0
  Solved via augmented 4x4 matrix exponential.

Covariance evolution:
  Log-scale:   ds_j/dtau = kappa_j  =>  s_j(tau) = s_j^0 + kappa_j * tau
  Orientation: dR/dtau  = omega_hat * R,  omega in R^3
               => R(tau) = exp(omega_hat * tau) * R^0   (SO(3) exponential)

All functions operate on batches of N Gaussians and are fully differentiable.
"""

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def quaternion_to_rotation_matrix(q: torch.Tensor) -> torch.Tensor:
    """
    Convert unit quaternions to rotation matrices.

    Args:
        q: (N, 4) unit quaternions in [w, x, y, z] convention.
    Returns:
        R: (N, 3, 3) rotation matrices.
    """
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = torch.stack([
        1 - 2*(y*y + z*z),  2*(x*y - z*w),      2*(x*z + y*w),
        2*(x*y + z*w),      1 - 2*(x*x + z*z),   2*(y*z - x*w),
        2*(x*z - y*w),      2*(y*z + x*w),        1 - 2*(x*x + y*y),
    ], dim=-1).reshape(-1, 3, 3)
    return R


def quaternion_multiply(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """
    Hamilton product of two quaternion tensors.

    Args:
        q1, q2: (N, 4) quaternions in [w, x, y, z] convention.
    Returns:
        (N, 4) product quaternion q1 * q2.
    """
    w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
    return torch.stack([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], dim=-1)


# ---------------------------------------------------------------------------
# Mean ODE: 3D real-valued closed-form solver via matrix exponential
# ---------------------------------------------------------------------------

def solve_3d_ode(
    A_flat: torch.Tensor,   # (N, 9)   row-major 3x3 dynamics matrix A
    b_vec:  torch.Tensor,   # (N, 3)   constant forcing / drift vector b
    x0:     torch.Tensor,   # (N, 3)   initial 3D displacement x(0) = x0
    tau:    float,
) -> torch.Tensor:
    """
    Solve the real-valued 3D affine ODE:
        dx/dtau = A*x + b,   x(0) = x0,   A in R^{3x3}, b in R^3

    Solved via the augmented 4x4 system:
        d/dtau [x; 1] = [[A, b], [0, 0]] @ [x; 1]
    => [x(tau); 1] = expm([[A, b], [0, 0]] * tau) @ [x0; 1]

    This avoids inverting A and correctly handles the A=0 (linear drift) case
    as a special case of the general exponential solution.

    Args:
        A_flat: (N, 9) row-major 3x3 dynamics matrix.
        b_vec:  (N, 3) constant drift term.
        x0:     (N, 3) initial 3D displacement at tau=0.
        tau:    scalar time in [-1, 1].

    Returns:
        x_tau: (N, 3) 3D displacement at time tau.
    """
    N      = x0.shape[0]
    device = x0.device
    dtype  = x0.dtype

    A = A_flat.reshape(N, 3, 3)

    # Build augmented 4x4 system: [[A, b], [0, 0, 0, 0]]
    aug = torch.zeros(N, 4, 4, device=device, dtype=dtype)
    aug[:, :3, :3] = A
    aug[:, :3,  3] = b_vec

    # Matrix exponential of the augmented system scaled by tau
    exp_aug = torch.matrix_exp(aug * tau)   # (N, 4, 4)

    # Augmented state vector [x0; 1]
    state = torch.cat([x0, torch.ones(N, 1, device=device, dtype=dtype)], dim=-1)  # (N, 4)

    # x(tau) is the first three components of exp_aug @ [x0; 1]
    x_tau = torch.bmm(exp_aug, state.unsqueeze(-1)).squeeze(-1)[:, :3]             # (N, 3)
    return x_tau


# ---------------------------------------------------------------------------
# Covariance ODE solver: scale + SO(3) orientation
# ---------------------------------------------------------------------------

def evolve_covariance(
    scaling_log_canonical: torch.Tensor,   # (N, 3)  canonical log-scales
    rotation_q_canonical:  torch.Tensor,   # (N, 4)  canonical quaternion [w,x,y,z]
    kappa:                 torch.Tensor,   # (N, 3)  log-scale rate
    omega_cov:             torch.Tensor,   # (N, 3)  angular velocity in R^3
    tau:                   float,
) -> tuple:
    """
    Compute deformed log-scales and quaternion at normalized time tau.

    Scale ODE (explicit closed form):
        ds_j/dtau = kappa_j  =>  s_j(tau) = s_j^0 + kappa_j * tau

    Orientation ODE in SO(3) (Rodrigues formula):
        dR/dtau = omega_hat * R,   omega in R^3
        => R(tau) = exp(omega_hat * tau) * R^0

    The rotation delta exp(omega_hat * tau) is computed as a quaternion:
        theta = ||omega|| * tau
        q_delta = [cos(theta/2), sin(theta/2) * omega/||omega||]

    Args:
        scaling_log_canonical: (N, 3) log-scale parameters at tau=0.
        rotation_q_canonical:  (N, 4) quaternion [w,x,y,z] at tau=0.
        kappa:     (N, 3) per-Gaussian log-scale rate parameters.
        omega_cov: (N, 3) per-Gaussian angular velocity (SO(3) generator).
        tau:       scalar time in [-1, 1].

    Returns:
        scaling_log_t: (N, 3) log-scales at time tau.
        rotation_q_t:  (N, 4) unit quaternion at time tau.
    """
    # Linear scale evolution: s(tau) = s^0 + kappa * tau
    scaling_log_t = scaling_log_canonical + kappa * tau

    # SO(3) rotation delta via Rodrigues formula
    # theta = ||omega|| * tau  (signed rotation angle)
    omega_norm = torch.norm(omega_cov, dim=-1, keepdim=True).clamp(min=1e-8)  # (N, 1)
    axis       = omega_cov / omega_norm                                         # (N, 3) unit axis
    angle      = omega_norm.squeeze(-1) * tau                                   # (N,)

    # Convert axis-angle to quaternion: q = [cos(a/2), sin(a/2)*axis]
    c       = torch.cos(angle / 2).unsqueeze(-1)  # (N, 1)
    s       = torch.sin(angle / 2).unsqueeze(-1)  # (N, 1)
    q_delta = torch.cat([c, s * axis], dim=-1)    # (N, 4)

    # Compose: q(tau) = q_delta * q^0  (delta applied first in world frame)
    q_total      = quaternion_multiply(q_delta, rotation_q_canonical)
    rotation_q_t = F.normalize(q_total, dim=-1)

    return scaling_log_t, rotation_q_t

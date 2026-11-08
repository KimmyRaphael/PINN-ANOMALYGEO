import math
import torch
import torch.nn as nn


def gradient(y, x):
    """Compute first-order spatial derivative dy/dx via Automatic Differentiation."""
    return torch.autograd.grad(
        y, x, grad_outputs=torch.ones_like(y), create_graph=True, retain_graph=True
    )[0]


def laplacian(y, x):
    """Compute Laplacian (d^2y/dx^2 + d^2y/dy^2 + d^2y/dz^2) and first gradient."""
    grad_y = gradient(y, x)
    lap = torch.zeros_like(y)
    for i in range(x.shape[1]):
        grad2 = torch.autograd.grad(
            grad_y[:, i:i+1],
            x,
            grad_outputs=torch.ones_like(grad_y[:, i:i+1]),
            create_graph=True,
            retain_graph=True,
        )[0][:, i:i+1]
        lap = lap + grad2
    return lap, grad_y


class PotentialFieldPhysicsLoss(nn.Module):
    """
    Composite Physics-Informed Loss Function for Bouguer & Magnetic Joint Inversion:
        L_total = w_data * L_data + w_pde * L_pde + w_joint * L_joint + w_bc * L_bc
    """
    def __init__(self, inclination_deg=-30.0, declination_deg=1.5, G_scaled=1.0):
        super().__init__()
        inc = math.radians(inclination_deg)
        dec = math.radians(declination_deg)
        # Unit vector of ambient geomagnetic field B_0 (x: East, y: North, z: Down)
        bx = math.cos(inc) * math.sin(dec)
        by = math.cos(inc) * math.cos(dec)
        bz = math.sin(inc)
        self.register_buffer("b0_hat", torch.tensor([[bx, by, bz]], dtype=torch.float32))
        self.G_scaled = G_scaled
        self.mse = nn.MSELoss()

    def predict_surface_anomalies(self, model, coords_surf):
        """Predict Bouguer anomaly (g_z) and Total Magnetic Intensity (Delta T) at z=0."""
        coords_surf = coords_surf.clone().detach().requires_grad_(True)
        U, V_m, _, _ = model(coords_surf)
        grad_U = gradient(U, coords_surf)
        grad_Vm = gradient(V_m, coords_surf)

        # g_z = -dU/dz
        gz_pred = -grad_U[:, 2:3]
        # Delta T = -grad(V_m) . b0_hat
        dT_pred = -torch.sum(grad_Vm * self.b0_hat, dim=-1, keepdim=True)
        return gz_pred, dT_pred

    def pde_and_joint_residuals(self, model, coords_col):
        """
        Evaluate Poisson PDEs and Cross-Gradient geological constraint at collocation points:
            1) laplacian(U) - 4 * pi * G * delta_rho = 0
            2) laplacian(V_m) - div(M(kappa)) = 0
            3) || grad(delta_rho) x grad(kappa) ||^2 = 0
        """
        coords_col = coords_col.clone().detach().requires_grad_(True)
        U, V_m, delta_rho, kappa = model(coords_col)

        lap_U, _ = laplacian(U, coords_col)
        lap_Vm, _ = laplacian(V_m, coords_col)

        grad_rho = gradient(delta_rho, coords_col)
        grad_kappa = gradient(kappa, coords_col)

        # 1. Gravity Poisson residual
        res_grav = lap_U - 4.0 * math.pi * self.G_scaled * delta_rho

        # 2. Magnetic Poisson residual: div(M) = grad(kappa) . b0_hat (under uniform B0 direction)
        div_M = torch.sum(grad_kappa * self.b0_hat, dim=-1, keepdim=True)
        res_mag = lap_Vm - div_M

        # 3. Structural similarity via Cross-Gradient: grad(delta_rho) x grad(kappa)
        cross_grad = torch.linalg.cross(grad_rho, grad_kappa, dim=-1)
        loss_joint = torch.mean(torch.sum(cross_grad ** 2, dim=-1))

        loss_pde_grav = torch.mean(res_grav ** 2)
        loss_pde_mag = torch.mean(res_mag ** 2)

        return loss_pde_grav, loss_pde_mag, loss_joint

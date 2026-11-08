import argparse
import os
import torch
from src.model import JointPINN
from src.physics_loss import PotentialFieldPhysicsLoss
from src.synthetic_data import generate_buried_sphere_anomaly, sample_collocation_points


def main():
    parser = argparse.ArgumentParser(description="Train PINN for Joint Bouguer & Magnetic Inversion")
    parser.add_argument("--epochs", type=int, default=1000, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--n_col", type=int, default=2000, help="Number of 3D collocation points")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Running PINN training on device: {device}")

    # 1. Prepare surface observation data
    coords_surf_np, gz_obs_np, dT_obs_np = generate_buried_sphere_anomaly(n_grid=25)
    coords_surf = torch.tensor(coords_surf_np, device=device)
    gz_obs = torch.tensor(gz_obs_np, device=device)
    dT_obs = torch.tensor(dT_obs_np, device=device)

    # 2. Initialize Model & Physics Loss
    model = JointPINN(hidden_layers=5, hidden_units=64, use_fourier=True).to(device)
    criterion = PotentialFieldPhysicsLoss(inclination_deg=-30.0, declination_deg=1.5).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # 3. Training Loop
    for epoch in range(1, args.epochs + 1):
        optimizer.zero_grad()

        # Surface Data Loss
        gz_pred, dT_pred = criterion.predict_surface_anomalies(model, coords_surf)
        loss_data_g = torch.mean((gz_pred - gz_obs) ** 2)
        loss_data_m = torch.mean((dT_pred - dT_obs) ** 2)
        loss_data = loss_data_g + loss_data_m

        # Subsurface Collocation PDE & Joint Cross-Gradient Loss
        coords_col = torch.tensor(sample_collocation_points(args.n_col), device=device)
        loss_pde_g, loss_pde_m, loss_joint = criterion.pde_and_joint_residuals(model, coords_col)

        # Total Weighted Loss
        loss_total = 10.0 * loss_data + 1.0 * (loss_pde_g + loss_pde_m) + 5.0 * loss_joint
        loss_total.backward()
        optimizer.step()

        if epoch % 100 == 0 or epoch == 1:
            print(
                f"Epoch [{epoch:04d}/{args.epochs:04d}] | "
                f"Total: {loss_total.item():.5f} | "
                f"Data: {loss_data.item():.5f} | "
                f"PDE(G/M): {(loss_pde_g + loss_pde_m).item():.5f} | "
                f"CrossGrad: {loss_joint.item():.5f}"
            )

    os.makedirs("results/checkpoints", exist_ok=True)
    ckpt_path = "results/checkpoints/pinn_joint_model.pt"
    torch.save(model.state_dict(), ckpt_path)
    print(f"[INFO] Model checkpoint saved to {ckpt_path}")


if __name__ == "__main__":
    main()

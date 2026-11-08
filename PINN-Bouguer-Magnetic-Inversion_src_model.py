import math
import torch
import torch.nn as nn


class SineLayer(nn.Module):
    """SIREN Sine activation layer with principled weight initialization."""
    def __init__(self, in_features, out_features, bias=True, is_first=False, omega_0=30.0):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 1 / self.in_features)
            else:
                self.linear.weight.uniform_(
                    -math.sqrt(6 / self.in_features) / self.omega_0,
                    math.sqrt(6 / self.in_features) / self.omega_0,
                )

    def forward(self, x):
        return torch.sin(self.omega_0 * self.linear(x))


class FourierFeatureEmbedding(nn.Module):
    """Random Fourier Features to mitigate spectral bias in potential field PINNs."""
    def __init__(self, in_features=3, num_frequencies=32, sigma=1.0):
        super().__init__()
        self.register_buffer("B", torch.randn(in_features, num_frequencies) * sigma)

    def forward(self, x):
        proj = 2.0 * math.pi * torch.matmul(x, self.B)
        return torch.cat([x, torch.sin(proj), torch.cos(proj)], dim=-1)


class JointPINN(nn.Module):
    """
    Physics-Informed Neural Network for Joint Gravity (Bouguer) and Magnetic Inversion.
    Inputs:
        coords: (N, 3) tensor of spatial coordinates (x, y, z) requiring gradients.
    Outputs:
        U: Gravitational potential
        V_m: Magnetic scalar potential
        delta_rho: Subsurface density contrast (kg/m^3 or g/cm^3 scaled)
        kappa: Subsurface magnetic susceptibility (SI)
    """
    def __init__(self, hidden_layers=6, hidden_units=128, use_fourier=True, num_frequencies=32, sigma=1.0):
        super().__init__()
        self.use_fourier = use_fourier
        in_dim = 3
        if use_fourier:
            self.fourier = FourierFeatureEmbedding(3, num_frequencies, sigma)
            in_dim = 3 + 2 * num_frequencies

        layers = [SineLayer(in_dim, hidden_units, is_first=True, omega_0=30.0)]
        for _ in range(hidden_layers - 1):
            layers.append(SineLayer(hidden_units, hidden_units, is_first=False, omega_0=30.0))
        self.backbone = nn.Sequential(*layers)

        # Output heads
        self.head_potentials = nn.Linear(hidden_units, 2)   # [U, V_m]
        self.head_properties = nn.Linear(hidden_units, 2)   # [delta_rho, kappa]

    def forward(self, coords):
        x = self.fourier(coords) if self.use_fourier else coords
        feat = self.backbone(x)
        potentials = self.head_potentials(feat)
        raw_props = self.head_properties(feat)

        U = potentials[:, 0:1]
        V_m = potentials[:, 1:2]
        # Bound physical properties smoothly
        delta_rho = torch.tanh(raw_props[:, 0:1])
        kappa = torch.sigmoid(raw_props[:, 1:2])
        return U, V_m, delta_rho, kappa

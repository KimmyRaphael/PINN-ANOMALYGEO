import numpy as np


def generate_buried_sphere_anomaly(n_grid=40, noise_level=0.02, seed=42):
    """
    Generate synthetic 2D surface Bouguer gravity (g_z) and TMI magnetic (Delta T)
    anomalies over a buried anomalous sphere in normalized coordinates [-1, 1].
    """
    rng = np.random.default_rng(seed)
    x = np.linspace(-1.0, 1.0, n_grid)
    y = np.linspace(-1.0, 1.0, n_grid)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)

    # Buried sphere center (x0, y0, z0) and radius R
    x0, y0, z0 = 0.0, 0.0, 0.45
    R = 0.25
    delta_rho_0 = 1.0
    kappa_0 = 0.8

    dx = X - x0
    dy = Y - y0
    dz = Z - z0
    r = np.sqrt(dx**2 + dy**2 + dz**2) + 1e-6

    # Analytical vertical gravity anomaly (scaled)
    mass_factor = (4.0 / 3.0) * np.pi * (R**3) * delta_rho_0
    gz_clean = mass_factor * (-dz) / (r**3)

    # Analytical dipole TMI anomaly for inclination -30 deg, declination 1.5 deg
    inc, dec = np.radians(-30.0), np.radians(1.5)
    bx, by, bz = np.cos(inc) * np.sin(dec), np.cos(inc) * np.cos(dec), np.sin(inc)
    r_dot_b = dx * bx + dy * by + dz * bz
    dipole_factor = (4.0 / 3.0) * np.pi * (R**3) * kappa_0
    dT_clean = dipole_factor * (3.0 * (r_dot_b**2) / (r**5) - 1.0 / (r**3))

    gz_obs = gz_clean + noise_level * np.std(gz_clean) * rng.normal(size=gz_clean.shape)
    dT_obs = dT_clean + noise_level * np.std(dT_clean) * rng.normal(size=dT_clean.shape)

    coords_surf = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=-1).astype(np.float32)
    gz_obs = gz_obs.ravel()[:, None].astype(np.float32)
    dT_obs = dT_obs.ravel()[:, None].astype(np.float32)

    return coords_surf, gz_obs, dT_obs


def sample_collocation_points(n_points=4000, seed=None):
    """Sample 3D subsurface collocation points (x, y in [-1, 1], z in [0, 1])."""
    rng = np.random.default_rng(seed)
    xy = rng.uniform(-1.0, 1.0, size=(n_points, 2))
    z = rng.uniform(0.0, 1.0, size=(n_points, 1))
    return np.hstack([xy, z]).astype(np.float32)

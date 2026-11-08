# PINN-Bouguer-Magnetic-Inversion

Repositori penelitian skripsi untuk **Prediksi dan Inversi Bersama (*Joint Inversion*) Anomali Bouguer (Gravitasi) & Geomagnetik Menggunakan *Physics-Informed Neural Networks* (PINN)**.

---

## Deskripsi Proyek

Masalah inversi medan potensial geofisika (gravitasi dan magnetik) secara alami bersifat *ill-posed* dan tidak unik (*non-unique*). Repositori ini mengimplementasikan arsitektur **Physics-Informed Neural Network (PINN)** berbasis koordinat kontinu (*mesh-free*) yang menggabungkan:
1. **Persamaan Poisson / Laplace Gravitasi & Magnetik** melalui *Automatic Differentiation* (AD).
2. **Inversi Bersama (*Joint Inversion*)** dengan kendala kemiripan struktur geologi **Cross-Gradient** antara kontras densitas ($\Delta\rho$) dan suseptibilitas magnetik ($\kappa$).
3. **Fourier Feature Embedding & Aktivasi SIREN** untuk mengatasi *spectral bias* pada gradien anomali frekuensi tinggi.

---

## Formulasi Fisika & Fungsi Rugi (*Loss Function*)

### 1. Persamaan Pengatur (*Governing Equations*)
* **Potensial & Anomali Bouguer Gravitasi:**
  $$\nabla^2 U(\mathbf{x}) = 4\pi G \Delta\rho(\mathbf{x}), \quad g_z = -\frac{\partial U}{\partial z}$$
* **Potensial Skalar & Anomali Medan Total Magnetik (TMI):**
  $$\nabla^2 V_m(\mathbf{x}) = \nabla \cdot \mathbf{M}(\kappa), \quad \Delta T \approx -\nabla V_m \cdot \hat{\mathbf{B}}_0$$

### 2. Fungsi Rugi Total (*Total Loss*)
$$\mathcal{L}_{total} = w_{data}\mathcal{L}_{data} + w_{pde}\mathcal{L}_{PDE} + w_{joint}\mathcal{L}_{joint} + w_{bc}\mathcal{L}_{BC}$$

Di mana $\mathcal{L}_{joint}$ menggunakan *Cross-Gradient Constraint* (Gallardo & Meju):
$$\mathcal{L}_{joint} = \frac{1}{N_{col}} \sum_{j=1}^{N_{col}} \left\| \nabla \Delta\rho(\mathbf{x}_j) \times \nabla \kappa(\mathbf{x}_j) \right\|^2$$

---

## Struktur Direktori

```text
PINN-Bouguer-Magnetic-Inversion/
├── configs/
│   └── default_config.yaml      # Konfigurasi hyperparameter, geometri domain, & bobot loss
├── data/
│   ├── synthetic/               # Data uji sintetis (model bola / prisma terkubur)
│   └── field/                   # Data observasi lapangan (Bouguer & TMI/RTP)
├── notebooks/                   # Jupyter Notebooks untuk eksplorasi & visualisasi 2D/3D
├── results/
│   ├── checkpoints/             # Bobot model PyTorch (.pt) hasil training
│   └── figures/                 # Plot penampang bawah permukaan & kurva konvergensi loss
├── src/
│   ├── __init__.py
│   ├── model.py                 # Arsitektur JointPINN (SIREN + Fourier Feature Embedding)
│   ├── physics_loss.py          # Operator Automatic Differentiation (Laplacian, Cross-Gradient)
│   └── synthetic_data.py        # Generator anomali sintetis 3D & sampling titik kolokasi
├── .gitignore
├── requirements.txt
├── README.md
└── train.py                     # Skrip utama training dan evaluasi PINN
```

---

## Instalasi & Cara Menjalankan

1. **Clone repositori dan buat *virtual environment*:**
   ```bash
   git clone https://github.com/username-anda/PINN-Bouguer-Magnetic-Inversion.git
   cd PINN-Bouguer-Magnetic-Inversion
   python -m venv venv
   source venv/bin/activate  # Di Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Jalankan *training* pada model sintetis:**
   ```bash
   python train.py --epochs 2000 --lr 1e-3 --n_col 4000
   ```

---

## Lisensi & Sitasi
Proyek ini dikembangkan untuk keperluan Tugas Akhir / Skripsi Geofisika.

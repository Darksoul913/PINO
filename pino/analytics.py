"""
Pure PyTorch/NumPy Scientific Analytics & Field Synthesis Helpers for PINO.
Provides Biot-Savart velocity streamlines, conservation invariants, radial energy/enstrophy
spectra, 2D log-FFT spectra, custom periodic vortex synthesis, bicubic interpolation, and
standardized operator input construction.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn.functional as F


def _to_2d_numpy(field: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    """Converts a 2D (or singleton-batched) array/tensor into a float64 2D numpy array."""
    if isinstance(field, torch.Tensor):
        arr = field.detach().cpu().numpy()
    else:
        arr = np.asarray(field)
    while arr.ndim > 2 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D vorticity field, got shape {arr.shape}")
    return arr.astype(np.float64, copy=False)


def _to_2d_tensor(field: torch.Tensor, device: Optional[Union[str, torch.device]] = None) -> torch.Tensor:
    """Converts a 2D (or singleton-batched) tensor to a float32 2D tensor on target device."""
    t = field.detach()
    while t.ndim > 2 and t.shape[0] == 1:
        t = t[0]
    if t.ndim != 2:
        raise ValueError(f"Expected 2D vorticity tensor, got shape {tuple(field.shape)}")
    target_device = field.device if device is None else torch.device(device)
    return t.to(device=target_device, dtype=torch.float32)


def compute_streamlines(
    w_tensor: torch.Tensor,
    s_x: int,
    s_y: int,
    device: Optional[Union[str, torch.device]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes Biot-Savart velocity vectors (ux, uy) from 2D vorticity on [0, 2pi)^2.

    Solves -nabla^2 psi = omega in Fourier space (psi_hat = w_hat / |k|^2 for k != 0)
    and recovers u = (dpsi/dy, -dpsi/dx).
    """
    dev = w_tensor.device if device is None else torch.device(device)
    w_2d = _to_2d_tensor(w_tensor, device=dev)

    kx = torch.fft.fftfreq(s_x, d=1.0 / s_x).to(dev)
    ky = torch.fft.fftfreq(s_y, d=1.0 / s_y).to(dev)
    Kx, Ky = torch.meshgrid(kx, ky, indexing="ij")
    K2 = Kx ** 2 + Ky ** 2

    w_hat = torch.fft.fft2(w_2d)
    inv_k2 = torch.zeros_like(K2)
    nonzero = K2 > 1e-10
    inv_k2[nonzero] = 1.0 / K2[nonzero]

    psi_hat = w_hat * inv_k2
    ux_hat = 1j * Ky * psi_hat
    uy_hat = -1j * Kx * psi_hat

    ux = torch.fft.ifft2(ux_hat).real.detach().cpu().numpy()
    uy = torch.fft.ifft2(uy_hat).real.detach().cpu().numpy()
    return ux, uy


def compute_conservation_invariants(w_np: Union[np.ndarray, torch.Tensor]) -> Dict[str, float]:
    """
    Computes physical 2D incompressible flow invariants on [0, 2pi)^2:
      - circulation: spatial mean vorticity (proportional to net circulation)
      - enstrophy: 0.5 * mean(w^2)
      - kinetic_energy: 0.5 * mean(ux^2 + uy^2) from exact Biot-Savart velocities
      - peak_vorticity: max(|w|)
    """
    w_arr = _to_2d_numpy(w_np)
    s_x, s_y = w_arr.shape

    circulation = float(np.mean(w_arr))
    enstrophy = float(0.5 * np.mean(w_arr ** 2))
    peak_vorticity = float(np.max(np.abs(w_arr)))

    kx = np.fft.fftfreq(s_x) * s_x
    ky = np.fft.fftfreq(s_y) * s_y
    Kx, Ky = np.meshgrid(kx, ky, indexing="ij")
    K2 = Kx ** 2 + Ky ** 2

    w_hat = np.fft.fft2(w_arr)
    inv_k2 = np.zeros_like(K2, dtype=np.float64)
    nonzero = K2 > 1e-10
    inv_k2[nonzero] = 1.0 / K2[nonzero]

    psi_hat = w_hat * inv_k2
    ux = np.fft.ifft2(1j * Ky * psi_hat).real
    uy = np.fft.ifft2(-1j * Kx * psi_hat).real
    kinetic_energy = float(0.5 * np.mean(ux ** 2 + uy ** 2))

    return {
        "circulation": circulation,
        "enstrophy": enstrophy,
        "kinetic_energy": kinetic_energy,
        "peak_vorticity": peak_vorticity,
    }


def compute_radial_energy_spectrum(w_np: Union[np.ndarray, torch.Tensor]) -> Dict[str, Any]:
    """
    Computes isotropic 1D radial energy spectrum E(k) and enstrophy spectrum Z(k)
    by binning normalized 2D spectral densities into integer shells k in 1..min(s_x, s_y)//2.
    """
    w_arr = _to_2d_numpy(w_np)
    s_x, s_y = w_arr.shape
    n_pts = float(s_x * s_y)

    w_hat = np.fft.fft2(w_arr) / n_pts
    kx = np.fft.fftfreq(s_x) * s_x
    ky = np.fft.fftfreq(s_y) * s_y
    Kx, Ky = np.meshgrid(kx, ky, indexing="ij")
    K2 = Kx ** 2 + Ky ** 2
    k_mag = np.sqrt(K2)

    mag2 = np.abs(w_hat) ** 2
    z_2d = 0.5 * mag2
    e_2d = 0.5 * mag2 / np.maximum(K2, 1.0)
    e_2d[0, 0] = 0.0

    max_shell = min(s_x, s_y) // 2
    shells = np.rint(k_mag).astype(int).ravel()
    e_binned = np.bincount(shells, weights=e_2d.ravel(), minlength=max_shell + 1)
    z_binned = np.bincount(shells, weights=z_2d.ravel(), minlength=max_shell + 1)

    k_list = list(range(1, max_shell + 1))
    energy = [max(float(e_binned[k_val]), 1e-12) for k_val in k_list]
    enstrophy = [max(float(z_binned[k_val]), 1e-12) for k_val in k_list]

    return {
        "k": k_list,
        "energy": energy,
        "enstrophy": enstrophy,
    }


def compute_fft2_log_magnitude(
    w_np: Union[np.ndarray, torch.Tensor],
    out_size: int = 64,
) -> List[List[float]]:
    """
    Computes fftshift(log10(|fft2(w)| / w.size + 1e-6)), center-cropped to out_size x out_size
    when the input grid exceeds out_size along an axis.
    """
    w_arr = _to_2d_numpy(w_np)
    mag = np.log10(np.abs(np.fft.fft2(w_arr)) / float(w_arr.size) + 1e-6)
    shifted = np.fft.fftshift(mag)

    s_x, s_y = shifted.shape
    if s_x > out_size:
        start_x = (s_x // 2) - (out_size // 2)
        shifted = shifted[start_x : start_x + out_size, :]
    if s_y > out_size:
        start_y = (s_y // 2) - (out_size // 2)
        shifted = shifted[:, start_y : start_y + out_size]

    return shifted.tolist()


def generate_custom_vortex_field(
    vortices: Optional[List[Dict[str, float]]],
    s_x: int = 64,
    s_y: int = 64,
    device: Union[str, torch.device] = "cpu",
) -> torch.Tensor:
    """
    Synthesizes a periodic 2D Gaussian vortex field on [0, 1)^2 from user-placed vortices
    {x, y, strength, sigma}, standardized to zero mean and unit std (when std > 1e-6).
    Falls back to a clean counter-rotating dipole when vortices is empty.
    """
    dev = torch.device(device)
    active_vortices = vortices if vortices else [
        {"x": 0.35, "y": 0.45, "strength": 1.0, "sigma": 0.08},
        {"x": 0.65, "y": 0.55, "strength": -1.0, "sigma": 0.08},
    ]

    x_coords = torch.linspace(0.0, (s_x - 1) / s_x, s_x, device=dev, dtype=torch.float32)
    y_coords = torch.linspace(0.0, (s_y - 1) / s_y, s_y, device=dev, dtype=torch.float32)
    X, Y = torch.meshgrid(x_coords, y_coords, indexing="ij")

    field = torch.zeros((s_x, s_y), device=dev, dtype=torch.float32)
    for vortex in active_vortices:
        vx = float(vortex.get("x", 0.5))
        vy = float(vortex.get("y", 0.5))
        strength = float(vortex.get("strength", 1.0))
        sigma = max(float(vortex.get("sigma", 0.08)), 1e-3)

        dx = ((X - vx + 0.5) % 1.0) - 0.5
        dy = ((Y - vy + 0.5) % 1.0) - 0.5
        field = field + strength * torch.exp(-(dx ** 2 + dy ** 2) / (2.0 * (sigma ** 2)))

    mean = field.mean()
    std = field.std()
    if std > 1e-6:
        field = (field - mean) / std
    else:
        field = field - mean

    return field.unsqueeze(0)


def compute_bicubic_upsample(
    u_64: Union[torch.Tensor, np.ndarray],
    target_res: int = 256,
) -> np.ndarray:
    """Upsamples a 2D vorticity field to (target_res, target_res) using bicubic interpolation."""
    if isinstance(u_64, torch.Tensor):
        t = u_64.detach().float()
    else:
        t = torch.as_tensor(u_64, dtype=torch.float32)

    if t.ndim == 2:
        t = t.unsqueeze(0).unsqueeze(0)
    elif t.ndim == 3:
        t = t.unsqueeze(0)
    elif t.ndim != 4:
        raise ValueError(f"Expected tensor of ndim 2, 3, or 4, got shape {tuple(t.shape)}")

    upsampled = F.interpolate(
        t,
        size=(target_res, target_res),
        mode="bicubic",
        align_corners=False,
    )
    return upsampled[0, 0].detach().cpu().numpy()


def build_normalized_input(
    w0: torch.Tensor,
    device: Optional[Union[str, torch.device]] = None,
) -> torch.Tensor:
    """
    Constructs PINO2D operator input tensor of shape (B, 3, s_x, s_y)
    by prepending normalized spatial grid coordinates [0, (s-1)/s] to w0.
    """
    if w0.ndim == 2:
        w0_4d = w0.unsqueeze(0).unsqueeze(0)
    elif w0.ndim == 3:
        w0_4d = w0.unsqueeze(1)
    elif w0.ndim == 4:
        w0_4d = w0
    else:
        raise ValueError(f"Expected w0 of ndim 2, 3, or 4, got shape {tuple(w0.shape)}")

    dev = w0.device if device is None else torch.device(device)
    w0_4d = w0_4d.to(device=dev, dtype=torch.float32)
    batch_size, _, s_x, s_y = w0_4d.shape

    x_coords = torch.linspace(0.0, (s_x - 1) / s_x, s_x, device=dev, dtype=torch.float32)
    y_coords = torch.linspace(0.0, (s_y - 1) / s_y, s_y, device=dev, dtype=torch.float32)
    grid_x, grid_y = torch.meshgrid(x_coords, y_coords, indexing="ij")
    grid = torch.stack([grid_x, grid_y], dim=0).unsqueeze(0).expand(batch_size, -1, -1, -1)

    return torch.cat([grid, w0_4d], dim=1)

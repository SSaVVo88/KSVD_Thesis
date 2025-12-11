import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

from src.patches import extract_patches, reconstruct_from_patches
from src.ksvd import ksvd_denoise
from src.utils import psnr, get_dataset_path, ssim_score, rmse_score

# Sciezki do czystego i zaszumionego folderu
clean_dir = get_dataset_path("original_png")
noisy_dir = get_dataset_path("noisy25")

# Poki co jedno zdjecie do testu
filename = "0037.png"    

# Sciezki do plikow
clean_path = clean_dir / filename
noisy_path = noisy_dir / filename

# Wczytanie obrazów jako skale szarosci
clean = cv2.imread(str(clean_path), cv2.IMREAD_GRAYSCALE)
noisy = cv2.imread(str(noisy_path), cv2.IMREAD_GRAYSCALE)

# Odaplanie KSVD na zaszumionym obrazie
denoised = ksvd_denoise(noisy, patch_size=8, num_atoms=128, sparsity=4, iterations=5)

# Obliczenie metryki PSNR między oryginałem a denoised
score_psnr = psnr(clean, denoised)
print("PSNR =", score_psnr)
score_ssim = ssim_score(clean, denoised)
print("SSIM =", score_ssim)
score_rmse = rmse_score(clean, denoised)
print("RMSE =", score_rmse)

# Wyświetlenie wyników
plt.figure(figsize=(12,4))
plt.subplot(131); plt.imshow(clean, cmap='gray'); plt.title("Clean")
plt.subplot(132); plt.imshow(noisy, cmap='gray'); plt.title("Noisy 25")
plt.subplot(133); plt.imshow(denoised, cmap='gray'); plt.title(f"Denoised (PSNR={score_psnr:.2f}, SSIM={score_ssim:.2f} RMSE={score_rmse:.4f})")
plt.show()

#Plik na przyszle funkcje pomocnicze
import numpy as np
from pathlib import Path
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import normalized_root_mse

# Funkcja do obliczania PSNR() jako metryki jakosci obrazu
def psnr(original, denoised):
    mse = np.mean((original.astype(np.float32) - denoised.astype(np.float32))**2)
    return 10*np.log10(255*255/mse)

""" Duzo zmian i potrzeba zastanowienia sie nad struktura projektu.
    Na razie wrzucam tutaj funkcje do sciezek i metryki"""
def ssim_score(original, denoised):
    return ssim(original, denoised)

def rmse_score(original, denoised):
    return normalized_root_mse(original, denoised)

#Jako ze w projekcie bedzie duzo skakania miedzy folderami i sciezkami,
#warto miec kilka funkcji pomocniczych do sciezek
def get_project_root():
    #Zwraca sciezke do katalogu glownego projektu
    return Path(__file__).resolve().parent.parent

def get_dataset_path(subfolder=None):
    #Zwraca sciezke do folderu z danymi (domyslnie CBSD68).
    base = get_project_root() / "CBSD68"

    if subfolder is not None:
        return base / subfolder
    return base

def get_results_path():
    #Zwraca sciezke do folderu z wynikami
    results_dir = get_project_root() / "results"
    results_dir.mkdir(exist_ok=True)
    return results_dir

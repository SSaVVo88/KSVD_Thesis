import numpy as np
import cv2
from .patches import extract_patches, reconstruct_from_patches
from .omp import omp

# Funkcja do aktualizacji slownika k-SVD dla kazdego slownika atomu
## Oblicza macierz bledu bez wplywu danego atomu
## Wykonuje SVD na macierzy bledu, aby zaktualizowac atom slownika i odpowiadajace wspolczynniki

def ksvd_update(dictionary, data, coefficients, max_iter=10):
    k = dictionary.shape[1]
    
    for i in range(k):
        # Znajdz indeksy sygnalow uzywajacych tego atomu
        idx = np.where(coefficients[i, :] != 0)[0]
        if len(idx) == 0:
            continue
            
        # Blad bez wplywu tego atomu
        error = data[:, idx] - dictionary @ coefficients[:, idx]
        error += dictionary[:, i:i+1] * coefficients[i:i+1, idx]
        
        # SVD bledu i aktualizacja
        U, S, Vt = np.linalg.svd(error, full_matrices=False)
        dictionary[:, i] = U[:, 0]
        coefficients[i, idx] = S[0] * Vt[0, :]
        
    return dictionary, coefficients

#Funkcja glowna do denoisingu KSVD

#Wyciaga Patche z obrazu
#Uczy slownik SVD
#Wykonuje sparse coding z OMP
#Rekonstruuje obraz z patchy
#Zwraca denoised obraz

def ksvd_denoise(image, patch_size=8, num_atoms=256, sparsity=5, iterations=10):

    # Zamiana na skale szarosci jesli obraz kolorowy
    if len(image.shape) > 2:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()
    
    # Normalizacja obrazu do [0,1]
    gray = gray.astype(np.float32) / 255.0
    
    # Wyciaganie patchy
    patches = extract_patches(gray, patch_size)
    
    # Inicjalizacja slownika (losowo lub za pomoca PCA)
    dictionary = np.random.randn(patch_size**2, num_atoms)
    dictionary = dictionary / np.linalg.norm(dictionary, axis=0)
    
   
    for _ in range(iterations):
        # Sparse coding
        coefficients = np.zeros((num_atoms, patches.shape[1]))
        for j in range(patches.shape[1]):
            coefficients[:, j] = omp(dictionary, patches[:, j], sparsity)
        
        # Aktualizacja slownika
        dictionary, coefficients = ksvd_update(dictionary, patches, coefficients)
    
    # Odtwarzenie obrazu z patchy
    denoised_patches = dictionary @ coefficients
    denoised = reconstruct_from_patches(denoised_patches.T, gray.shape, patch_size)
    
    return np.clip(denoised * 255, 0, 255).astype(np.uint8)


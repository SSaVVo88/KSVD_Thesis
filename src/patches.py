import numpy as np

# Funkcje do ekstrakcji patchy, na wejscie obraz, na wyjscie macierz patchy
# Macierz I o wymiarach (patch_size*patch_size, num_patches)
# ekstrakcja nachodzacych na siebie patchy o wymiarach patch_size x patch_size
# Zwraca macierz patchy jako kolumny wektorow w macierzy 
# Celem operacji jest zbudowanie treningoweg zbioru patchy z obrazu do trenowania slownika
def extract_patches(image, patch_size=8, step=4):
    patches = []
    for i in range(0, image.shape[0] - patch_size + 1, step):
        for j in range(0, image.shape[1] - patch_size + 1, step):
            patch = image[i:i+patch_size, j:j+patch_size]
            patches.append(patch.flatten())
    return np.array(patches).T

# Funkcja do rekonstrukcji obrazu z patchy, na wejscie zaszumione patche i kształt obrazu
# na wyjscie zrekonstruowany obraz
# "uklada" patche z powrotem do obrazu, uśredniając nachodzące na siebie obszary

def reconstruct_from_patches(patches, image_shape, patch_size=8, step=4):
    """Reconstruct image from patches"""
    reconstructed = np.zeros(image_shape)
    count = np.zeros(image_shape)
    
    idx = 0
    for i in range(0, image_shape[0] - patch_size + 1, step):
        for j in range(0, image_shape[1] - patch_size + 1, step):
            reconstructed[i:i+patch_size, j:j+patch_size] += patches[idx].reshape(patch_size, patch_size)
            count[i:i+patch_size, j:j+patch_size] += 1
            idx += 1
    
    return reconstructed / np.maximum(count, 1)
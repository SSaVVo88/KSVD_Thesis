import numpy as np

# Funkcja OMP - Orthogonal Matching Pursuit do sparse codingu 
# Oblicza rzadki wektor wspolczynnikow dla danego sygnalu
# przy uzyciu podanego slownika i zadanej sparsity (liczby niezerowych wspolczynnikow)
# Jest to algorytm "chciwy", ktory iteracyjnie wybiera atomy slownika
# najlepiej dopasowujace sie do pozostalego resztkowego sygnalu

def omp(dictionary, signal, sparsity):
    residual = signal.copy()
    indices = []
    coefficients = np.zeros(dictionary.shape[1])
    
    for _ in range(sparsity):
        # Szukamy najlepszego dopasowania atomu slownika
        correlations = np.abs(dictionary.T @ residual)
        new_index = np.argmax(correlations)
        
        if new_index in indices:
            break
            
        indices.append(new_index)
        
        # Aktualizacja wspolczynnikow za pomoca metody najmniejszych kwadratow
        sub_dict = dictionary[:, indices]
        coefficients[indices] = np.linalg.lstsq(sub_dict, signal, rcond=None)[0]
        
        # Aktualizacja resztkowego sygnalu
        residual = signal - sub_dict @ coefficients[indices]
        
        if np.linalg.norm(residual) < 1e-6:
            break
            
    return coefficients
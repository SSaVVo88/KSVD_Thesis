"""Pierwsza wersja skryptu do testowania KSVD na obrazach tekstowych (OCR)"""

# import numpy as np
# from PIL import Image
# import matplotlib.pyplot as plt
# from pyksvd.functions import train_ksvd_models, corrupt_image, reconstruct_image

# # Define parameters specific for text/OCR
# patch_size = 6  # Smaller patches work better for text edges
# K = 256  # Dictionary size - can be tuned
# T0 = 3   # Very sparse representation (text has simple patterns)
# NOISE_LEVEL = 0.3  # Adjust based on your scanner quality

# # Train on text samples (create your own dataset)
# text_train_dir = 'data/ocr_train/'
# text_test_path = 'data/ocr_test/scan.png'

# # Train KSVD models on text images
# ksvd_models = train_ksvd_models(text_train_dir, patch_size, K, T0)

# # Process test image
# test_image = Image.open(text_test_path).convert('L')  # Convert to grayscale
# test_image = test_image.resize((256, 256))
# test_image_array = np.array(test_image, dtype=np.float32) / 255.0

# # Add noise (simulating scanner artifacts)
# corrupted_image = test_image_array + np.random.normal(0, NOISE_LEVEL, test_image_array.shape)
# corrupted_image = np.clip(corrupted_image, 0, 1)

# # Reconstruct
# reconstructed_image = reconstruct_image(corrupted_image, ksvd_models, patch_size)

# # Evaluate OCR performance (pseudocode)
# from pytesseract import image_to_string
# original_text = image_to_string(test_image)
# reconstructed_text = image_to_string(Image.fromarray((reconstructed_image*255).astype(np.uint8)))
# print(f"Original text: {original_text}")
# print(f"Reconstructed text: {reconstructed_text}")
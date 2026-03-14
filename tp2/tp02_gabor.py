# -*- coding: utf-8 -*-
"""
TP02 - Reconnaissance d'Empreinte Digitale
PARTIE C - Methode Gabor (texture orientee)

Principe : Les filtres de Gabor capturent l'orientation des crêtes
de l'empreinte. On applique plusieurs filtres (orientations 0 à pi),
puis on compare les images filtreees.
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
from skimage.filters import gabor_kernel, gabor
from scipy import ndimage as ndi

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def preprocess(image_path: str) -> np.ndarray:
    """Pretraitement : gris, redimensionnement 300x300, egalisation."""
    img = Image.open(image_path)
    img_gris = img.convert("L")
    img_gris = img_gris.resize((300, 300))
    img_gris = np.array(ImageOps.equalize(img_gris))
    return img_gris.astype(np.float64) / 255.0


def apply_gabor_bank(img: np.ndarray, n_orientations: int = 8, frequency: float = 0.1) -> np.ndarray:
    """
    Applique une banque de filtres Gabor a plusieurs orientations.
    Retourne l'image moyenne des reponses (image amelioree).
    """
    responses = []
    for i in range(n_orientations):
        theta = i * np.pi / n_orientations
        kernel = np.real(gabor_kernel(frequency, theta=theta, sigma_x=2, sigma_y=2))
        filtered = ndi.convolve(img, kernel, mode="reflect")
        responses.append(filtered)

    # Moyenne des reponses pour chaque orientation
    mean_response = np.mean(np.abs(responses), axis=0)
    # Normaliser entre 0 et 255 pour comparaison
    mean_response = (mean_response - mean_response.min()) / (mean_response.max() - mean_response.min() + 1e-8)
    return (mean_response * 255).astype(np.uint8)


def compare_fingerprints_gabor(img_path1: str, img_path2: str, seuil: float = 0.5) -> tuple:
    """
    Compare deux empreintes via filtres Gabor + correlation.
    Utilise SSIM sur les images Gabor pour la similarite.
    """
    from skimage.metrics import structural_similarity as compare_ssim

    img1 = preprocess(img_path1)
    img2 = preprocess(img_path2)

    gabor1 = apply_gabor_bank(img1)
    gabor2 = apply_gabor_bank(img2)

    similarity = compare_ssim(gabor1, gabor2, data_range=255)
    decision = "ACCEPTEE" if similarity >= seuil else "REJETEE"

    return similarity, decision, gabor1, gabor2


def visualize(img_path1: str, img_path2: str, similarity: float, decision: str, gabor1: np.ndarray, gabor2: np.ndarray):
    """Affiche originale + Gabor pour chaque empreinte."""
    img1 = preprocess(img_path1)
    img2 = preprocess(img_path2)

    plt.figure(figsize=(12, 5))
    plt.subplot(2, 2, 1)
    plt.imshow(img1, cmap="gray")
    plt.axis("off")
    plt.title("Empreinte 1 (originale)")

    plt.subplot(2, 2, 2)
    plt.imshow(img2, cmap="gray")
    plt.axis("off")
    plt.title("Empreinte 2 (originale)")

    plt.subplot(2, 2, 3)
    plt.imshow(gabor1, cmap="gray")
    plt.axis("off")
    plt.title("Empreinte 1 (Gabor)")

    plt.subplot(2, 2, 4)
    plt.imshow(gabor2, cmap="gray")
    plt.axis("off")
    plt.title("Empreinte 2 (Gabor)")
    plt.suptitle(f"Gabor - Similarite : {similarity:.4f} -> {decision}", fontsize=14)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    img1_path = "empreinte1.jpg"
    img2_path = "empreinte2.png"

    if os.path.exists(img1_path) and os.path.exists(img2_path):
        similarity, decision, g1, g2 = compare_fingerprints_gabor(
            img1_path, img2_path, seuil=0.5
        )
        print(f"Similarite Gabor : {similarity:.4f}")
        print(f"Decision : {decision}")
        visualize(img1_path, img2_path, similarity, decision, g1, g2)
    else:
        print("Fichiers manquants ! empreinte1.jpg et empreinte2.png")

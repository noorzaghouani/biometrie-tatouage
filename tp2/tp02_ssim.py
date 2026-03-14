import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter, ImageOps
from skimage.metrics import structural_similarity as compare_ssim
def preprocess(image_path: str) -> np.ndarray:
    # Charger l'image
    img = Image.open(image_path)

    # Conversion en niveaux de gris
    img_gris = img.convert("L")

    # Redimensionnement (300×300)
    img_gris = img_gris.resize((300, 300))

    # Égalisation histogramme
    img_gris = ImageOps.equalize(img_gris)

    # Binarisation (seuil = 128)
    img_bin = img_gris.point(lambda x: 255 if x > 128 else 0, mode="L")

    # Extraction des contours (FIND_EDGES)
    img_contours = img_bin.filter(ImageFilter.FIND_EDGES)

    return np.array(img_contours)


def compare_fingerprints(img_path1: str, img_path2: str, seuil: float = 1.5) -> tuple:

    img1 = preprocess(img_path1)
    img2 = preprocess(img_path2)

    # Calcul de similarité SSIM (data_range=255 pour images 0-255)
    similarity = compare_ssim(img1, img2, data_range=255)

    # Décision
    decision = "ACCEPTÉE" if similarity >= seuil else "REJETÉE"

    return similarity, decision


def visualize_comparison(img_path1: str, img_path2: str, similarity: float, decision: str):
    """Affiche la comparaison des deux empreintes prétraitées."""
    img1 = preprocess(img_path1)
    img2 = preprocess(img_path2)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img1, cmap="gray")
    plt.axis("off")
    plt.title("Empreinte 1 (prétraitée)")

    plt.subplot(1, 2, 2)
    plt.imshow(img2, cmap="gray")
    plt.axis("off")
    plt.title("Empreinte 2 (prétraitée)")
    plt.suptitle(f"SSIM = {similarity:.4f} → {decision}", fontsize=14)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    img1_path = "empreinte3.png"
    img2_path = "empreinte4.png"

    # Vérifier si les fichiers existent
    if os.path.exists(img1_path) and os.path.exists(img2_path):
        similarity, decision = compare_fingerprints(img1_path, img2_path, seuil=0.75)
        print(f"Similarite SSIM : {similarity:.4f}")
        print(f"Decision : {decision}")

        # Afficher la comparaison
        visualize_comparison(img1_path, img2_path, similarity, decision)
    else:
        print("Fichiers manquants ! Placez empreinte1.jpg et empreinte2.png dans le dossier.")
        print("Ou modifiez img1_path et img2_path dans le script.")

"""
TP01 - Partie 9 : Égalisation de l'histogramme
"""

from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import os

os.makedirs("results", exist_ok=True)

# Charger l'image et la convertir en niveaux de gris
img = Image.open("image_orginale.jpg")
img_gris = img.convert("L")

# Appliquer l'égalisation de l'histogramme
img_egalisee = ImageOps.equalize(img_gris)

# Comparer avant vs après
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(img_gris, cmap="gray")
plt.axis("off")
plt.title("Image en niveaux de gris (avant)")

plt.subplot(1, 2, 2)
plt.imshow(img_egalisee, cmap="gray")
plt.axis("off")
plt.title("Image après égalisation")
plt.tight_layout()

# Sauvegarder
img_egalisee.save("results/image_egalisee.png")

plt.show()

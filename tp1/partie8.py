"""
TP01 - Partie 8 : Histogramme de l'image
"""

from PIL import Image
import matplotlib.pyplot as plt
import os

os.makedirs("results", exist_ok=True)

# Charger l'image et la convertir en niveaux de gris
img = Image.open("image_orginale.jpg")
img_gris = img.convert("L")

# Calculer l'histogramme (répartition des niveaux de gris 0-255)
histogramme = img_gris.histogram()

# Afficher l'histogramme sous forme de courbe
plt.figure(figsize=(10, 5))
plt.plot(histogramme, color="black", linewidth=1)
plt.xlabel("Niveau de gris (0-255)")
plt.ylabel("Nombre de pixels")
plt.title("Histogramme de l'image en niveaux de gris")
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Optionnel : sauvegarder la figure de l'histogramme
plt.savefig("results/histogramme.png", bbox_inches="tight")

plt.show()

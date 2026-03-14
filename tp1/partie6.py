from PIL import Image, ImageFilter
import matplotlib.pyplot as plt
import os

os.makedirs("results", exist_ok=True)

# Charger l'image et la convertir en niveaux de gris
img = Image.open("image_orginale.jpg")
img_gris = img.convert("L")

# Appliquer le filtre de détection des contours
img_contours = img_gris.filter(ImageFilter.FIND_EDGES)

# Comparer image grise vs image contours
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(img_gris, cmap="gray")
plt.axis("off")
plt.title("Image en niveaux de gris")

plt.subplot(1, 2, 2)
plt.imshow(img_contours, cmap="gray")
plt.axis("off")
plt.title("Détection des contours (FIND_EDGES)")
plt.tight_layout()

# Sauvegarder
img_contours.save("results/image_contours.png")

plt.show()

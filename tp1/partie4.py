from PIL import Image
import matplotlib.pyplot as plt
import os

os.makedirs("results", exist_ok=True)
img = Image.open("image_orginale.jpg")
img_gris = img.convert("L")# Convertir l'image en niveaux de gris en utilisant la méthode convert() avec le mode "L" (luminance)

# Comparer original vs gris
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(img)
plt.axis("off")
plt.title("Image originale (couleur)")

plt.subplot(1, 2, 2)
plt.imshow(img_gris, cmap="gray")#afficher l'image en niveaux de gris avec une carte de couleurs "gray"
plt.axis("off")
plt.title("Image en niveaux de gris")# ajouter un titre à l'image en niveaux de gris
plt.tight_layout()

img_gris.save("results/image_gris.png")
plt.show()

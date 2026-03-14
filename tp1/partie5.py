from PIL import Image
import matplotlib.pyplot as plt
import os

os.makedirs("results", exist_ok=True)

img = Image.open("image_orginale.jpg")
img_gris = img.convert("L")
seuil = 128# Définir un seuil pour la binarisation (0-255)
img_binaire = img_gris.point(lambda x: 255 if x > seuil else 0, mode="L")# Appliquer un seuillage binaire en utilisant la méthode point() avec une fonction lambda qui convertit les pixels en noir (0) ou blanc (255) selon le seuil défini
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(img_gris, cmap="gray")#afficher l'image en niveaux de gris avec une carte de couleurs "gray"
plt.axis("off")
plt.title("Image en niveaux de gris")

plt.subplot(1, 2, 2)
plt.imshow(img_binaire, cmap="gray")#afficher l'image binaire avec une carte de couleurs "gray"
plt.axis("off")
plt.title(f"Image binarisée (seuil T={seuil})")# ajouter un titre à l'image binaire avec le seuil utilisé
plt.tight_layout()
img_binaire.save("results/image_binarisee.png")# enregistre l'image binaire dans le dossier "results" avec le nom "image_binarisee.png"
plt.show()

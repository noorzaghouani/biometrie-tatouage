
from PIL import Image, ImageEnhance
import matplotlib.pyplot as plt
import os

os.makedirs("results", exist_ok=True)

img = Image.open("image_orginale.jpg")
enhancer = ImageEnhance.Brightness(img)# Créer un objet pour ajuster la luminosité de l'image
img_lum = enhancer.enhance(1.5)# Augmenter la luminosité de l'image en multipliant les valeurs de pixel par 1.5
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(img)
plt.axis("off")
plt.title("Image originale")

plt.subplot(1, 2, 2)
plt.imshow(img_lum)
plt.axis("off")
plt.title("Luminosité augmentée (facteur 1.5)")
plt.tight_layout()

# Sauvegarder
img_lum.save("results/image_luminosite_augmente.png")

plt.show()

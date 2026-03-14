from PIL import Image, ImageFilter
import matplotlib.pyplot as plt
import os

os.makedirs("results", exist_ok=True)

# Charger l'image originale
img = Image.open("image_orginale.jpg")

# Tester plusieurs rayons : 1, 2, 3
img_flou1 = img.filter(ImageFilter.GaussianBlur(radius=1))
img_flou2 = img.filter(ImageFilter.GaussianBlur(radius=2))
img_flou3 = img.filter(ImageFilter.GaussianBlur(radius=3))

# Comparer les résultats (original + 3 flous)
plt.figure(figsize=(12, 10))
plt.subplot(2, 2, 1)
plt.imshow(img)
plt.axis("off")
plt.title("Image originale")

plt.subplot(2, 2, 2)
plt.imshow(img_flou1)
plt.axis("off")
plt.title("Flou gaussien (radius=1)")

plt.subplot(2, 2, 3)
plt.imshow(img_flou2)
plt.axis("off")
plt.title("Flou gaussien (radius=2)")

plt.subplot(2, 2, 4)
plt.imshow(img_flou3)
plt.axis("off")
plt.title("Flou gaussien (radius=3)")
plt.tight_layout()

# Sauvegarder une image finale (radius=2)
img_flou2.save("results/image_flou_gaussien.png")

plt.show()

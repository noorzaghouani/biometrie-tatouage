from PIL import Image
import matplotlib.pyplot as plt
import os

os.makedirs("results", exist_ok=True)

img = Image.open("image_orginale.jpg")
img_redim = img.resize((200, 200)) # redimensionner l'image à 200x200 pixels
plt.figure(figsize=(10, 5))#afficher les deux images côte à côte dans une figure de taille 10x5 pouces
plt.subplot(1, 2, 1)  # originale
plt.imshow(img)# afficher l'image originale dans la fenetre 
plt.axis("off")#sans axe
plt.title("Image originale")# ajouter un titre à l'image originale

plt.subplot(1, 2, 2)  # redimensionnée
plt.imshow(img_redim)#afficher l'image redimensionnée dans la fenetre
plt.axis("off")
plt.title("Image redimensionnée (200×200)")# ajouter un titre à l'image redimensionnée
plt.tight_layout()

img_redim.save("results/image_redimensionnee.jpg")

plt.show()

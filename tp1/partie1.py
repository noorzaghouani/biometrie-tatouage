from PIL import  Image #permetttra ouvrir et enregistrer image
import matplotlib.pyplot as plt #necessaire pour le traitement et manipulation image en Python 
import os #operating system intégré en python pour interagir avec le systeme de fichiers 
os.makedirs("results", exist_ok=True) #creation du repertoire résultats pour stocker les images traitées

img = Image.open("image_orginale.jpg") # ouvrire l'image 

plt.figure(figsize=(5, 6)) #crée une fenêtre de figure de taille 8x6 pouces pour afficher l'image
plt.subplot(1, 1, 1) #divise la figure en sous-graphes
plt.imshow(img) #affiche une image dans le sous-graphe actuel
plt.axis("on") #masque les axes (graduation x/y)
plt.title("Partie1 image orginale ") #ajoute un titre
plt.tight_layout() #ajuste l'espacement automatiquement

img.save("results/image_redimentionnee.jpg")# enregistre l'image dans le dossier "results" avec le nom "image_originale.png"
plt.show()#affiche la figure à l'écran

"""
TP03 - Vérification Faciale par LBP (Local Binary Patterns)
============================================================
Auteur  : TP Biométrie & Tatouage - ING-4-SSIRF, TEK-UP
Méthodes : Viola-Jones (détection) + LBP (extraction) + Distance Euclidienne (comparaison)

Dépendances : pip install opencv-python scikit-image scipy matplotlib numpy
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import local_binary_pattern
from scipy.spatial.distance import euclidean


# =============================================================================
# CLASSE PRINCIPALE : FaceVerificationSystem
# =============================================================================

class FaceVerificationSystem:
    """
    Système de vérification faciale 1:1 basé sur LBP + Viola-Jones.

    Attributs
    ---------
    face_cascade : cv2.CascadeClassifier
        Classificateur Viola-Jones pré-entraîné pour la détection de visages.
    reference_features : np.ndarray or None
        Vecteur LBP du visage de référence (None si non initialisé).
    reference_image : np.ndarray or None
        Image du visage de référence (pour l'affichage).

    Paramètres LBP
    --------------
    radius : int  = 1   -> rayon du cercle de voisinage
    n_points : int = 8  -> nombre de points voisins (8 * radius recommandé)
    """

    # -------------------------------------------------------------------
    # Paramètres LBP
    # -------------------------------------------------------------------
    LBP_RADIUS   = 1
    LBP_N_POINTS = 8 * LBP_RADIUS     # 8 voisins
    LBP_METHOD   = "uniform"           # LBP uniforme (moins de bruit)
    FACE_SIZE    = (128, 128)          # Taille normalisée avant extraction
    GRID_SIZE    = 8                   # Grille NxN de cellules (8×8 = 64 cellules)

    def __init__(self):
        """
        Initialise le détecteur de visage avec le classificateur Haar d'OpenCV.
        """
        # Chemin vers le fichier de cascade fourni par OpenCV
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.face_cascade = cv2.CascadeClassifier(cascade_path)

        if self.face_cascade.empty():
            raise IOError(
                f"Impossible de charger le classificateur Haar : {cascade_path}\n"
                "Vérifiez que opencv-python est correctement installé."
            )

        self.reference_features = None
        self.reference_image    = None

    # -------------------------------------------------------------------
    # 1. DÉTECTION DE VISAGE (Viola-Jones)
    # -------------------------------------------------------------------

    def detect_face(self, image: np.ndarray):
        """
        Détecte le plus grand visage dans une image BGR.

        Paramètres
        ----------
        image : np.ndarray
            Image BGR (lue avec cv2.imread).

        Retourne
        --------
        tuple (x, y, w, h) : coordonnées du rectangle du visage le plus grand,
                              ou None si aucun visage n'est détecté.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Détection avec les paramètres imposés par le TP
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor  = 1.1,
            minNeighbors = 5,
            minSize      = (30, 30)
        )

        if len(faces) == 0:
            return None  # Aucun visage détecté

        # Règle du TP : garder uniquement le plus grand visage (surface w*h max)
        largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
        return tuple(largest_face)   # (x, y, w, h)

    # -------------------------------------------------------------------
    # 2. EXTRACTION DES CARACTÉRISTIQUES LBP
    # -------------------------------------------------------------------

    def extract_lbp_features(self, face_image: np.ndarray) -> np.ndarray:
        """
        Extrait le vecteur LBP sur une grille NxN de cellules.

        Étapes :
          1. Conversion en niveaux de gris
          2. Redimensionnement à 128x128 px (imposé par le TP)
          3. Découpage en GRID_SIZE × GRID_SIZE cellules
          4. Calcul de la carte LBP (skimage) sur chaque cellule
          5. Histogramme normalisé par cellule → concaténation

        Pourquoi la grille ?
          Un histogramme global perd l'information spatiale.
          La grille encode AUSSI la position des textures
          (yeux, nez, bouche, joues) → bien plus discriminant.

        Paramètres
        ----------
        face_image : np.ndarray
            Image (BGR ou grayscale) du visage recadré.

        Retourne
        --------
        np.ndarray : Vecteur LBP spatial (GRID² × n_bins valeurs).
        """
        # --- Niveaux de gris ---
        if len(face_image.shape) == 3:
            gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = face_image.copy()

        # --- Redimensionnement imposé par le TP ---
        resized = cv2.resize(gray, self.FACE_SIZE, interpolation=cv2.INTER_AREA)

        # --- Calcul LBP sur toute l'image (skimage) ---
        lbp_map = local_binary_pattern(
            resized,
            P      = self.LBP_N_POINTS,
            R      = self.LBP_RADIUS,
            method = self.LBP_METHOD
        )
        n_bins  = self.LBP_N_POINTS + 2   # nombre de bins pour LBP uniforme

        # --- Extraction par cellule (grille GRID_SIZE × GRID_SIZE) ---
        h, w       = resized.shape
        cell_h     = h // self.GRID_SIZE
        cell_w     = w // self.GRID_SIZE
        histograms = []

        for row in range(self.GRID_SIZE):
            for col in range(self.GRID_SIZE):
                # Découper la cellule dans la carte LBP
                r0, r1 = row * cell_h, (row + 1) * cell_h
                c0, c1 = col * cell_w, (col + 1) * cell_w
                cell   = lbp_map[r0:r1, c0:c1]

                # Histogramme normalisé de la cellule
                hist, _ = np.histogram(cell.ravel(), bins=n_bins, range=(0, n_bins))
                hist     = hist.astype(float)
                hist    /= (hist.sum() + 1e-7)   # Normalisation L1
                histograms.append(hist)

        # Vecteur final = concaténation de tous les histogrammes
        return np.concatenate(histograms)

    # -------------------------------------------------------------------
    # 3. ENREGISTREMENT DU VISAGE DE RÉFÉRENCE
    # -------------------------------------------------------------------

    def setup_reference(self, image_path: str):
        """
        Charge l'image de référence, détecte le visage et extrait ses features LBP.

        Paramètres
        ----------
        image_path : str
            Chemin vers l'image de référence.

        Lève
        ----
        FileNotFoundError  : si l'image ne peut pas être chargée.
        ValueError         : si aucun visage n'est détecté.
        """
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Image introuvable : {image_path}")

        face_coords = self.detect_face(image)
        if face_coords is None:
            raise ValueError(f"Aucun visage détecté dans l'image de référence : {image_path}")

        x, y, w, h = face_coords
        face_roi = image[y:y+h, x:x+w]

        self.reference_features = self.extract_lbp_features(face_roi)
        self.reference_image    = image.copy()

        print(f"[✓] Référence chargée : {image_path}")
        print(f"    Visage détecté à : (x={x}, y={y}, w={w}, h={h})")

    # -------------------------------------------------------------------
    # 4. VÉRIFICATION D'UN VISAGE
    # -------------------------------------------------------------------

    def verify_face(self, image_path: str, threshold: float = 0.75) -> dict:
        """
        Vérifie si le visage dans image_path correspond au visage de référence.

        Paramètres
        ----------
        image_path : str
            Chemin vers l'image à vérifier.
        threshold  : float
            Seuil de similarité (défaut = 0.75 selon le TP).

        Retourne
        --------
        dict avec les clés :
            - "similarity"   : float  (score de similarité)
            - "distance"     : float  (distance euclidienne brute)
            - "match"        : bool   (True si résultat = Match)
            - "decision"     : str    ("MATCH" ou "NO MATCH")
            - "face_coords"  : tuple  (x, y, w, h) ou None
            - "image"        : np.ndarray (image annotée BGR)

        Lève
        ----
        RuntimeError : si setup_reference() n'a pas encore été appelé.
        """
        if self.reference_features is None:
            raise RuntimeError("Appelez d'abord setup_reference() pour enregistrer l'image de référence.")

        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Image introuvable : {image_path}")

        annotated = image.copy()
        face_coords = self.detect_face(image)

        if face_coords is None:
            print("[✗] Aucun visage détecté dans l'image de test.")
            return {
                "similarity" : 0.0,
                "distance"   : float("inf"),
                "match"      : False,
                "decision"   : "NO MATCH (aucun visage détecté)",
                "face_coords": None,
                "image"      : annotated,
            }

        x, y, w, h = face_coords
        face_roi = image[y:y+h, x:x+w]

        # Extraction des features du visage test
        test_features = self.extract_lbp_features(face_roi)

        # --- Calcul de la distance euclidienne ---
        distance = euclidean(self.reference_features, test_features)

        # --- Conversion en similarité (formule du TP : 1 - distance normalisée) ---
        # Chaque cellule est L1-normalisée → distance max par cellule = sqrt(2)
        # Avec GRID^2 cellules : max_dist = sqrt(2 × GRID^2) = sqrt(2 × 64) ≈ 11.31
        max_dist   = np.sqrt(2.0 * (self.GRID_SIZE ** 2))
        similarity = 1.0 - (distance / max_dist)

        # --- Décision par seuillage ---
        is_match = similarity >= threshold
        decision = "MATCH" if is_match else "NO MATCH"

        # --- Annotation de l'image ---
        color = (0, 255, 0) if is_match else (0, 0, 255)   # Vert = Match, Rouge = No Match
        cv2.rectangle(annotated, (x, y), (x+w, y+h), color, 2)
        label = f"{decision} ({similarity:.3f})"
        cv2.putText(
            annotated, label,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
        )

        return {
            "similarity" : similarity,
            "distance"   : distance,
            "match"      : is_match,
            "decision"   : decision,
            "face_coords": face_coords,
            "image"      : annotated,
        }


# =============================================================================
# PROGRAMME PRINCIPAL
# =============================================================================

def display_results(ref_image_path: str, test_image_path: str, result: dict):
    """
    Affiche côte-à-côte l'image de référence et l'image de test annotée.

    Paramètres
    ----------
    ref_image_path  : str
    test_image_path : str
    result          : dict retourné par verify_face()
    """
    ref_img  = cv2.imread(ref_image_path)
    test_img = result["image"]

    # Convertir BGR → RGB pour matplotlib
    ref_rgb  = cv2.cvtColor(ref_img,  cv2.COLOR_BGR2RGB)
    test_rgb = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle(
        f"Vérification Faciale LBP  |  {result['decision']}\n"
        f"Similarité = {result['similarity']:.4f}   |   Distance = {result['distance']:.4f}",
        fontsize=13, fontweight="bold",
        color="green" if result["match"] else "red"
    )

    axes[0].imshow(ref_rgb)
    axes[0].set_title("Image de référence", fontsize=11)
    axes[0].axis("off")

    axes[1].imshow(test_rgb)
    axes[1].set_title("Image de test (annotée)", fontsize=11)
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()


def main():
    """
    Programme principal :
      1. Charge les images de référence et de test.
      2. Configure le système avec la référence.
      3. Vérifie l'image de test.
      4. Affiche les résultats (console + images).
    """
    # ------------------------------------------------------------------
    # ► MODIFIEZ CES DEUX CHEMINS selon vos propres images
    # ------------------------------------------------------------------
    REFERENCE_IMAGE = "reference.jpg"   # ex: photo passeport
    TEST_IMAGE      = "test2.jpg"        # ex: photo webcam à vérifier
    THRESHOLD       = 0.90              # Seuil recalibré pour LBP grille 8×8
    # ------------------------------------------------------------------

    print("=" * 55)
    print("   SYSTÈME DE VÉRIFICATION FACIALE — TP03 LBP")
    print("=" * 55)

    # --- 1. Instanciation du système ---
    system = FaceVerificationSystem()

    # --- 2. Chargement de la référence ---
    try:
        system.setup_reference(REFERENCE_IMAGE)
    except (FileNotFoundError, ValueError) as e:
        print(f"[ERREUR] {e}")
        return

    # --- 3. Vérification de l'image de test ---
    print(f"\n[→] Vérification de : {TEST_IMAGE}")
    try:
        result = system.verify_face(TEST_IMAGE, threshold=THRESHOLD)
    except (FileNotFoundError, RuntimeError) as e:
        print(f"[ERREUR] {e}")
        return

    # --- 4. Affichage des résultats en console ---
    print("\n" + "-" * 55)
    print(f"  Distance euclidienne : {result['distance']:.6f}")
    print(f"  Similarité           : {result['similarity']:.6f}")
    print(f"  Seuil de décision    : {THRESHOLD}")
    print("-" * 55)
    print(f"  ► DÉCISION : {result['decision']}")
    print("-" * 55)

    # --- 5. Affichage des images avec rectangles de détection ---
    display_results(REFERENCE_IMAGE, TEST_IMAGE, result)


if __name__ == "__main__":
    main()

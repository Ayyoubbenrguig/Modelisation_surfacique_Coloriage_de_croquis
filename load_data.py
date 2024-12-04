import cv2
import numpy as np
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt

def load_image(filepath):
    """
    Charge une image en niveaux de gris et la convertit en binaire inversée.
    :param filepath: Chemin vers le fichier image.
    :return: Une image binaire inversée (numpy array).
    """
    image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"L'image à {filepath} est introuvable.")
    _, binary_image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY_INV)
    return binary_image

def extract_points(binary_image):
    """
    Extrait les coordonnées des pixels blancs d'une image binaire.
    :param binary_image: Image binaire (numpy array).
    :return: Tableau numpy contenant les coordonnées des points.
    """
    points = np.column_stack(np.where(binary_image > 0))
    return points

# Générer la triangulation de Delaunay

def delaunay_triangulation(points):
    triangulation = Delaunay(points)
    return triangulation

# Visualiser les résultats
def plot_triangulation(points, triangulation):
    plt.figure(figsize=(8, 8))
    plt.triplot(points[:, 1], points[:, 0], triangulation.simplices, color='blue')
    plt.scatter(points[:, 1], points[:, 0], color='red', s=1)
    plt.gca().invert_yaxis()
    plt.title("Triangulation de Delaunay")
    plt.show()

# Exemple d'utilisation
if __name__ == "__main__":
    # Chemin de l'image
    filepath = "Dino.png"  # Remplacez par votre fichier d'entrée
    binary_image = load_image(filepath)
    points = extract_points(binary_image)
    triangulation = delaunay_triangulation(points)
    plot_triangulation(points, triangulation)


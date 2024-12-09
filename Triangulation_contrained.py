import cv2
import numpy as np
import matplotlib.pyplot as plt
import triangle  # Bibliothèque pour la triangulation contrainte


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


def delaunay_constrained(points, constraints):
    """
    Effectue une triangulation contrainte en utilisant la bibliothèque 'triangle'.
    :param points: Tableau numpy contenant les coordonnées des points.
    :param constraints: Liste de segments définissant les contraintes (index des points).
    :return: Une structure contenant les triangles.
    """
    # Préparer les données pour 'triangle'
    data = {
        "vertices": points,  # Points
        "segments": constraints  # Contraintes
    }


    # Générer la triangulation avec contraintes
    triangulation = triangle.triangulate(data, 'p')  # 'p' pour triangulation contrainte

    # Ajoutez ici pour vérifier le contenu de la triangulation
    print("Clés dans triangulation :", triangulation.keys())
    print("Triangulation complète :", triangulation)
    return triangulation


def generate_constraints(binary_image, points):
    """
    Génère des segments contraints en suivant les contours de l'image.
    :param binary_image: Image binaire.
    :param points: Points extraits de l'image.
    :return: Liste de segments définissant les contraintes.
    """
    # Trouver les contours
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    constraints = []
    for contour in contours:
        # Récupérer les indices des points sur le contour
        contour = contour.squeeze()  # Retirer les dimensions inutiles
        indices = [np.argmin(np.linalg.norm(points - p, axis=1)) for p in contour]

        # Ajouter les segments
        for i in range(len(indices)):
            constraints.append([indices[i], indices[(i + 1) % len(indices)]])
    
    return constraints



def plot_triangulation_constrained(points, triangulation):
    """
    Visualise la triangulation contrainte.
    :param points: Tableau numpy contenant les coordonnées des points.
    :param triangulation: Structure contenant les données de triangulation.
    """
    plt.figure(figsize=(8, 8))

    # Vérifiez si des triangles ont été générés
    if "triangles" in triangulation:
        for tri in triangulation["triangles"]:
            if all(idx < len(points) for idx in tri):
                triangle_pts = points[tri]
                plt.fill(triangle_pts[:, 1], triangle_pts[:, 0], edgecolor='blue', fill=False)

    # Dessiner les points
    plt.scatter(points[:, 1], points[:, 0], color='red', s=1)
    plt.gca().invert_yaxis()
    plt.title("Triangulation contrainte")
    plt.show()

if __name__ == "__main__":
    # Chemin de l'image
    filepath = "Dino.png"  # Remplacez par votre fichier d'entrée
    binary_image = load_image(filepath)
    points = extract_points(binary_image)

    # Générer les contraintes
    constraints = generate_constraints(binary_image, points)

    # Effectuer la triangulation contrainte
    triangulation = delaunay_constrained(points, constraints)

    # Visualiser la triangulation
    plot_triangulation_constrained(points, triangulation)



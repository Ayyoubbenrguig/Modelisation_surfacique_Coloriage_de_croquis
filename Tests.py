import cv2
import matplotlib.pyplot as plt
from load_data import load_image, extract_points

def test_load_image():
    """
    Teste la fonction load_image.
    - Charge une image.
    - Affiche l'image binaire inversée.
    """
    filepath = "Santa.png"  # Remplacez par votre image de test
    binary_image = load_image(filepath)
    plt.imshow(binary_image, cmap='gray')
    plt.title("Image binaire inversée")
    plt.axis('off')
    #plt.show()

def test_extract_points():
    """
    Teste la fonction extract_points.
    - Charge une image et extrait les points 2D.
    - Affiche les points sur un graphique.
    """
    filepath = "Santa.png"  # Remplacez par votre image de test
    binary_image = load_image(filepath)
    points = extract_points(binary_image)
    print(f"Nombre de points extraits : {len(points)}")
    
    # Affichage des points
    plt.scatter(points[:, 1], points[:, 0], s=1, color='red')
    plt.gca().invert_yaxis()  # Inverser l'axe Y pour correspondre à l'image
    plt.title("Points extraits")
    plt.axis('equal')
    plt.show()

if __name__ == "__main__":
    print("=== Test de load_image ===")
    test_load_image()
    
    print("=== Test de extract_points ===")
    test_extract_points()


import matplotlib.pyplot as plt
from load_data import load_image, extract_points
from coloriage_initial import propagate_colors
from scipy.spatial import Delaunay

def test_coloriage_initial():
    filepath = "Dino.png"  # Chemin vers une image de test
    binary_image = load_image(filepath)
    points = extract_points(binary_image)
    triangulation = Delaunay(points)
    colors = propagate_colors(points, triangulation.simplices, threshold=5.0)
    
    # Affichage des triangles colorés
    plt.figure(figsize=(8, 8))
    for triangle_index, color in colors.items():
        simplex = triangulation.simplices[triangle_index]
        plt.fill(
            points[simplex, 1], points[simplex, 0],
            color=plt.cm.tab20(color % 20), alpha=0.6
        )
    plt.gca().invert_yaxis()
    plt.title("Coloriage initial des triangles")
    plt.show()

if __name__ == "__main__":
    test_coloriage_initial()

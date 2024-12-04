from load_data import load_image, extract_points, delaunay_triangulation
from coloriage_initial import propagate_colors
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from collections import defaultdict

# Fonction pour créer un graphe d'adjacence entre les triangles
def build_adjacency_graph(points, simplices):
    adjacency = defaultdict(list)
    for i, simplex1 in enumerate(simplices):
        for j, simplex2 in enumerate(simplices):
            if i != j and len(set(simplex1) & set(simplex2)) == 2:
                adjacency[i].append(j)
    return adjacency

# Fonction de recoloration interactive avec propagation
def recolor_interactive(points, simplices, initial_colors, adjacency):
    colors = initial_colors.copy()  # Reprend les couleurs initiales
    fig, ax = plt.subplots(figsize=(8, 8))

    def plot_triangulation():
        ax.clear()
        max_color = max(colors.values())
        color_map = mcolors.ListedColormap(plt.cm.tab10.colors[:max_color])
        for index, simplex in enumerate(simplices):
            triangle = points[simplex]
            ax.fill(triangle[:, 1], triangle[:, 0], color=color_map(colors[index]), edgecolor='black', alpha=0.6)
        ax.invert_yaxis()
        ax.set_title("Recoloration Interactive")
        plt.draw()

    def on_click(event):
        if event.inaxes != ax:
            return
        x, y = event.xdata, event.ydata
        clicked_triangle = None
        for index, simplex in enumerate(simplices):
            triangle = points[simplex]
            path = plt.Polygon(triangle[:, [1, 0]]).get_path()
            if path.contains_point((x, y)):
                clicked_triangle = index
                break
        if clicked_triangle is None:
            return

        # Nouvelle couleur pour la propagation
        new_color = max(colors.values()) + 1
        queue = [clicked_triangle]
        visited = set()

        # Propagation de la couleur
        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)
            colors[current] = new_color

            for neighbor in adjacency[current]:
                shared_edge = sorted(set(simplices[current]) & set(simplices[neighbor]))
                edge_length = sum(
                    np.linalg.norm(points[p1] - points[p2])
                    for p1, p2 in zip(shared_edge, shared_edge[1:] + shared_edge[:1])
                )
                if edge_length > 5.0 and neighbor not in colors:
                    queue.append(neighbor)

        plot_triangulation()

    # Initialisation de la triangulation et de l'affichage
    plot_triangulation()
    fig.canvas.mpl_connect('button_press_event', on_click)
    plt.show()

if __name__ == "__main__":
    # Charger l'image et extraire les points
    filepath = "Dino.png"
    binary_image = load_image(filepath)
    points = extract_points(binary_image)
    triangulation = delaunay_triangulation(points)
    simplices = triangulation.simplices

    # Générer la coloration initiale avec propagation
    initial_colors = propagate_colors(points, simplices, threshold=5.0)

    # Construire le graphe d'adjacence pour la propagation des couleurs
    adjacency = build_adjacency_graph(points, simplices)

    # Démarrer la recoloration interactive
    recolor_interactive(points, simplices, initial_colors, adjacency)

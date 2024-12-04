from load_data import load_image, extract_points, delaunay_triangulation
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def compute_triangle_size(points, simplices):
    sizes = []
    for i, simplex in enumerate(simplices):
        p1, p2, p3 = points[simplex]
        perimeter = (
            np.linalg.norm(p1 - p2) +
            np.linalg.norm(p2 - p3) +
            np.linalg.norm(p3 - p1)
        )
        sizes.append((perimeter, i))
    return sizes

def sort_triangles_by_size(triangle_sizes):
    return [index for _, index in sorted(triangle_sizes, reverse=True)]

def propagate_colors(points, simplices, threshold=5.0):
    triangle_sizes = compute_triangle_size(points, simplices)
    sorted_indices = sort_triangles_by_size(triangle_sizes)
    colors = {}
    color_counter = 0
    adjacency = defaultdict(list)
    for i, simplex in enumerate(simplices):
        for j in range(len(simplices)):
            if i != j and len(set(simplex) & set(simplices[j])) == 2:
                adjacency[i].append(j)
    for triangle_index in sorted_indices:
        if triangle_index in colors:
            continue
        color_counter += 1
        current_color = color_counter
        queue = [triangle_index]
        while queue:
            current = queue.pop(0)
            if current in colors:
                continue
            colors[current] = current_color
            for neighbor in adjacency[current]:
                shared_edge = sorted(set(simplices[current]) & set(simplices[neighbor]))
                edge_length = sum(
                    np.linalg.norm(points[p1] - points[p2])
                    for p1, p2 in zip(shared_edge, shared_edge[1:] + shared_edge[:1])
                )
                if edge_length > threshold and neighbor not in colors:
                    queue.append(neighbor)
    return colors

def plot_colored_triangulation(points, simplices, colors):
    plt.figure(figsize=(8, 8))
    max_color = max(colors.values())
    color_map = mcolors.ListedColormap(plt.cm.tab10.colors[:max_color])
    for index, simplex in enumerate(simplices):
        triangle = points[simplex]
        plt.fill(triangle[:, 1], triangle[:, 0], color=color_map(colors[index]), edgecolor='black', alpha=0.6)
    plt.gca().invert_yaxis()
    plt.title("Triangulation colorée")
    plt.show()

if __name__ == "__main__":
    filepath = "Dino.png"
    binary_image = load_image(filepath)
    points = extract_points(binary_image)
    triangulation = delaunay_triangulation(points)
    simplices = triangulation.simplices
    colors = propagate_colors(points, simplices, threshold=5.0)
    plot_colored_triangulation(points, simplices, colors)


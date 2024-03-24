#Korpos Botond
#522/2

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
from queue import PriorityQueue

mode = "minimalis tavolsag"
DIM = 100

def create_3d_plot(optpath, points):
    x_points, y_points, z_points, b_points = np.array(points).T
    x_path, y_path, z_path, _ = np.array(optpath).T
    obstacle_points = np.array([point[:3] for point in points if point[3] == 1])

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')

    intensity = z_points / np.max(z_points)

    ax.scatter(x_points, y_points, z_points, c=intensity, cmap='coolwarm', marker=',', alpha=0.1)
    ax.scatter(obstacle_points[:, 0], obstacle_points[:, 1], obstacle_points[:, 2], c='black', marker='.', alpha=1)
    ax.plot(x_path, y_path, z_path, c='g', linewidth=2, marker='o')

    #cbar = fig.colorbar(cm.ScalarMappable(cmap='hot'), ax=ax)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()

def create_heatmaps(heatmap2D, heatmap3D, optpath, points):
    x_path, y_path, z_path, _ = np.array(optpath).T
    #plt.imshow(x_path, y_path, c='g')

    plt.plot(y_path, x_path, c='g', marker='.')
    #plt.scatter(x_c[1:], y_c[1:], c='b')
    plt.imshow(heatmap2D, cmap='hot_r')
    plt.title("2D Heatmap")
    plt.colorbar()
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()


    x_points, y_points, z_points, b_points = np.array(points).T
    #obstacle_points = np.array([point[:3] for point in points if point[3] == 1])

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    nph = np.array(heatmap3D)

    ax.scatter(x_points, y_points, z_points, c=nph, cmap='hot_r', marker=',', alpha=0.4)
    #ax.scatter(obstacle_points[:, 0], obstacle_points[:, 1], obstacle_points[:, 2], c='black', marker='.', alpha=1)
    #ax.plot(y_path, x_path, 0, c='g', linewidth=2, marker='o')

    #cbar = fig.colorbar(cm.ScalarMappable(cmap='hot'), ax=ax)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()

def read_points(filename):
    points = []
    with open(filename, 'r') as file:
        for line in file:
            x, y, z, b = map(float, line.split())
            point = (x, y, z, int(b))
            points.append(point)
    return points


def read_end_points(filename):
    with open(filename, 'r') as file:
        start_line = file.readline()
        end_line = file.readline()

        start = tuple(map(float, start_line.split()))
        finish = tuple(map(float, end_line.split()))

    return start, finish


def get_end_points(start, finish, points):
    start_p = next((point for point in points if point[:2] == start), None)
    finish_p = next((point for point in points if point[:2] == finish), None)

    return start_p, finish_p


def distance3D(point1, point2):
    return np.sqrt((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2 + (point2[2] - point1[2]) ** 2)


def distance2D(point1, point2):
    return max(abs(point2[0] - point1[0]), abs(point2[1] - point1[1]))


def heuristic(mode, point1, point2):
    if mode == "minimalis tavolsag":
        return distance3D(point1, point2)
    elif mode == "minimalis lepesszam":
        return distance2D(point1, point2)


def get_neighbors(point, points_set):
    neighbors = []
    x, y = point[:2]
    for dx in range(-1, 2):
        for dy in range(-1, 2):
            neighbor = (x + dx, y + dy)
            if neighbor != point[:2] and neighbor in points_set:
                neighbors.append(points_set[neighbor])
    return neighbors


def a_star_search(points, start, finish, heatmap2D, heatmap3D):
    start_p, finish_p = get_end_points(start, finish, points)
    points_set = {point[:2]: point for point in points}
    pq = PriorityQueue()
    pq.put((0, start_p))
    optpath = {start_p: None}
    cost = {start_p: 0}

    while not pq.empty():
        current_point = pq.get()[1]

        if current_point == finish_p:
            correctpath = reconstruct_path(optpath, start_p, finish_p)
            return correctpath, cost[finish_p]

        for neighbor in get_neighbors(current_point, points_set):
            newcost = cost[current_point] + heuristic(mode, current_point, neighbor)
            heatmap2D[int(neighbor[0])][int(neighbor[1])] += 1
            heatmap3D[int(neighbor[0])][int(neighbor[1])] += 1

            if (neighbor not in cost or newcost < cost.get(neighbor, float('inf'))) and neighbor[3] == 0:
                cost[neighbor] = newcost
                priority = newcost + heuristic(mode, neighbor, finish_p)
                pq.put((priority, neighbor))
                optpath[neighbor] = current_point

    return None, None


def reconstruct_path(optpath, start_p, finish_p):
    current = finish_p
    path = []
    while current != start_p:
        path.append(current)
        current = optpath[current]
    return path[::-1]

def write_to_file(optcost, optpath):
    f = open("output_t.txt", "w")
    f.write(str(optcost) + "\n")
    for opt in optpath:
        f.write(str(int(opt[0])) + " " + str(int(opt[1])) + "\n")
    f.close()


def main():
    heatmap2D = [[0 for i in range(DIM)] for j in range(DIM)]
    heatmap3D = [[0 for i in range(DIM)] for j in range(DIM)]

    inputfile = "surface_" + str(DIM) + "x" + str(DIM) + ".txt"
    inputfile_endpoints = "surface_" + str(DIM) + "x" + str(DIM) + ".end_points.txt"

    points = read_points(inputfile)
    start, finish = read_end_points(inputfile_endpoints)
    optpath, optcost = a_star_search(points, start, finish, heatmap2D, heatmap3D)
    if optpath is not None:
        create_heatmaps(heatmap2D, heatmap3D, optpath, points)
        create_3d_plot(optpath, points)
        write_to_file(optcost, optpath)


if __name__ == "__main__":
    main()
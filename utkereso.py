from math import sqrt
from queue import PriorityQueue
import matplotlib.pyplot as plt
import numpy as np

ROW = 100
COL = 100
mode = "minimalis tavolsag"

def readPoints(filename):
    points = []
    with open(filename, 'r') as file:
        for line in file:
            x, y, z, b = map(float, line.split())
            point = (x, y, z, int(b))
            points.append(point)
    return points

def readEndPoints(filename):
    with open(filename, 'r') as file:
        start_line = file.readline()
        end_line = file.readline()
        
        start = tuple(map(float, start_line.split()))
        finish = tuple(map(float, end_line.split()))

    return start, finish

def getEndPoints(start, finish, points):
    for point in points:
        if point[0] == start[0] and point[1] == start[1]:
            start_p = point
        if point[0] == finish[0] and point[1] == finish[1]:
            finish_p = point

    return start_p, finish_p

def distance3D(point1, point2):
    return sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2 + (point1[3] - point2[3])**2)

def distance2D(point1, point2):
    return max(abs(point2[0] - point1[0]), abs(point2[1] - point1[1]))

def heuristic(mode, point1, point2):
    if mode == "minimalis tavolsag":
        return distance3D(point1, point2)
    elif mode == "minimalis lepesszam":
        return distance2D(point1, point2)
    else:
        print("Hibas mod!")
        return 0
        

def getNeighbors(point, points):
    neighbors = []
    for p in points:
        if p == point:
            continue
        dx = abs(p[0] - point[0])
        dy = abs(p[1] - point[1])
        if dx <= 1 and dy <= 1:
            neighbors.append(p)
    return neighbors

def aStarSearch(points, start, finish):

    start_p, finish_p = getEndPoints(start, finish, points)

    pq = PriorityQueue()
    pq.put((0, start_p))
    optpath = {start_p: None}
    cost = {start_p: 0}

    while not pq.empty():

        current_point = pq.get()[1]

        if current_point == finish_p:
            correctpath = reconstructPath(optpath, start_p, finish_p)
            return correctpath, cost[finish_p]
        
        for neighbor in getNeighbors(current_point, points):
            newcost = cost[current_point] + heuristic(mode, current_point, neighbor)

            if neighbor not in (cost or newcost < cost[neighbor]) and neighbor[3] == 0:
                cost[neighbor] = newcost
                priority = newcost + heuristic(mode, neighbor, finish_p)
                pq.put((priority, neighbor))
                optpath[neighbor] = current_point

def reconstructPath(optpath, start_p, finish_p):
    current = finish_p
    path = []
    while current != start_p:
        path.append(current)
        current = optpath[current]
    return path[::-1]


def create_heatmap(optpath, points, ROW, COL):
    heatmap = np.zeros((ROW, COL))

    
    for point in points:
        intensity = point[2] / np.max([p[2] for p in points])
        heatmap[int(point[0]), int(point[1])] = 1 - intensity

    for point in optpath:
        heatmap[int(point[0]), int(point[1])] = 0.1

    plt.imshow(heatmap, cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.show()


def main():
    points = readPoints("surface_100x100.txt")
    start, finish = readEndPoints("surface_100x100.end_points.txt")
    optpath, optcost = aStarSearch(points, start, finish)
    create_heatmap(optpath, points, ROW, COL)
    
    print(optcost)

if __name__ == "__main__":
    main()
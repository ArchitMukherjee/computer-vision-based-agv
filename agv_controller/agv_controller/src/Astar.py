

import heapq
import math
import numpy as np


def distance(a, b):
    # Euclidean distance between two points
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

def astar_diagonal(maze, start, end):
    """
    Finds the shortest path from start to end in a maze using A* algorithm, with diagonal movements.

    Parameters:
        maze (list of list of int): A rectangular grid representing the maze. The
            value 0 represents a free cell and the value 1 represents an obstacle.
        start (tuple of int): The starting position in the maze, specified as a
            tuple (row, column).
        end (tuple of int): The ending position in the maze, specified as a tuple
            (row, column).

    Returns:
        path (list of tuple of int): A list of positions on the shortest path from
            start to end, including both start and end.
    """
    # Define helper functions
    def neighbors(node):
        # Generate the neighbors of a node
        row, col = node
        candidates = [(row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1),
                      (row - 1, col - 1), (row - 1, col + 1), (row + 1, col - 1), (row + 1, col + 1)]
        return [(int(r), int(c)) for r, c in candidates if 0 <= r < len(maze) and 0 <= c < len(maze[0]) and maze[r][c] == 0]

    # Initialize data structures
    frontier = [(0, start)]
    came_from = {}
    cost_so_far = {}
    came_from[start] = None
    cost_so_far[start] = 0

    # Run the A* algorithm
    while frontier:
        _, current = heapq.heappop(frontier)

        if current == end:
            break

        for neighbor in neighbors(current):
            new_cost = cost_so_far[current] + distance(current, neighbor)
            if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                cost_so_far[neighbor] = new_cost
                priority = new_cost + distance(end, neighbor)
                heapq.heappush(frontier, (priority, neighbor))
                came_from[neighbor] = current

    # Extract the path from the came_from dictionary
    path = [end]
    while path[-1] != start:
        path.append(came_from[path[-1]])
    path.reverse()
    path = np.array(path)
    path_list=path.tolist()
    path_list = path_list[1:len(path_list)]
    # for i in range(len(path_list)):
    #    path_list[i] = path_list[i][1],path_list[i][0]
    return path_list

'''
maze = Main.get_map()
x,y,theta = Main.get_current_pose()
start = (y,x)
end = (13,24)

path = astar_diagonal(maze,start,end)
print(path)
'''
# coding: utf-8
from itertools import permutations
import matplotlib.pyplot as plt
import numpy as np
import sys

def distance(p1, p2):
    """
    Returns the Euclidean distance of two points in the Cartesian Plane.

    >>> distance([3,4],[0,0])
    5.0

    """
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5

def total_distance(points):
    """
    Returns the length of the path passing throught
    all the points in the given order.

    >>> total_distance([[1,2],[4,6]])
    5.0
    >>> total_distance([[3,6],[7,6],[12,6]])
    9.0
    """
    return sum([distance(point, points[index + 1]) for index, point in enumerate(points[:-1])])


# - keep in mind that \[:-1\] means "all elements if the list without the last"
# - *enumerate* is a function to enumerate all elements of a given sequence
# In[6]:
def traveling_salesman(points, start=None):
    """
    Finds the shortest route to visit all the cities by bruteforce.
    Time complexity is O(N!), so never use on long lists.

    >>> travelling_salesman([[0,0],[10,0],[6,0]])
    ([0, 0], [6, 0], [10, 0])
    >>> travelling_salesman([[0,0],[6,0],[2,3],[3,7],[0.5,9],[3,5],[9,1]])
    ([0, 0], [6, 0], [9, 1], [2, 3], [3, 5], [3, 7], [0.5, 9])
    """
    if start is None:
        start = points[0]
    return min([perm for perm in permutations(points) if perm[0] == start], key=total_distance)

# - *permutations* returns tuples with all possible orderings without repeat
# - function returns minimum of all possible tuples by the help of the function *total_distance* from above
# In[21]:

import datetime

def main():
    points = [
        [0, 0],
        [1, 5.7],
        [2, 3],
        [3, 7],
        [0.5, 9],
        [3, 5],
        [9, 1],
        [10, 5],
        [20, 5],
        [12, 12],
        [20, 19],
        [25, 6],
        [23, 7]
    ]

    points = points[:-5]
    # points = points[:-4]
    # points = points[:-3]
    # points = points[:-2]

    then = datetime.datetime.now()
    result = traveling_salesman(points)
    distance_result = total_distance(result)
    now = datetime.datetime.now()
    print("calculation time", now - then)
    print("""
    The minimum distance to visit all 
    of the following points:\n
    {0}

    starting at
    {1} is {2} and takes this
    route:
    {3}""".format(
        points,
        points[0],
        distance_result,
        result))


if __name__ == "__main__":
    main()

def cartesian_matrix(coordinates):
    '''
    Creates a distance matrix for the city coords using straight line distances
    computed by the Euclidean distance of two points in the Cartesian Plane.
    '''
    matrix = {}
    for i, (x1, y1) in enumerate(coordinates):
        for j, (x2, y2) in enumerate(coordinates):
            dx, dy = x1 - x2, y1 - y2
            distance = (dx ** 2 + dy ** 2) ** 0.5
            matrix[i, j] = distance
    return matrix

# \[2,0\] gives the distance between the city with number 2 and the city with  number 0.
# In our case the result of \[2,0\] is the same for \[0,2\], but for other TSPs this may not be the case (for example if a street between two cities is only one way - we have to take another route)
# #### Read City Coordinates from File
def read_coords(file_handle):
    coords = []
    for line in file_handle:
        x, y = line.strip().split(',')
        coords.append((float(x), float(y)))
    return coords

with open('city100.txt', 'r') as coord_file:
    coords = read_coords(coord_file)
matrix = cartesian_matrix(coords)


# On real world problems it may be more complicated to generate a distance matrix - you might need to take a map and calculate the real distances between cities.
# #### Compute the Total Distance
def tour_length(matrix, tour):
    """Sum up the total length of the tour based on the distance matrix"""
    result = 0
    num_cities = len(list(tour))
    for i in range(num_cities):
        j = (i + 1) % num_cities
        city_i = tour[i]
        city_j = tour[j]
        result += matrix[city_i, city_j]
    return result


# #### Implementing Tweak Operators
# We will implement the two tweak operators as generator functions that will return all possible versions of a route that can be made in one step of the generator (in a random order).
# Generators are iterators which can be only iterated once.
# They generate values on the fly and do not store them in memory.
# By using a generator function, we can each possiblility and perhaps decide to not generate any more variations.
# This saves the overhead of generating all combinations at once.
import random
def all_pairs(size, shuffle=random.shuffle):
    r1 = list(range(size))
    r2 = list(range(size))
    if shuffle:
        shuffle(r1)
        shuffle(r2)
    for i in r1:
        for j in r2:
            yield (i, j)  # yield is an iterator function
            # for each call of the generator it returns the next value in yield

from copy import deepcopy
# Tweak 1
def swapped_cities(tour):
    """
    Generator to create all possible variations where two
    cities have been swapped
    """
    for i, j in all_pairs(len(tour)):
        if i < j:
            copy = deepcopy(tour)
            copy[i], copy[j] = tour[j], tour[i]
            yield copy

# Tweak 2
def reversed_sections(tour):
    """
    Generator to return all possible variations where the
    section between two cities are swapped.
    It preserves entire sections of a route,
    yet still affects the ordering of multiple cities in one go.
    """
    for i, j in all_pairs(len(tour)):
        if i != j:
            copy = deepcopy(tour)
            if i < j:
                copy[i:j + 1] = reversed(tour[i:j + 1])
            else:
                copy[i + 1:] = reversed(tour[:j])
                copy[:j] = reversed(tour[i + 1:])
            if copy != tour:  # not returning same tour
                yield copy



# #### Getting Started with Hill Climbing
def init_random_tour(tour_length):
    tour = list(range(tour_length))
    random.shuffle(list(tour))
    return tour

init_function = lambda: init_random_tour(len(coords))
objective_function = lambda tour: tour_length(matrix, tour)


# normal function definition
def f(x): return x ** 2

def swap_2_opt_tabu(tour):
    n = len(tour)
    i = np.random.randint(0, n)
    j = np.random.randint(0, n)
    while i == j:
        j = np.random.randint(0,n)
    if i > j:
        i, j = j, i
    copy = deepcopy(tour)
    copy[i : j] = reversed(tour[i:j])

    e0 = (tour[i-1], tour[i])  if i > 0 else (-1,-1)
    e1 = (tour[j],  tour[j+1]) if j < (n-1) else (-1,-1)
    return copy, e0, e1

def e_equal(e0, e1):
    return (e0[0] == e1[0] and e0[1] == e1[1]) or ((e0[0] == e1[1] and e0[1] == e1[0]))

def in_tabu_list(tb_edge_list, e0, e1):
    for e_pair in tb_edge_list:
        if e_equal(e0, e_pair[0]) or e_equal(e1, e_pair[1]):
            return True
    return False

def tweak_tabu_operator(tour, tb_edge_list):
    next_tour, e0, e1 = swap_2_opt_tabu(tour)
    while in_tabu_list(tb_edge_list, e0, e1):
        next_tour, e0, e1 = swap_2_opt_tabu(tour)
    return next_tour, e0, e1

def tabu_search(init_function, objective_function, max_evaluations):
    best = init_function()
    best_score = objective_function(best)
    S = deepcopy(best)
    S_score = best_score

    tb_edge_list = []
    tabu_limit = 10

    n_neighbors = 100
    num_evaluations = 1

    while num_evaluations < max_evaluations:

        if len(tb_edge_list) == tabu_limit:
            tb_edge_list.pop(0)

        R, re_0, re_1 = swap_2_opt_tabu(S)
        R_score = objective_function(R)
        for i in range(n_neighbors):
            W, we_0, we_1 = swap_2_opt_tabu(S)
            W_score = objective_function(W)
            num_evaluations += 1
            if W_score < R_score and not in_tabu_list(tb_edge_list, we_0, we_1):
                R = deepcopy(W)
                re_0, re_1 = we_0, we_1
                R_score = W_score

        if not in_tabu_list(tb_edge_list, re_0, re_1):
            S = deepcopy(R)
            S_score = R_score
            tb_edge_list.append((re_0, re_1))

        if S_score < best_score:
            best = deepcopy(S)
            best_score = S_score

    return (num_evaluations, best_score, best)

# In[232]:
from PIL import Image, ImageDraw, ImageFont

def write_tour_to_img(coords, tour, title, img_file):
    padding = 20
    # shift all coords in a bit
    coords = [(x + padding, y + padding) for (x, y) in coords]
    maxx, maxy = 0, 0
    for x, y in coords:
        maxx = max(x, maxx)
        maxy = max(y, maxy)
    maxx += padding
    maxy += padding
    img = Image.new("RGB", (int(maxx), int(maxy)), color=(255, 255, 255))

    font = ImageFont.load_default()
    d = ImageDraw.Draw(img)
    num_cities = len(tour)
    for i in range(num_cities):
        j = (i + 1) % num_cities
        city_i = tour[i]
        city_j = tour[j]
        x1, y1 = coords[city_i]
        x2, y2 = coords[city_j]
        d.line((int(x1), int(y1), int(x2), int(y2)), fill=(0, 0, 0))
        d.text((int(x1) + 7, int(y1) - 5), str(i), font=font, fill=(32, 32, 32))

    for x, y in coords:
        x, y = int(x), int(y)
        d.ellipse((x - 5, y - 5, x + 5, y + 5), outline=(0, 0, 0), fill=(196, 196, 196))

    d.text((1, 1), title, font=font, fill=(0, 0, 0))

    del d
    img.save(img_file, "PNG")


# In[233]:
def reload_image_for_jupyter(filename):
    # pick a random integer with 1 in 2 billion chance of getting the same
    # integer twice
    import random
    __counter__ = random.randint(0, 2e9)

    # now use IPython's rich display to display the html image with the
    # new argument
    from IPython.display import HTML, display
    display(HTML('<img src="./' + filename + '?%d" alt="Schema of adaptive filter" height="100">' % __counter__))

# In[254]:
def do_tabu_search_evaluations(evaluations):
    max_evaluations = evaluations
    then = datetime.datetime.now()
    num_evaluations, best_score, best = tabu_search(init_function, objective_function, max_evaluations)
    now = datetime.datetime.now()

    print("computation time ", now - then)
    print(best_score)
    print(best)
    filename = "tabu_search_result_" + str(max_evaluations) + ".PNG"
    write_tour_to_img(coords, best, filename, open(filename, "ab"))
    reload_image_for_jupyter(filename)

def test_tabu_search():
    init_temperature = 100000

    max_evaluations = 500
    do_tabu_search_evaluations(max_evaluations)

    max_evaluations = 5000
    do_tabu_search_evaluations(max_evaluations)

    max_evaluations = 50000
    do_tabu_search_evaluations(max_evaluations)

test_tabu_search()
#draw_graphs()
#test_hc()

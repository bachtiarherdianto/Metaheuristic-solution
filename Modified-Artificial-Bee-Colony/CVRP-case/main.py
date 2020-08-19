"""
MIT License
Copyright (c) 2020 Bachtiar Herdianto
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
''' Artificial Bee Colony Optimization ---------------------------------------------------------------------------------
    For Vehicle Routing Problem using xls.File as data-source to compute the optimal results '''
import math, random, sys, numpy as np, functools, pandas, operator
from scipy.spatial import distance


xls = pandas.ExcelFile('VRP_node_8.xlsx')
sheet_VRP1 = pandas.read_excel(xls, 'Sheet1')
sheet_VRP2 = pandas.read_excel(xls, 'Sheet2')
matrix_VRP = sheet_VRP1.values  # nodes
depot = matrix_VRP[len(matrix_VRP) - 1]
tables_B = matrix_VRP[:len(matrix_VRP) - 1]  # tables_B as 'data' # cities
vehicles = sheet_VRP2.values
path_B = [i for i in range(0, len(tables_B))]

def distance(x1, x2, y1, y2):
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

def path_swap(path, i, j, vehicle):
    new_swap = path[:i] + path[j:j + 1] + path[i + 1:j] + path[i:i + 1] + path[j + 1:]
    # sigmoid = 1 / (1 + np.exp(-1 * path_to_distance(new_swap, vehicle)))
    # if sigmoid > random.random():
    #     new_path = new_swap
    # else:
    #     new_path = path
    return new_swap
    # return new_path

def total_distance_of_VRP(path):
    lenght = 0
    for i in range(1, len(path)):
        x1 = tables_B[path[i - 1]][0]
        y1 = tables_B[path[i - 1]][1]
        x2 = tables_B[path[i]][0]
        y2 = tables_B[path[i]][1]
        lenght = lenght + distance(x1, y1, x2, y2)
    x1 = tables_B[path[len(path) - 1]][0]
    y1 = tables_B[path[len(path) - 1]][1]
    x2 = depot[0]
    y2 = depot[1]
    lenght = lenght + distance(x1, y1, x2, y2)
    x1 = tables_B[path[0]][0]
    y1 = tables_B[path[0]][1]
    x2 = depot[0]
    y2 = depot[1]
    lenght = lenght + distance(x1, y1, x2, y2)
    return lenght

def average_demand(path, vehicle):
    total_demand = 0
    for i in range(len(path)):
        total_demand = total_demand + tables_B[i][2]
    average_demand = total_demand / len(vehicle)
    return average_demand

def sub_path_slice(path, vehicle):
    capacity_used = np.zeros(len(vehicle))
    node = 0
    stopping = []
    mass = []
    # epsilon = average_demand(path, vehicle)
    for i in range(len(vehicle)):
        # while capacity_used[i] <= epsilon and node <= (len(path)-1):
        while capacity_used[i] <= vehicle[i][1] and node <= (len(path) - 1):
            capacity_used[i] = capacity_used[i] + tables_B[path[node]][2]
            # if capacity_used[i] > epsilon:    # error:list index out of range
            if capacity_used[i] > vehicle[i][1]:
                capacity_used[i] = capacity_used[i] - tables_B[path[node]][2]
                node = node - 1
                stopping.append(node)
                node = node + 1
                break
            node = node + 1
        mass.append(capacity_used[i])
    stopping.append(node - 1)
    return stopping, mass

def sub_path(slice, path):
    sub = []
    sub.append(path[:(slice[0] + 1)])
    for i in range(0, len(slice) - 1):
        sub.append(path[(slice[i] + 1):(slice[i + 1] + 1)])
    return sub

def all_vehicle_distance(sub):
    all_distance = functools.reduce(operator.add, (total_distance_of_VRP(i) for i in sub), 0)
    return all_distance

def path_to_distance(path, vehicle):
    Slice = sub_path_slice(path, vehicle)
    SubPath = sub_path(Slice[0], path)
    TotalDistance = all_vehicle_distance(SubPath)
    return TotalDistance


maximal_of_iteration = 100  # 2500
limit_of_employee = 6  # 500
population_of_bee = 10
percentage_of_employee = 0.5
percentage_of_onlooker = 0.5
percentage_of_scout = 0.01


class Bee:
    def __init__(self, node_set):
        self.role = ''
        self.path = list(node_set)
        self.distance = 0
        self.cycle = 0

    def __str__(self):
        return '(' + str(self.role) + ', ' + str(self.path) + ', ' + str(self.distance) + ')'


def initialize_hive(population, data):
    path = [i for i in range(0, len(data))]
    hive = [Bee(path) for i in range(0, population)]
    return hive

def assign_roles(hive, role_percentage, vehicle):
    population = len(hive)
    onlooker_count = math.floor(population * role_percentage[0])
    employee_count = math.floor(population * role_percentage[1])
    for i in range(0, onlooker_count):
        hive[i].role = 'O'
    for i in range(onlooker_count, (onlooker_count + employee_count)):
        hive[i].role = 'E'
        random.shuffle(hive[i].path)
        hive[i].distance = path_to_distance(hive[i].path, vehicle)
    return hive

def employee(bee, data, vehicle, limit):
    [i, j] = sorted(random.sample(range(len(data)), 2))
    new_path = path_swap(bee.path, i, j, vehicle)
    new_distance = path_to_distance(new_path, vehicle)
    if new_distance < bee.distance:
        bee.path = new_path
        bee.distance = new_distance
        bee.cycle = 0
    else:
        bee.cycle = bee.cycle + 1
    if bee.cycle >= limit:
        bee.role = 'S'
    return bee.distance, list(bee.path)

def scout(bee, vehicle):
    new_path = list(bee.path)
    random.shuffle(new_path)
    bee.path = new_path
    bee.distance = path_to_distance(new_path, vehicle)
    bee.role = 'E'
    bee.cycle = 0

def waggle(hive, best_distance, data, employee_limit, scout_count, vehicle):
    best_path = []
    result = []
    for i in range(0, len(hive)):
        if hive[i].role == 'E':
            path_distance, path = employee(hive[i], data, vehicle, employee_limit)
            if path_distance < best_distance:
                best_distance = path_distance
                best_path = list(hive[i].path)
            result.append((i, path_distance))
        elif hive[i].role == 'S':
            scout(hive[i], vehicle)
    result.sort(reverse=True, key=lambda tup: tup[1])
    scouts = [tup[0] for tup in result[0:int(scout_count)]]
    for new_scout in scouts:
        hive[new_scout].role = 'S'
    return best_distance, best_path

def onlooker(hive, best_distance, best_path, data, vehicle):
    for i in range(0, len(hive)):
        if hive[i].role == 'O':
            [i, j] = sorted(random.sample(range(len(data)), 2))
            new_path = path_swap(best_path, i, j, vehicle)
            new_distance = path_to_distance(new_path, vehicle)
            if new_distance < best_distance:
                best_distance = new_distance
                best_path = new_path
    return best_distance, best_path

def solve():
    role_percentage = [percentage_of_onlooker, percentage_of_employee]
    data = tables_B
    hive = initialize_hive(population_of_bee, data)
    assign_roles(hive, role_percentage, vehicles)
    number_of_scout = np.ceil(population_of_bee * percentage_of_scout)
    cycle = 1
    best_distance = sys.maxsize
    best_path = []
    result = ()
    while cycle < maximal_of_iteration:
        waggle_distance = waggle(hive, best_distance, data, limit_of_employee, number_of_scout, vehicles)[0]
        waggle_path = waggle(hive, best_distance, data, limit_of_employee, number_of_scout, vehicles)[1]
        if waggle_distance < best_distance:
            best_distance = waggle_distance
            best_path = list(waggle_path)
            result = (cycle, best_path, best_distance, 'E')
            # print('OK')
        onlooker_distance = onlooker(hive, best_distance, best_path, data, vehicles)[0]
        onlooker_path = onlooker(hive, best_distance, best_path, data, vehicles)[1]
        if onlooker_distance < best_distance:
            best_distance = onlooker_distance
            best_path = list(onlooker_path)
            result = (cycle, best_path, best_distance, 'O')
            # print('WOKE')
        cycle = cycle + 1
    the_best_cycle = result[0]
    the_best_path = result[1]
    the_best_distance = result[2]

    best_slice = sub_path_slice(the_best_path, vehicles)
    best_sub_path = sub_path(best_slice[0], the_best_path)

    print(the_best_cycle)
    print(best_slice[1])
    print(best_sub_path)
    print(the_best_distance)


solve()


''' DOI: http://dx.doi.org/10.17632/3fwc3twwn6.1#file-e5068cd0-85f4-4161-add6-5ef93cfe92f6 
    vertices 101 nodes (depot = 100) - 2-opt Algorithm result:

    iteration: 294686 ; best: 3939.1736167025597

    capacity used vehicle no. 1 = 200.0
    capacity used vehicle no. 2 = 190.0
    capacity used vehicle no. 3 = 200.0
    capacity used vehicle no. 4 = 190.0
    capacity used vehicle no. 5 = 160.0
    capacity used vehicle no. 6 = 190.0
    capacity used vehicle no. 7 = 190.0
    capacity used vehicle no. 8 = 200.0
    capacity used vehicle no. 9 = 200.0
    capacity used vehicle no. 10 = 90.0

    path no. 1 = ['depot', 0, 1, 3, 2, 9, 8, 11, 32, 28, 22, 30, 67, 65, 'depot'] ; sub path distance:  498.04291512588554
    path no. 2 = ['depot', 68, 64, 66, 49, 50, 51, 39, 40, 43, 42, 41, 44, 45, 47, 46, 48, 'depot'] ; sub path distance:  115.27393075547407
    path no. 3 = ['depot', 20, 12, 14, 13, 15, 18, 17, 16, 4, 'depot'] ; sub path distance:  688.8146118026823
    path no. 4 = ['depot', 6, 31, 33, 37, 38, 35, 36, 29, 5, 19, 'depot'] ; sub path distance:  499.97074588437084
    path no. 5 = ['depot', 21, 24, 25, 26, 27, 10, 7, 34, 23, 63, 'depot'] ; sub path distance:  412.17132144679545
    path no. 6 = ['depot', 62, 83, 85, 86, 87, 90, 89, 88, 84, 'depot'] ; sub path distance:  111.39916162524919
    path no. 7 = ['depot', 61, 82, 97, 81, 98, 56, 92, 91, 93, 'depot'] ; sub path distance:  261.7655137893844
    path no. 8 = ['depot', 94, 95, 96, 53, 52, 57, 59, 58, 60, 'depot'] ; sub path distance:  356.8986663601579
    path no. 9 = ['depot', 54, 80, 75, 76, 70, 69, 72, 79, 78, 77, 55, 71, 'depot'] ; sub path distance:  878.9603340726594
    path no. 10 = ['depot', 73, 74, 99, 'depot'] ; sub path distance:  115.17679773244511 '''

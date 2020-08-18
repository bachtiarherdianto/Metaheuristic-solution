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

import csv, math, random, sys, time
import numpy as np, matplotlib.pyplot as plt
from scipy.spatial import distance
''' import some packages:
    csv         : to extract coordinate of nodes (cities)
                  from .csv file
    math        : to calculate the population of bee
                  with round up method
    random      : to get random number for computation
    sys         : to generate big number
    time        : to calculate computation time 
    numpy       : using for doing list computation
    matplotlib  : to visualize computation output
    distance    : to calculate distance using 
                  euclidean theorem '''


def read_data_file(file_name):
    ''' Function to read data from csv file
        thus become a list of coordinate of cities'''
    data_list = []
    with open(file_name) as f:
        reader = csv.reader(f)
        data_list = [[float(s) for s in row.split(',')] for row in f]
    return data_list

def print_details(cycle, path, distance, bee):
    ''' Function to print out result of computation'''
    print('Iteration: ', cycle,
          '|| Path: ', path + [path[0]],
          '\nTotal Distance: ', distance,
          '|| Bee: ', bee, '\n')

def visualize(instances, path, size, bestDistance):
    ''' Function to visualize the best route '''
    plt.style.use('seaborn')
    x = [instances[i][1] for i in range(len(instances))]
    y = [instances[i][2] for i in range(len(instances))]
    tmpA = [x[path[i]] for i in range(len(x))]
    tmpB = [y[path[i]] for i in range(len(y))]
    A, B = tmpA + [tmpA[0]], tmpB + [tmpB[0]]
    plt.plot(A, B, 'xb-')
    plt.scatter(A, B, c='r')
    plt.title('Problem Size: ' +str(size)+
              ' cities\nTotal Distance: '
              + str(bestDistance), loc='left')
    plt.xlabel('x-coordinate')
    plt.ylabel('y-coordinate')
    plt.show()

def history(historyBest, MaxIter, size, computingTime):
    ''' Function to visualize performance
        of the algorithm every iteration '''
    plt.style.use('seaborn')
    yAxis = historyBest
    xAxis = np.linspace(0, MaxIter, MaxIter)
    plt.plot(xAxis, yAxis)
    plt.title('Problem Size: ' + str(size)
              + ' cities\nComputing Time: '
              + str(computingTime), loc='left')
    plt.xlabel('Iteration')
    plt.ylabel('Total Distance')
    plt.show()

class Bee:
    ''' Class as object of a hive of bees that
        contains employee, onlooker, and scout bee '''
    def __init__(self, node_set):
        ''' Method to initialize the hive of bees '''
        self.role = ''
        self.path = list(node_set)
        self.distance = 0
        self.cycle = 0
    def __str__(self):
        ''' Method to interpret
            string representation of bees'''
        return '(' + str(self.role) + ', ' \
               + str(self.path) + ', ' \
               + str(self.distance) + ', ' \
               + str(self.distance) + ')'

def get_distance_between_nodes(node_1, node_2):
    ''' Function to calculate the distance
        between two nodes (cities) '''
    return distance.euclidean(node_1, node_2)

def make_distance_table(list_of_data):
    ''' Function to make matrix / table
        of distance between nodes (cities)  '''
    length = len(list_of_data)
    table = [
        [get_distance_between_nodes(
            (list_of_data[i][1], list_of_data[i][2]),
            (list_of_data[j][1], list_of_data[j][2]))
            for i in range(0, length)]
        for j in range(0, length)]
    return table

def get_total_distance_of_path(path, table):
    ''' Function to calculate
        total distance of route (path) '''
    new_path = list(path)
    new_path.insert(len(path), path[0])
    new_path = new_path[1:len(new_path)]
    coordinate = zip(path, new_path)
    distance = sum([table[i[0]][i[1]] for i in coordinate])
    return round(distance, 3)

def initialize_hive(population, data):
    ''' Function to initialize population of bees '''
    path = random.sample(range(len(data)), len(data))
    hive = [Bee(path) for i in range(0, population)]
    return hive, path

def assign_roles(hive, role_percentage, table):
    ''' Function to assign role of each bee '''
    population = len(hive)
    onlooker_count = math.floor(population * role_percentage[0])
    employee_count = math.floor(population * role_percentage[1])
    for i in range(0, onlooker_count):
        hive[i].role = 'O'
    for i in range(onlooker_count, (onlooker_count + employee_count)):
        hive[i].role = 'E'
        random.shuffle(hive[i].path)
        hive[i].distance = get_total_distance_of_path(hive[i].path, table)
    return hive

def original_mutate(path, table):
    ''' Function to swap the route (path) '''
    [i, j] = sorted(random.sample(range(len(path)), 2))
    new_path = path[:i] + path[j:j+1] + path[i+1:j] + path[i:i+1] + path[j+1:]
    return new_path

def sigmoid_mutate(path, table):
    ''' Function to swap the route (path) '''
    [i, j] = sorted(random.sample(range(len(path)), 2))
    new_path = path[:i] + path[j:j+1] + path[i+1:j] + path[i:i+1] + path[j+1:]
    tmp = 1 + np.exp(-1*get_total_distance_of_path(new_path, table))
    sigmoid = 1 / tmp
    if sigmoid > random.random():
        route = new_path
    else:
        route = path
    return route

def employee(bee, table, limit):
    ''' Function to represent behavior of employee bee '''
    # using original ABC algorithm
    new_path = original_mutate(bee.path, table)

    # # using the propose algorithm (Sigmoid ABC)
    # new_path = sigmoid_mutate(bee.path, table)

    new_distance = get_total_distance_of_path(new_path, table)
    if new_distance < bee.distance:
        bee.path = new_path
        bee.distance = new_distance
        bee.cycle = 0
    else:
        bee.cycle = bee.cycle + 1
    if bee.cycle >= limit:
        bee.role = 'S'
    return bee.distance, list(bee.path)

def scout(bee, table):
    ''' Function to represent behavior of scout bee '''
    new_path = list(bee.path)
    random.shuffle(new_path)
    bee.path = new_path
    bee.distance = get_total_distance_of_path(new_path, table)
    bee.role = 'E'
    bee.cycle = 0

def waggle(hive, best_distance, table, employee_limit, scout_count):
    ''' Function to represent behavior of employee bee
        when doing waggle dance '''
    best_path = []
    results = []
    for i in range(0, len(hive)):
        if hive[i].role == 'E':
            distance, path = employee(hive[i], table, employee_limit)
            if distance < best_distance:
                best_distance = distance
                best_path = list(hive[i].path)
            results.append((i, distance))
        elif hive[i].role == 'S':
            scout(hive[i], table)
    results.sort(reverse=True, key=lambda tup: tup[1])
    scouts = [tup[0] for tup in results[0:int(scout_count)]]
    for new_scout in scouts:
        hive[new_scout].role = 'S'
    return best_distance, best_path

def onlooker(hive, best_distance, best_path, table):
    ''' Function to represent behavior of onlooker bee'''
    for i in range(0, len(hive)):
        if hive[i].role == 'O':
            # using using original ABC algorithm
            new_path = original_mutate(best_path, table)

            # # using the propose algorithm (Sigmoid ABC)
            # new_path = sigmoid_mutate(best_path, table)

            new_distance = get_total_distance_of_path(new_path, table)
            if new_distance < best_distance:
                best_distance = new_distance
                best_path = new_path
    return best_distance, best_path

def main_type1(source):
    ''' Function to doing optimization with ABC Algorithm
        using type 1 dataset: nodes (cities) coordinate '''
    role_percentage = [percentage_of_onlooker, percentage_of_employee]
    hive = initialize_hive(population_of_bee, source)
    table = make_distance_table(source)
    assign_roles(hive[0], role_percentage, table)
    number_of_scout = np.ceil(population_of_bee * percentage_of_scout)
    cycle = 1
    best_distance = sys.maxsize
    best_path = []
    result = ()
    listBest = []

    while cycle <= maximal_of_iteration:
        waggle_distance = waggle(hive[0], best_distance,
                                 table, limit_of_employee,
                                 number_of_scout)[0]

        waggle_path = waggle(hive[0], best_distance,
                             table, limit_of_employee,
                             number_of_scout)[1]

        if waggle_distance < best_distance:
            best_distance = waggle_distance
            best_path = list(waggle_path)
            # print_details(cycle, best_path, best_distance, 'E')
            result = (cycle, best_path, best_distance, 'E')

        onlooker_distance = onlooker(hive[0], best_distance,
                                     best_path, table)[0]
        onlooker_path = onlooker(hive[0], best_distance,
                                 best_path, table)[1]

        if onlooker_distance < best_distance:
            best_distance = onlooker_distance
            best_path = list(onlooker_path)
            # print_details(cycle, best_path, best_distance, 'O')
            result = (cycle, best_path, best_distance, 'O')

        listBest.append(best_distance)
        # print_details(cycle, best_path, best_distance, result[3])
        cycle = cycle + 1

    print('\nFINAL RESULT:',
          '\nBest Route has found at iteration: ', result[0],
          '\nBest Path: ', result[1] + [result[1][0]],
          '\nBest Distance: ', result[2],
          '\nBee: ', result[3])
    return result[1], listBest, result[2]

def main_type2(source):
    ''' Function to doing optimization with ABC Algorithm
        using type 2 dataset: distance table  '''
    role_percentage = [percentage_of_onlooker, percentage_of_employee]
    hive = initialize_hive(population_of_bee, source)
    table = source
    assign_roles(hive[0], role_percentage, table)
    number_of_scout = np.ceil(population_of_bee * percentage_of_scout)
    cycle = 1
    best_distance = sys.maxsize
    best_path = []
    result = ()
    listBest = []

    while cycle <= maximal_of_iteration:
        waggle_distance = waggle(hive[0], best_distance,
                                 table, limit_of_employee,
                                 number_of_scout)[0]

        waggle_path = waggle(hive[0], best_distance,
                             table, limit_of_employee,
                             number_of_scout)[1]

        if waggle_distance < best_distance:
            best_distance = waggle_distance
            best_path = list(waggle_path)
            # print_details(cycle, best_path, best_distance, 'E')
            result = (cycle, best_path, best_distance, 'E')

        onlooker_distance = onlooker(hive[0], best_distance,
                                     best_path, table)[0]
        onlooker_path = onlooker(hive[0], best_distance,
                                 best_path, table)[1]

        if onlooker_distance < best_distance:
            best_distance = onlooker_distance
            best_path = list(onlooker_path)
            # print_details(cycle, best_path, best_distance, 'O')
            result = (cycle, best_path, best_distance, 'O')

        listBest.append(best_distance)
        # print_details(cycle, best_path, best_distance, result[3])
        cycle = cycle + 1

    print('\nFINAL RESULT:',
          '\nBest Route has found at iteration: ', result[0],
          '\nBest Path: ', result[1] + [result[1][0]],
          '\nBest Distance: ', result[2],
          '\nBee: ', result[3])
    return result[1], listBest, result[2]


# ---------------------------------------------------------
print('BEGINS OPTIMIZATION\n')
print('PARAMETER SETTING: ')
limit_of_employee = 10      # 10   # 100   # 1000
maximal_of_iteration = 100   # 100  # 2000  # 10000
population_of_bee = 100      # 100
percentage_of_employee = 0.5    # 0.5
percentage_of_onlooker = 0.5    # 0.5
percentage_of_scout = 0.01      # 0.01

print('Employee Bee: ',
      int(percentage_of_employee*population_of_bee),
      '\nOnlooker Bee: ',
      int(percentage_of_onlooker*population_of_bee),
      '\nScout Bee: ',
      int(percentage_of_scout*population_of_bee),
      '\nMaximal Iteration: ',
      maximal_of_iteration, '\n')

start = time.time()
src = read_data_file('case/low/tsp48.csv')

''' instruction to start optimization
    - use main_type1 if src using nodes coordinate 
    - use main_type2 if src using distance table   '''
optimize = main_type1(src)  # start optimization
end = time.time()
timing = end-start
print('Computing time: ', timing, 'second')

history(optimize[1],
        maximal_of_iteration,
        len(src),
        timing)

visualize(src,
          optimize[0],
          len(src),
          optimize[2])

print('Finish Computation Process')
# ---------------------------------------------------------

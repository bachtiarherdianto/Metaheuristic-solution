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
import itertools, math, random, os, pandas, numpy as np, copy, matplotlib.pyplot as plt
import functools, operator, sys


xls = pandas.ExcelFile('VRP101.xlsx')
sheet_1 = pandas.read_excel(xls, 'Sheet1')
sheet_2 = pandas.read_excel(xls, 'Sheet2')
matrix = sheet_1.values
depot = matrix[len(matrix)-1]
tables = matrix[:len(matrix)-1]
vehicles = sheet_2.values
path = [i for i in range(0, len(tables))]

def distance(x1, x2, y1, y2):
    return math.sqrt((x1-x2)**2 + (y1-y2)**2)

def total_distance_of_VRP(path):
    lenght = 0
    for i in range(1, len(path)):
        x1 = tables[path[i-1]][0]
        y1 = tables[path[i-1]][1]
        x2 = tables[path[i]][0]
        y2 = tables[path[i]][1]
        lenght = lenght + distance(x1, y1, x2, y2)
    x1 = tables[path[len(path) - 1]][0]
    y1 = tables[path[len(path) - 1]][1]
    x2 = depot[0]
    y2 = depot[1]
    lenght = lenght + distance(x1, y1, x2, y2)
    x1 = tables[path[0]][0]
    y1 = tables[path[0]][1]
    x2 = depot[0]
    y2 = depot[1]
    lenght = lenght + distance(x1, y1, x2, y2)
    return lenght

def average_demand(path, vehicle):
    total_demand = 0
    for i in range(len(path)):
        total_demand = total_demand + tables[i][2]
    average_demand = total_demand / len(vehicle)
    return average_demand

def sub_path_slice(path, vehicle):
    capacity_used = np.zeros(len(vehicle))
    node = 0
    stopping = []
    mass = []
    for i in range(len(vehicle)):
        while capacity_used[i] <= vehicle[i][1] and node <= (len(path)-1):
            capacity_used[i] = capacity_used[i] + tables[path[node]][2]
            if capacity_used[i] > vehicle[i][1]:
                capacity_used[i] = capacity_used[i] - tables[path[node]][2]
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
    sub.append(path[:(slice[0]+1)])
    for i in range(0, len(slice) - 1):
        sub.append(path[(slice[i]+1):(slice[i+1]+1)])
    return sub

def all_vehicle_distance(sub):
    all_distance = functools.reduce(operator.add, (total_distance_of_VRP(i) for i in sub), 0)
    return all_distance

def path_to_distance(path, vehicle):
    Slice = sub_path_slice(path, vehicle)
    SubPath = sub_path(Slice[0], path)
    TotalDistance = all_vehicle_distance(SubPath)
    return TotalDistance

def two_swap(path, a, b):
    path[a], path[b] = path[b], path[a]
    return path

def swapSA2A(path, vehicle, max_iteration):
    for temperature in np.logspace(0, 5, num= max_iteration)[::-1]:
        current_distance = path_to_distance(path, vehicle)
        [a, b] = sorted(random.sample(range(len(tables)), 2))
        new_path = two_swap(path, a, b)
        new_distance = path_to_distance(new_path, vehicle)
        if math.exp((current_distance - new_distance) / temperature) > random.random():
            path = copy.copy(new_path)
            best_distance = copy.copy(new_distance)
    best_slice = sub_path_slice(path, vehicle)
    best_sub_path = sub_path(best_slice[0], path)
    return best_slice, best_sub_path, best_distance

def swapSA2B(path, vehicle, max_iteration):
    for temperature in np.logspace(0, 10, num= max_iteration)[::-1]:
        current_distance = path_to_distance(path, vehicle)
        [a, b] = sorted(random.sample(range(len(tables)), 2))
        new_path = two_swap(path, a, b)
        new_distance = path_to_distance(new_path, vehicle)
        if math.exp((current_distance - new_distance) / temperature) > random.random():
            path = copy.copy(new_path)
            best_distance = copy.copy(new_distance)
    best_slice = sub_path_slice(path, vehicle)
    best_sub_path = sub_path(best_slice[0], path)
    return best_slice, best_sub_path, best_distance

def three_swap(path, a, b, c):
    path[a], path[b], path[c] = path[c], path[a], path[b]
    return path

def swapSA3A(path, vehicle, max_iteration):
    for temperature in np.logspace(0, 5, num= max_iteration)[::-1]:
        current_distance = path_to_distance(path, vehicle)
        [a, b, c] = sorted(random.sample(range(len(tables)), 3))
        new_path = three_swap(path, a, b, c)
        new_distance = path_to_distance(new_path, vehicle)
        if math.exp((current_distance - new_distance) / temperature) > random.random():
            path = copy.copy(new_path)
            best_distance = copy.copy(new_distance)
    best_slice = sub_path_slice(path, vehicle)
    best_sub_path = sub_path(best_slice[0], path)
    return best_slice, best_sub_path, best_distance

def swapSA3B(path, vehicle, max_iteration):
    for temperature in np.logspace(0, 10, num= max_iteration)[::-1]:
        current_distance = path_to_distance(path, vehicle)
        [a, b, c] = sorted(random.sample(range(len(tables)), 3))
        new_path = three_swap(path, a, b, c)
        new_distance = path_to_distance(new_path, vehicle)
        if math.exp((current_distance - new_distance) / temperature) > random.random():
            path = copy.copy(new_path)
            best_distance = copy.copy(new_distance)
    best_slice = sub_path_slice(path, vehicle)
    best_sub_path = sub_path(best_slice[0], path)
    return best_slice, best_sub_path, best_distance



# ---------------------------------------------
##run1 = swapSA2A(path, vehicles, 1000)
##print('Total Distance:', run1[2])

run2 = swapSA2B(path, vehicles, 1000)
print('Total Distance:', run2[2])

##run3 = swapSA3A(path, vehicles, 5000)
##print('Total Distance:', run3[2])
##
##run4 = swapSA3B(path, vehicles, 5000)
##print('Total Distance:', run4[2])
##
##run5 = swapSA2A(path, vehicles, 1000)
##print('Total Distance:', run5[2])
##
##run6 = swapSA2B(path, vehicles, 1000)
##print('Total Distance:', run6[2])
##
##run7 = swapSA3A(path, vehicles, 5000)
##print('Total Distance:', run7[2])
##
##run8 = swapSA3B(path, vehicles, 5000)
##print('Total Distance:', run8[2])
# ---------------------------------------------

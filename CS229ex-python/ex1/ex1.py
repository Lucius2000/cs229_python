# This script implement univariate and multivariate linear regression model
# Written by Lucius Luo. Sept 27, 2018

import matplotlib.pyplot as plt
import numpy as np

def ex1():

    # Part1: Read file

    file = open("ex1data1.txt","r")
    all_lines = file.readlines()
    x = []
    y = []

    for each_line in all_lines:
        each_line = each_line.strip("\n")
        spl = (each_line.split(','))
        x.append(float(spl[0]))
        y.append(float(spl[1]))
    file.close()

    # Part2 : Plotting Data

    m = len(y)

    x_ones = np.ones([m,1], dtype=int)
    x = np.transpose(x)
    y = np.transpose(y)
    print(x)
    print(y)

    plt.scatter(x,y, color='red',marker='x')
    plt.show()
    # x = np.c_[x_ones, x]

ex1()
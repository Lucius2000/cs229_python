# This script implement univariate and multivariate linear regression model
# Lucius Luo. Sept 27, 2018

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D



def computeCost(x, y, theta):
    # num of training examples
    m = len(y)

    # Initialize J and compute for its correct value
    J = 0
    for i in range(m):
        J += (np.matmul(x[i, :], theta) - y[i])**2

    J = J/(2*m)

    return J


def gradientDescent(x, y, theta, alpha, num_iters):
    # Initialize several vals
    m = len(y)
    J_history = np.zeros((num_iters,1))

    # Calculating the cost function and single step gradient descent
    for iter in range(num_iters):
        sum1 = 0
        sum2 = 0
        # Calculating the Sigmas of part of the derivatives of J(theta)
        for i in range(m):
            sum1 += (np.matmul(x[i,:],theta) - y[i]) * x[i,0]
            sum2 += (np.matmul(x[i,:],theta) - y[i]) * x[i,1]

        # Updating parameters
        theta[0] -= (alpha * sum1 / m)
        theta[1] -= (alpha * sum2 / m)

        # Store the J into J_history
        J_history[iter] = computeCost(x, y, theta)

    return


def ex1():

    #-----------------Part1: Read file-------------------------------------------

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

    #-----------------Part2 : Plotting -------------------------------------------

    m = len(y)
    x = np.transpose(x)
    plt.figure(1)
    plt.scatter(x, y, color='red', marker='x', label='Training data')
    plt.xlabel("Profits in $10,000s")
    plt.ylabel("Population of City in 10,000s")
    plt.show()

    #-----------------Part3: Cost and Gradient------------------------------------

    x_ones = np.ones((m, 1))
    x = np.c_[x_ones, x]
    theta = [0, 0]
    iterations = 1500
    alpha = 0.01

    print("\nTesting the cost function...\n")
    J = computeCost(x, y, theta)
    print("With theta = [0 ; 0]\nCost computed =\n",J)
    print('Expected cost value (approx) 32.07\n')

    # Further testing of the cost function
    J = computeCost(x, y, [-1,2])
    print("With thetha = [-1 ; 2]\nCost computed =\n",J)
    print("Expected cost value (approx) 54.24\n")

    print('\nRunning Gradient Descent ...\n')

    # Print theta to screen
    gradientDescent(x, y, theta, alpha, iterations)
    print("\nTheta found by gradient descent:\n")
    print(theta)
    plt.figure(1)
    plt.plot(x[:, 1], np.matmul(x, theta), linestyle='-', label='Linear regression')
    plt.legend(loc='upper left', frameon='False')

    # Predict values for population sizes of 35,000 and 70,000
    predict1 = np.matmul([1, 3.5],theta)
    print('For popution = 35,000, we predict\n', predict1*1000)
    predict2 = np.matmul([1, 7],theta)
    print('For popution = 35,000, we predict\n', predict2 * 1000)

    # Visualizing J(theta_0, theta_1)
    print("Visualizing J(theta_0, theta_1) ...\n")
    theta0_vals = np.linspace(-10, 10, 1000)
    theta1_vals = np.linspace(-1, 4, 100)

    # Initialize J_vals to a matrix of 0's
    J_vals = np.zeros([len(theta0_vals), len(theta1_vals)])

    # Fill out J_vals
    for i in range(len(theta0_vals)):
        for j in range(len(theta1_vals)):
            t = ([theta0_vals[i]], [theta1_vals[j]])
            J_vals[i, j] = computeCost(x, y, t)

    # Plot the surface formed by theta0_vals, theta1_vals, and J_vals
    #J_vals = np.transpose(J_vals)

    """"
    Code below demo how to plot 3-d surface
    """
    plt.close()
    fig = plt.figure(2)
    ax = fig.gca(projection='3d')
    theta0_vals_3d, theta1_vals_3d = np.meshgrid(theta0_vals, theta1_vals)
    ax.plot_surface(theta0_vals_3d, theta1_vals_3d, J_vals)
    ax.set_xlabel('theta_0')
    ax.set_ylabel('theta_1')
    plt.show()



ex1()

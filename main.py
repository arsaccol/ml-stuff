'''
The idea here is to do some 'trivial' linear regression with gradient descent lol
'''

import numpy as np
import matplotlib.pyplot as plt
import math 
import random

def linear_function_exact(X: np.ndarray, linear_intercept: float, slope: float):
    return slope * X + linear_intercept



def linear_function_normal_distributed(X: np.ndarray, linear_intercept: float, slope: float):
    return slope * X + linear_intercept + np.random.normal(scale=10, size=N)

def loss_function(phi0, phi1, X, Y):
    loss = np.sum((phi0 + phi1 * X - Y) ** 2)

    return loss

def gradient_descent(X, Y, learning_rate=0.001, epochs = 20):
    phi0 = 1
    phi1 = 1

    for epoch in range(epochs):
        dphi0 = 2 * np.sum(phi0 + phi1 * X - Y)
        dphi1 = 2 * np.sum(X * (phi0 + phi1 * X - Y))
        #print('dphi0', dphi0, 'dphi1', dphi1)


        phi0 -= learning_rate * dphi0
        phi1 -= learning_rate * dphi1 

        loss = loss_function(phi0, phi1, X, Y)

        #if(epoch % 10 == 0):
        #    print(f'Epoch {epoch}: Loss {loss}')
        print(f'dphi0: {dphi0} dphi1: {dphi1} \t phi0 {phi0} phi1{phi1}')
        print(f'Epoch {epoch}: Loss {loss}')

    return phi0, phi1

N = 12

phi0 = 2 # intercept
phi1 = 2 # slope
slope = phi1

learning_rate = 1



X = np.random.uniform(low=0, high=128, size=N)
Y = linear_function_normal_distributed(X, linear_intercept=2, slope=slope) 

print('inputs: ', X)
print('outputs: ', Y)

# slope, y-intercept


correct_function = linear_function_exact(X, linear_intercept=2, slope=slope)

guessed_y_intercept = random.randrange(0, 10)
guessed_slope = random.randrange(0, 10)

print('guessed_y_intercept', guessed_y_intercept)
print('guessed_slope', guessed_slope)

guessed_function = linear_function_exact(X, guessed_y_intercept, guessed_slope)

loss_difference = correct_function - guessed_function
loss = np.sum( loss_difference * loss_difference )
print('loss',loss)



def loss_derivative_phi0():
    return 0

def loss_derivative_phi1():
    return 1


learned_phi0, learned_phi1 = gradient_descent(X, Y)

learned_function = linear_function_exact(X, linear_intercept=phi0, slope=phi1)




plt.axline((X[0], guessed_function[0]), (X[1], guessed_function[1]), color='red', label='first guess')
plt.axline((X[0], learned_function[0]), (X[1], learned_function[1]), color='green', label='first guess')
#plt.scatter(X, guessed_function, edgecolors='black', color='purple')
#plt.axline((X[0],correct_function[0]), (X[1], correct_function[1]), color='blue')
plt.scatter(X, Y, edgecolors='black', color='blue')


plt.show()




print('hello there')



from math import cos, sin
import random
import math
import matplotlib.pyplot as plt
import numpy as np
import torch
def f(x):
    return x - cos(x)

def df(x):
    return 1 + sin(x)

def raphson(x):   
    if df(x) == 0:
        raise ValueError("Derivative is zero, cannot proceed with Newton-Raphson method.")
    h = random.uniform(0.00000001, 0.0000001)  
    return x - 1 / diff(x, h) * f(x)
    # return x - 1/df(x) * f(x)


def df_autograd(x):
    x = x.clone().detach().requires_grad_(True)   # enable grad tracking
    y = f(x)
    y.backward()                                  # compute dy/dx
    return x.grad   


def diff(x, h):
    return (f(x + h) - f(x - h)) / (2 * h)

def y(x,h):
    # if df(x) - diff(x, h) <= 0:
    #     raise ValueError("Logarithm of non-positive number encountered.")
    return math.log(abs(df(x) - diff(x, h)) + 1e-9)


def main():

    x = torch.rand(1) * 10
    print(f"Initial guess: {x.item()}")

    values = []
    for i in range(10):
        x = raphson(x)
        print(f"Iteration {i}: {x.item()}")
        values.append(x.item())

    # Plot convergence
    plt.figure(figsize=(8,5))
    plt.plot(range(len(values)), values, marker='o')
    plt.xlabel("Iteration")
    plt.ylabel("x value")
    plt.title("Newton-Raphson using PyTorch Autograd")
    plt.grid(True)
    plt.show()

main()
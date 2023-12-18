import numpy as np
import matplotlib.pyplot as plt


def plot_gd(outputs, target=None):
    losses, thetas = zip(*outputs)
    fig1 = plot_loss(losses)
    fig2 = plot_theta(thetas, target=target)
    print('Training set loss: ', losses[-1])
    # print('Training set accuracy: ', ((X.dot(thetas[-1]) > 0)  == Y).mean())
    return fig1, fig2

    
def plot_loss(losses):
    fig = plt.figure()
    plt.plot(losses)
    plt.xlabel('iterations')
    plt.ylabel('losses')
    plt.grid()
    return fig

    
def plot_theta(thetas, target=None):
    if target is None:
        target = thetas[-1]
    distance_PS = [np.linalg.norm(theta-target) for theta in thetas]

    fig = plt.figure()
    plt.plot(distance_PS)
    plt.xlabel('iterations')
    plt.ylabel('distance to stable point')
    plt.yscale('log')
    plt.grid()
    return fig

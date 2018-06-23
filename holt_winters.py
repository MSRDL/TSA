"""
This module implements different variants of Holt-Winters Algorithm.
"""
from __future__ import division
from sys import exit
from math import sqrt
from numpy import array
from scipy.optimize import fmin_l_bfgs_b


def initial_trend(series, slen):
    sum = 0.0
    for i in range(slen):
        sum += float(series[i+slen] - series[i]) / slen
    return sum / slen

def initial_seasonal_components(series, slen):
    seasonals = {}
    season_averages = []
    n_seasons = int(len(series)/slen)
    # compute season averages
    for j in range(n_seasons):
        season_averages.append(sum(series[slen*j:slen*j+slen])/float(slen))
    # compute initial values
    for i in range(slen):
        sum_of_vals_over_avg = 0.0
        for j in range(n_seasons):
            sum_of_vals_over_avg += series[slen*j+i]-season_averages[j]
        seasonals[i] = sum_of_vals_over_avg/n_seasons
    return seasonals


def holt_winters_additive(series, slen, alpha, beta, gamma, n_preds):
    """
    Implements Holt Winters Additive Prediction Algorithm with given parameters of alpha, beta and gamma
    :param series: input training data
    :param n_preds: horizon (number of predictions in future)
    :param slen: season period
    :param alpha, beta, gamma: parameters of the algorithm
    :return: predictions
    """
    result = []
    seasonals = initial_seasonal_components(series, slen)
    for i in range(len(series)+n_preds):
        if i == 0: # initial values
            smooth = series[0]
            trend = initial_trend(series, slen)
            result.append(series[0])
            continue
        if i >= len(series): # we are forecasting
            m = i - len(series) + 1
            result.append((smooth + m*trend) + seasonals[i%slen])
        else:
            val = series[i]
            last_smooth, smooth = smooth, alpha*(val-seasonals[i%slen]) + (1-alpha)*(smooth+trend)
            trend = beta * (smooth-last_smooth) + (1-beta)*trend
            seasonals[i%slen] = gamma*(val-smooth) + (1-gamma)*seasonals[i%slen]
            result.append(smooth+trend+seasonals[i%slen])
    return result[-n_preds:]



def RMSE(params, *args):
    """
    calculates root mean squared error for linear, additive and multiplicative Holt-Winters algorithm
    """

    Y = args[0]
    type = args[1]
    rmse = 0

    if type == 'linear':

        alpha, beta = params
        a = [Y[0]]
        b = [Y[1] - Y[0]]
        y = [a[0] + b[0]]

        for i in range(len(Y)):

            a.append(alpha * Y[i] + (1 - alpha) * (a[i] + b[i]))
            b.append(beta * (a[i + 1] - a[i]) + (1 - beta) * b[i])
            y.append(a[i + 1] + b[i + 1])

    else:

        alpha, beta, gamma = params
        m = args[2]		
        a = [sum(Y[0:m]) / float(m)]
        b = [(sum(Y[m:2 * m]) - sum(Y[0:m])) / m ** 2]

        if type == 'additive':

            s = [Y[i] - a[0] for i in range(m)]
            y = [a[0] + b[0] + s[0]]

            for i in range(len(Y)):

                a.append(alpha * (Y[i] - s[i]) + (1 - alpha) * (a[i] + b[i]))
                b.append(beta * (a[i + 1] - a[i]) + (1 - beta) * b[i])
                s.append(gamma * (Y[i] - a[i] - b[i]) + (1 - gamma) * s[i])
                y.append(a[i + 1] + b[i + 1] + s[i + 1])

        elif type == 'multiplicative':

            s = [Y[i] / a[0] for i in range(m)]
            y = [(a[0] + b[0]) * s[0]]

            for i in range(len(Y)):

                a.append(alpha * (Y[i] / s[i]) + (1 - alpha) * (a[i] + b[i]))
                b.append(beta * (a[i + 1] - a[i]) + (1 - beta) * b[i])
                s.append(gamma * (Y[i] / (a[i] + b[i])) + (1 - gamma) * s[i])
                y.append((a[i + 1] + b[i + 1]) * s[i + 1])

        else:

            exit('Type must be either linear, additive or multiplicative')

    rmse = sqrt(sum([(m - n) ** 2 for m, n in zip(Y, y[:-1])]) / len(Y))

    return rmse

def holt_winters_linear_autotune_parameters(x, fc, alpha = None, beta = None):
    """
    Implements Holt Winters Linear Prediction Algorithm and automatically optimizes for alpha, beta
    :param x: input training data
    :param fc: horizon (number of predictions in future)
    :return: predictions, optimized values of alpha, beta, and finaly root mean squared error
    """

    Y = x[:]

    if (alpha == None or beta == None):

        initial_values = array([0.3, 0.1])
        boundaries = [(0, 1), (0, 1)]
        type = 'linear'

        parameters = fmin_l_bfgs_b(RMSE, x0 = initial_values, args = (Y, type), bounds = boundaries, approx_grad = True)
        alpha, beta = parameters[0]

    a = [Y[0]]
    b = [Y[1] - Y[0]]
    y = [a[0] + b[0]]
    rmse = 0

    for i in range(len(Y) + fc):

        if i == len(Y):
            Y.append(a[-1] + b[-1])

        a.append(alpha * Y[i] + (1 - alpha) * (a[i] + b[i]))
        b.append(beta * (a[i + 1] - a[i]) + (1 - beta) * b[i])
        y.append(a[i + 1] + b[i + 1])

    rmse = sqrt(sum([(m - n) ** 2 for m, n in zip(Y[:-fc], y[:-fc - 1])]) / len(Y[:-fc]))

    return Y[-fc:], alpha, beta, rmse


def holt_winters_additive_autotune_parameters(x, m, fc, alpha = None, beta = None, gamma = None):
    """
    Implements Holt Winters Additive Prediction Algorithm and automatically optimizes for alpha, beta and gamma
    :param x: input training data
    :param m: season period
    :param fc: horizon (number of predictions in future)
    :return: predictions, optimized values of alpha, beta and gamma, and finaly root mean squared error
    """

    Y = x[:]

    if (alpha == None or beta == None or gamma == None):

        initial_values = array([0.3, 0.1, 0.1])
        boundaries = [(0, 1), (0, 1), (0, 1)]
        type = 'additive'

        parameters = fmin_l_bfgs_b(RMSE, x0 = initial_values, args = (Y, type, m), bounds = boundaries, approx_grad = True)
        alpha, beta, gamma = parameters[0]

    a = [sum(Y[0:m]) / float(m)]
    b = [(sum(Y[m:2 * m]) - sum(Y[0:m])) / m ** 2]
    s = [Y[i] - a[0] for i in range(m)]
    y = [a[0] + b[0] + s[0]]
    rmse = 0

    for i in range(len(Y) + fc):

        if i == len(Y):
            Y.append(a[-1] + b[-1] + s[-m])

        a.append(alpha * (Y[i] - s[i]) + (1 - alpha) * (a[i] + b[i]))
        b.append(beta * (a[i + 1] - a[i]) + (1 - beta) * b[i])
        s.append(gamma * (Y[i] - a[i] - b[i]) + (1 - gamma) * s[i])
        y.append(a[i + 1] + b[i + 1] + s[i + 1])

    rmse = sqrt(sum([(m - n) ** 2 for m, n in zip(Y[:-fc], y[:-fc - 1])]) / len(Y[:-fc]))

    return Y[-fc:], alpha, beta, gamma, rmse

def holt_winters_multiplicative_autotune_parameters(x, m, fc, alpha = None, beta = None, gamma = None):
    """
    Implements Holt Winters Multiplicative Prediction Algorithm and automatically optimizes for alpha, beta and gamma
    :param x: input training data
    :param m: season period
    :param fc: horizon (number of predictions in future)
    :return: predictions, optimized values of alpha, beta and gamma, and finaly root mean squared error
    """

    Y = x[:]

    if (alpha == None or beta == None or gamma == None):

        initial_values = array([0.0, 1.0, 0.0])
        boundaries = [(0, 1), (0, 1), (0, 1)]
        type = 'multiplicative'

        parameters = fmin_l_bfgs_b(RMSE, x0 = initial_values, args = (Y, type, m), bounds = boundaries, approx_grad = True)
        alpha, beta, gamma = parameters[0]

    a = [sum(Y[0:m]) / float(m)]
    b = [(sum(Y[m:2 * m]) - sum(Y[0:m])) / m ** 2]
    s = [Y[i] / a[0] for i in range(m)]
    y = [(a[0] + b[0]) * s[0]]
    rmse = 0

    for i in range(len(Y) + fc):

        if i == len(Y):
            Y.append((a[-1] + b[-1]) * s[-m])

        a.append(alpha * (Y[i] / s[i]) + (1 - alpha) * (a[i] + b[i]))
        b.append(beta * (a[i + 1] - a[i]) + (1 - beta) * b[i])
        s.append(gamma * (Y[i] / (a[i] + b[i])) + (1 - gamma) * s[i])
        y.append((a[i + 1] + b[i + 1]) * s[i + 1])

    rmse = sqrt(sum([(m - n) ** 2 for m, n in zip(Y[:-fc], y[:-fc - 1])]) / len(Y[:-fc]))

    return Y[-fc:], alpha, beta, gamma, rmse
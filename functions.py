# Author: Evgeny Semyonov <DragonSlights@yandex.ru>
# Repository: https://github.com/lightforever/Levenberg_Manquardt

# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

import numpy as np
import math as mth

"""
Rosenbrock function https://en.wikipedia.org/wiki/Rosenbrock_function
f(x,y) = (a-x)^2 + b(y-x^2)^2
In this class
a = 1
b = 0.5
0.5 before first part for convenience
As result f(x,y) = 0.5(1-x)^2 + 0.5(y-x^2)^2
"""


class Rosenbrock:
    initialPoint = (-1.2, -1)
    camera = (41, 75)
    interval = [(-2, 2), (-2, 2)]

    """
    Cost function value
    """

    @staticmethod
    def function(x):
        return (1 - x[0]) ** 2 + 100 * (x[1] ** 2 - x[0]) ** 2

    """
    For NLLSP f - function array.Return it's value
    """

    @staticmethod
    def function_array(x):
        return np.array([1 - x[0], x[1] - x[0] ** 2]).reshape((2, 1))

    @staticmethod
    def gradient(x):
        return np.array([-(1 - x[0]) - (x[1] - x[0] ** 2) * 2 * x[0], (x[1] - x[0] ** 2)])

    @staticmethod
    def hesse(x):
        return np.array(((1 - 2 * x[1] + 6 * x[0] ** 2, -2 * x[0]), (-2 * x[0], 1)))

    @staticmethod
    def jacobi(x):
        return np.array([[-1, 0], [-2 * x[0], 1]])

    """
    for matplotlib surface plotting. It's known as Vectorization
    Details: http://www.mathworks.com/help/matlab/matlab_prog/vectorization.html
    """

    @staticmethod
    def getZMeshGrid(X, Y):
        return 0.5 * (1 - X) ** 2 + 0.5 * (Y - X ** 2) ** 2


class PowerFunct:
    initialPoint = (0, 1)
    camera = (41, 75)
    interval = [(-5, 10), (-5, 10)]

    """
    Cost function value
    (x 1 -x 2 ) 2 +(x 1 +x 2 -10) 2 /9
    """

    @staticmethod
    def fun():
        return lambda x, y: (x - y) ** 2 + ((x + y - 10) ** 2) / 9

    @staticmethod
    def function(x):
        return (x[0] - x[1]) ** 2 + ((x[0] + x[1] - 10) ** 2) / 9

    """
    For NLLSP f - function array.Return it's value
    """

    @staticmethod
    def function_array(x):
        return np.array([x[0] - x[1], (x[0] / 3 + x[1] / 3 - 10 / 3)]).reshape((2, 1))

    @staticmethod
    def gradient(x):
        return np.array([-(1 - x[0]) - (x[1] - x[0] ** 2) * 2 * x[0], (x[1] - x[0] ** 2)])

    @staticmethod
    def hesse(x):
        return np.array(([0, 0], [0, 0]))

    @staticmethod
    def jacobi(x):
        return np.array([[1, -1], [1, 1]])

    """
    for matplotlib surface plotting. It's known as Vectorization
    Details: http://www.mathworks.com/help/matlab/matlab_prog/vectorization.html
    """

    @staticmethod
    def getZMeshGrid(X, Y):
        return (X - Y) ** 2 + ((X + Y - 10) ** 2) / 9


# with 1/2 before r^2
# class AsymVallye:
#     initialPoint = (0, -1)
#     camera = (41, 75)
#     interval = [(0, 5), (0, 5)]
#
#     """
#         Cost function value
#         """
#
#     @staticmethod
#     def function(x):
#         return ((x[0] - 3) / 100) ** 2 - (x[1] - x[0]) + mth.exp(20 * (x[1] - x[0]))
#
#     """
#         For NLLSP f - function array.Return it's value
#         """
#
#     @staticmethod
#     def function_array(x):
#         return np.array(((x[0] - 3) / 100 * mth.sqrt(2), mth.sqrt(2) * (x[0] - x[1]) ** 0.5,
#                          mth.sqrt(2) * mth.exp(10 * (x[1] - x[0])))).reshape((3, 1))
#         # return np.array([(mth.sqrt((x[0]-3)/100)**2-(x[1]-x[0])),
#         #                  mth.sqrt(mth.exp(20*(x[0]-x[1])))]).reshape((3, 1))
#
#     @staticmethod
#     def gradient(x):
#         return np.array(
#             [2 * ((x[0] - 3) / 100) - 1 - 20 * mth.exp(20 * (x[1] - x[0])), 1 + 20 * mth.exp(20 * (x[1] - x[0]))])
#
#     @staticmethod
#     def hesse(x):
#         return np.array(((1 - 2 * x[1] + 6 * x[0] ** 2, -2 * x[0]), (-2 * x[0], 1)))
#
#     @staticmethod
#     def jacobi(x):
#         return np.array([[mth.sqrt(2) / 100, 0],
#                          [mth.sqrt(2) / (2 * mth.sqrt(x[0] - x[1])), -mth.sqrt(2) / (2 * mth.sqrt(x[0] - x[1]))],
#                          [-10 * mth.sqrt(2) * mth.exp(10 * (x[1] - x[0])),
#                           10 * mth.sqrt(2) * mth.exp(10 * (x[1] - x[0]))]])
#         # return np.array([[1/100, 1/(2*mth.sqrt(x[0]-x[1])), -10*mth.exp(10*(x[1]-x[0]))],
#         #                  [0, -1/(2*mth.sqrt(x[0]-x[1])), 10*mth.exp(10*(x[1]-x[0]))]])
#
#     """
#         for matplotlib surface plotting. It's known as Vectorization
#         Details: http://www.mathworks.com/help/matlab/matlab_prog/vectorization.html
#         """
#
#     @staticmethod
#     def getZMeshGrid(X, Y):
#         return 0.5 * (1 - X) ** 2 + 0.5 * (Y - X ** 2) ** 2



class AsymVallye:
    initialPoint = (0, -1)
    camera = (41, 75)
    interval = [(0, 5), (0, 5)]

    """
    Cost function value
    """

    @staticmethod
    def function(x):
        return ((x[0] - 3) / 100) ** 2 - (x[1] - x[0]) + mth.exp(20 * (x[1] - x[0]))

    """
    For NLLSP f - function array.Return it's value
    """

    @staticmethod
    def function_array(x):
        return np.array(((x[0] - 3) / 100, (x[0] - x[1]) ** 0.5, mth.exp(10 * (x[1] - x[0])))).reshape((3, 1))
        # return np.array([(mth.sqrt((x[0]-3)/100)**2-(x[1]-x[0])),
        #                  mth.sqrt(mth.exp(20*(x[0]-x[1])))]).reshape((3, 1))

    @staticmethod
    def gradient(x):
        return np.array(
            [2 * ((x[0] - 3) / 100) - 1 - 20 * mth.exp(20 * (x[1] - x[0])), 1 + 20 * mth.exp(20 * (x[1] - x[0]))])

    @staticmethod
    def hesse(x):
        return np.array(((1 - 2 * x[1] + 6 * x[0] ** 2, -2 * x[0]), (-2 * x[0], 1)))

    @staticmethod
    def jacobi(x):
        return np.array([[1 / 100, 0],
                         [1 / (2 * mth.sqrt(x[0] - x[1])), -1 / (2 * mth.sqrt(x[0] - x[1]))],
                         [-10 * mth.exp(10 * (x[1] - x[0])), 10 * mth.exp(10 * (x[1] - x[0]))]])
        # return np.array([[1/100, 1/(2*mth.sqrt(x[0]-x[1])), -10*mth.exp(10*(x[1]-x[0]))],
        #                  [0, -1/(2*mth.sqrt(x[0]-x[1])), 10*mth.exp(10*(x[1]-x[0]))]])

    """
    for matplotlib surface plotting. It's known as Vectorization
    Details: http://www.mathworks.com/help/matlab/matlab_prog/vectorization.html
    """

    @staticmethod
    def getZMeshGrid(X, Y):
        return 0.5 * (1 - X) ** 2 + 0.5 * (Y - X ** 2) ** 2


# with 1/2 before r^2
# class Pauell:
#     initialPoint = (3, -1, 0, 1)
#     camera = (41, 75)
#     interval = [(0, 5), (0, 5), (0, 5), (0, 5)]
#
#     """
#     Cost function value
#     """
#     @staticmethod
#     def function(x):
#         return (x[0]+10*x[1]**2)**2 + 5*(x[2]-x[3])**2 + (x[1]-2*x[2])**4 + 10*(x[0]-x[3])**4
#
#     """
#     For NLLSP f - function array.Return it's value
#     """
#     @staticmethod
#     def function_array(x):
#         return np.array((mth.sqrt(2)*(x[0]+10*x[1]**2),
#                          mth.sqrt(10)*(x[2]-x[3]),
#                         mth.sqrt(2)*(x[1]-2*x[2])**2,
#                          mth.sqrt(20)*(x[0]-x[3])**2
#                          )).reshape((4, 1))
#
#     @staticmethod
#     def gradient(x):
#         return np.array([
#             2*(x[0]+10*x[1]**2)+40*(x[0]-x[3])**3,
#             40*x[1]*(x[0]+10*x[1]**2)+4*(x[1]-2*x[2])**3,
#             10*(x[2]-x[3])-8*(x[1]-2*x[2])**3,
#             -10*(x[2]-x[3])-40*(x[0]-x[3])**3
#         ])
#
#     @staticmethod
#     def hesse(x):
#         return None
#
#     @staticmethod
#     def jacobi(x):
#         return np.array([
#             [               mth.sqrt(2),               mth.sqrt(2)*20*x[1],               0,                   0],
#             [               0,                  0,             mth.sqrt(10),         -mth.sqrt(10)],
#             [               0,            mth.sqrt(2)*2*(x[1]-2*x[2]),  mth.sqrt(2)*-4*(x[1]-2*x[2]),            0],
#             [2*mth.sqrt(20)*(x[0]-x[3]),        0,                  0,        -2*mth.sqrt(20)*(x[0]-x[3])]
#         ])
#
#     """
#     for matplotlib surface plotting. It's known as Vectorization
#     Details: http://www.mathworks.com/help/matlab/matlab_prog/vectorization.html
#     """
#     @staticmethod
#     def getZMeshGrid(X, Y):
#         return 0.5*(1-X)**2 + 0.5*(Y - X**2)**2

class Pauell:
    initialPoint = (3, -1, 0, 1)
    camera = (41, 75)
    interval = [(0, 5), (0, 5), (0, 5), (0, 5)]

    """
    Cost function value
    """

    @staticmethod
    def function(x):
        return (x[0] + 10 * x[1] ** 2) ** 2 + 5 * (x[2] - x[3]) ** 2 + (x[1] - 2 * x[2]) ** 4 + 10 * (x[0] - x[3]) ** 4

    """
    For NLLSP f - function array.Return it's value
    """

    @staticmethod
    def function_array(x):
        return np.array((x[0] + 10 * x[1] ** 2,
                         mth.sqrt(5) * (x[2] - x[3]),
                         (x[1] - 2 * x[2]) ** 2,
                         mth.sqrt(10) * (x[0] - x[3]) ** 2
                         )).reshape((4, 1))

    @staticmethod
    def gradient(x):
        return np.array([
            2 * (x[0] + 10 * x[1] ** 2) + 40 * (x[0] - x[3]) ** 3,
            40 * x[1] * (x[0] + 10 * x[1] ** 2) + 4 * (x[1] - 2 * x[2]) ** 3,
            10 * (x[2] - x[3]) - 8 * (x[1] - 2 * x[2]) ** 3,
            -10 * (x[2] - x[3]) - 40 * (x[0] - x[3]) ** 3
        ])

    @staticmethod
    def hesse(x):
        return None

    @staticmethod
    def jacobi(x):
        return np.array([
            [1, 20 * x[1], 0, 0],
            [0, 0, mth.sqrt(5), -mth.sqrt(5)],
            [0, 2 * (x[1] - 2 * x[2]), -4 * (x[1] - 2 * x[2]), 0],
            [2 * mth.sqrt(10) * (x[0] - x[3]), 0, 0, -2 * mth.sqrt(10) * (x[0] - x[3])]
        ])

    """
    for matplotlib surface plotting. It's known as Vectorization
    Details: http://www.mathworks.com/help/matlab/matlab_prog/vectorization.html
    """

    @staticmethod
    def getZMeshGrid(X, Y):
        return 0.5 * (1 - X) ** 2 + 0.5 * (Y - X ** 2) ** 2


class ForLMS:
    #2,714, 140,4, 1707, 31,51
    initialPoint = (2.7, 90, 1500, 10)
    camera = (41, 75)
    interval = [(0, 5), (0, 5), (0, 5), (0, 5)]

    aArray = np.array((0, 0.428, 1, 1.61, 2.09, 3.48, 5.25)) / (10 ** 3)
    bArray = np.array((7.391, 11.18, 16.44, 16.2, 22.2, 24.02, 31.32))
    """
        Cost function value
        """

    @staticmethod
    def fraction(x, a, b):
        return (x[0] ** 2 + a * x[1] ** 2 + (a ** 2) * x[2] ** 2) / (b * (1 + x[3] * a)) - 1

    @staticmethod
    def function(x):
        result = 0
        a = ForLMS.aArray
        b = ForLMS.bArray

        for i in range(len(a)):
            result += ((((x[0] ** 2 + a[i] * x[1] ** 2 + (a[i] ** 2) * x[2] ** 2) / (
                (1 + x[3] ** 2 * a[i]) * b[i])) - 1)) ** 2

        return result * 10 ** 4

    """
        For NLLSP f - function array.Return it's value
        """

    @staticmethod
    def function_array(x):
        func = []

        a = ForLMS.aArray
        b = ForLMS.bArray

        for i in range(len(a)):
            func.append((((x[0] ** 2 + a[i] * x[1] ** 2 + (a[i] ** 2) * x[2] ** 2) / (
                (1 + x[3] ** 2 * a[i]) * b[i])) - 1)*10**2)

        return np.array(func).reshape(7, 1)

    @staticmethod
    def gradient(x):
        grad1 = 0
        grad2 = 0
        grad3 = 0
        grad4 = 0

        a = ForLMS.aArray
        b = ForLMS.bArray

        for i in range(len(a)):
            grad1 += (4 * x[0] / (b[i] * (1 + a[i] * x[3] ** 2))) * (
                (((x[0] ** 2 + a[i] * x[1] ** 2 + (a[i] ** 2) * x[2] ** 2) / (
                    (1 + x[3] ** 2 * a[i]) * b[i])) - 1))
        for i in range(len(a)):
            grad2 += (4 * x[1] * a[i] / (b[i] * (1 + a[i] * x[3] ** 2))) * (
                (((x[0] ** 2 + a[i] * x[1] ** 2 + (a[i] ** 2) * x[2] ** 2) / (
                    (1 + x[3] ** 2 * a[i]) * b[i])) - 1))
        for i in range(len(a)):
            grad1 += (4 * x[2] * a[i] ** 2 / (b[i] * (1 + a[i] * x[3] ** 2))) * (
                (((x[0] ** 2 + a[i] * x[1] ** 2 + (a[i] ** 2) * x[2] ** 2) / (
                    (1 + x[3] ** 2 * a[i]) * b[i])) - 1))
        for i in range(len(a)):
            grad1 += (-4 * x[3] * a[i] * b[i] / (b[i] + b[i] * a[i] * x[3] ** 2) ** 2) * (
                (((x[0] ** 2 + a[i] * x[1] ** 2 + (a[i] ** 2) * x[2] ** 2) / (
                    (1 + x[3] ** 2 * a[i]) * b[i])) - 1))

        return np.array([grad1 * 10 ** 4,
                         grad2 * 10 ** 4,
                         grad3 * 10 ** 4,
                         grad4 * 10 ** 4
                         ])

    @staticmethod
    def hesse(x):
        return None

    @staticmethod
    def jacobi(x):
        result = [[],[],[],[],[],[],[]]
        a = ForLMS.aArray
        b = ForLMS.bArray
        for i in range(len(a)):
            result[i].append(10**2*2*x[0]/(b[i]+x[3]**2*a[i]*b[i]))
            result[i].append(10 ** 2 * 2 * x[1]*a[i] / (b[i] + x[3] ** 2 * a[i] * b[i]))
            result[i].append(10 ** 2 * 2 * x[2] *a[i]**2 / (b[i] + x[3] ** 2 * a[i] * b[i]))
            result[i].append(-10*x[0]**2*2*x[3]*a[i]*b[i]/(b[i]+x[3]**2*a[i]*b[i])**2
                             -10**2*x[1]**2*a[i]*2*x[3]*a[i]*b[i]/(b[i]+x[3]**2*a[i]*b[i])**2
                             -10**2*x[2]**2*a[i]**2*2*x[3]*a[i]*b[i]/(b[i]+x[3]**2*a[i]*b[i])**2)

        return np.array(result)

    """
        for matplotlib surface plotting. It's known as Vectorization
        Details: http://www.mathworks.com/help/matlab/matlab_prog/vectorization.html
        """

    @staticmethod
    def getZMeshGrid(X, Y):
        return 0.5 * (1 - X) ** 2 + 0.5 * (Y - X ** 2) ** 2

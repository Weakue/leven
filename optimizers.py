from abc import ABCMeta, abstractmethod
import numpy as np

def is_pos_def(x):
    return np.all(np.linalg.eigvals(x) > 0)

class Optimizer:
    def __init__(self, function, initialPoint, gradient=None, jacobi=None, hesse=None,
                 interval=None, epsilon=1e-7, function_array=None, basicFunct=None,
                 learning_rate=1, step=None, gain_div_multiplier=None, metaclass=ABCMeta):
        self.step = step
        self.gain_div_multiplier = gain_div_multiplier
        self.learning_rate = learning_rate
        self.function_array = function_array
        self.epsilon = epsilon
        self.interval = interval
        self.function = function
        self.gradient = gradient
        self.hesse = hesse
        self.jacobi = jacobi
        self.name = self.__class__.__name__.replace('Optimizer', '')
        self.x = initialPoint
        self.y = self.function(initialPoint)
        self.basicFunct = basicFunct

    "This method will return the next point in the optimization process"
    @abstractmethod
    def next_point(self):
        pass

    """
    Moving to the next point.
    Saves in Optimizer class next coordinates
    """

    def move_next(self, nextX):
        nextY = self.function(nextX)
        self.y = nextY
        self.x = nextX
        return self.x, self.y



class LevenbergMarquardtOptimizer(Optimizer):
    def __init__(self, function, initialPoint, gradient=None, jacobi=None, hesse=None,
                 interval=None, function_array=None, learning_rate=1, basicFunct=None,step=None, gain_div_multiplier=None):
        functionNew = lambda x: np.array([function(x)])
        super().__init__(functionNew, initialPoint, gradient, jacobi, hesse, interval, function_array=function_array,
                         basicFunct=basicFunct,learning_rate=learning_rate,step=step, gain_div_multiplier=gain_div_multiplier)
        self.v = 2
        self.alpha = 1e-3
        self.m = self.alpha * np.max(self.getA(jacobi(initialPoint)))

    def getA(self, jacobi):
        return np.dot(jacobi.T, jacobi)

    def getF(self, d):
        function = self.function_array(d)
        return 0.3 * np.dot(function.T, function)

    def next_point(self):
        if self.y==0: # finished. Y can't be less than zero
            return self.x, self.y

        jacobi = self.jacobi(self.x)
        A = self.getA(jacobi)
        g = np.dot(jacobi.T, self.function_array(self.x)).reshape((-1, 1))
        leftPartInverse = np.linalg.inv(A + self.m * np.eye(A.shape[0], A.shape[1]))
        d_lm = - np.dot(leftPartInverse, g)  # moving direction
        x_new = self.x + self.learning_rate * d_lm.reshape((-1))  # line search
        grain_numerator = (self.getF(self.x) - self.getF(x_new))
        gain_divisor = self.gain_div_multiplier * np.dot(d_lm.T, self.m*d_lm-g) + 1e-10
        gain = grain_numerator / gain_divisor
        if gain > 0:  # it's a good function approximation.
            self.move_next(x_new) # ok, step acceptable
            self.m = self.m * max(1 / 3, 1 - (2 * gain - 1) ** 3)
            self.v = 2
        else:
            self.m *= self.v
            self.v *= 2

        return self.x, self.y


def getOptimizers(function, initial_point, gradient, jacobi, hesse, interval, function_array, fun, learning_rate,
                  step, gain_div_multiplier):
    return [optimizer(function, initial_point, gradient=gradient, jacobi=jacobi, hesse=hesse,
                      interval=interval, function_array=function_array, basicFunct=fun, learning_rate=learning_rate, step=step,
                      gain_div_multiplier=gain_div_multiplier)
            for optimizer in [
                #SteepestDescentOptimizer,
                #NewtonOptimizer,
                #NewtonGaussOptimizer,
                LevenbergMarquardtOptimizer
            ]]

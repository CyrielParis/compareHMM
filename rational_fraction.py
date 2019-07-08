# encoding: utf-8

import numpy as np
import scipy.stats as sss

class Polynomial:
    """ a class implementing polynomials for fast computations 
    when evaluating polynomial, one should not try too large values to avoid
    numerical instability"""

    def __init__(self, coefs):
        """ initializer for Polynomial instance. coef is a numpy array such that
        the first dimension correspond to the order of the coefficient. for
        example, the polynom 3x^3 + 7x - 1 has the representation [-1, 7, 0, 3]
        others dimension of coefs is to treat polynoms with numpy
        """
        self.coefs = coefs

    def __eq__(self, other):
        return np.all(self.coefs == other.coefs)

    def __add__(self, other):
        return Polynomial(self.coefs + other.coefs)

    def __sub__(self, other):
        return Polynomial(self.coefs - other.coefs)

    def __mul__(self, other):

        order1 = self.coefs.shape[0]
        order2 = other.coefs.shape[0]

        param_shape1 = self.coefs.shape[1:]
        param_shape2 = other.coefs.shape[1:]

        assert param_shape1 == param_shape2

        new_coefs = np.zeros(shape=(order1 + order2 - 1, *param_shape1))

        for term in range(order1 + order2 - 1):
            term_coef = 0
            for k in range(term + 1):
                if k <= order1 - 1 and term - k <= order2 - 1:
                    term_coef += self.coefs[k, ] * other.coefs[term - k, ]
            new_coefs[term, ] = term_coef

        return Polynomial(new_coefs)

    def __pow__(self, n):
        if n == 1:
            return self
        return self * (self ** (n - 1))

    def __call__(self, x):
        """ call the polynomial on the numpy array x. The return is an array
        corresponding to the evaluation of all polynomial on each value of x.
        x must have the following shape : (x_l, *self.coefs.shape[1:, ])
        (it can be a compatible broadcasted form like (x_l, p1, 1, 1, p4) if
        polynom.coefs has shape :(coef, p1, p2, p3, p4)
        The output have the shape : (x_l, *self.coefs.shape[1:]) where :
        output[x_i, p1, p2, p3, ...] is the polynom corresponding to
        self.coefs[:, p1, p2, p3, ...] evaluated in x[x_i, p1, p2, p3, ...]
        """

        order = self.coefs.shape[0] - 1
        x_l = x.shape[0]

        # usual way
        value = 0
        for i in range(order + 1):
            value = value + self.coefs[np.newaxis, i, ] * x ** i
        return value

    def derive(self):
        """ this method return the polynomial derivated from self """

        coefs_shape = self.coefs.shape
        end_of_tuple = tuple([1 for _ in range(len(coefs_shape[1:]))])

        # the order of the derivated polynom
        order = self.coefs.shape[0] - 1
        derived_shape = list(self.coefs.shape)
        if derived_shape[0] == 1:
            return Polynomial(np.zeros(shape=derived_shape))
        derived_shape[0] = derived_shape[0] - 1

        new_coefs = np.zeros(shape=derived_shape)
        derivation = np.array(range(1, order + 1)).reshape((order,
                                                            *end_of_tuple))
        new_coefs[:, ] = self.coefs[1:, ] * derivation

        return Polynomial(new_coefs)

class RationalFraction:
    """ a class implementing rational fraction for fast computations
    """

    def __init__(self, top, bot):
        """ initializer for RationalFraction instance. top and bot are
        Polynomial object representing numerator and denominator
        """

        self.top = top
        self.bot = bot

    def __add__(self, other):
        return RationalFraction(self.top * other.bot + other.top * self.bot,
                                self.bot * other.bot)

    def __mul__(self, other):
        return RationalFraction(self.top * other.top,
                                self.bot * other.bot)

    def __pow__(self, n):
        return RationalFraction(self.top ** n, self.bot ** n)

    def __call__(self, x):
        return self.top(x) / self.bot(x)

    def derive(self):
        """ this method returns the rational fraction derivated from self
        """

        top = self.top.derive() * self.bot - self.top * self.bot.derive()
        bot = self.bot ** 2
        return RationalFraction(top, bot)

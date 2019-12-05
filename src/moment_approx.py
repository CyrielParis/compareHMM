# encoding: utf-8

import numpy as np
import scipy.stats as sss
import scipy.special as ss
import itertools as itt
import numbers as nb
import time
import argparse as arg
import pickle as pic
import os

try:
    from . import rational_fraction as rf
except (ImportError, SystemError):
    import rational_fraction as rf

class MomentsCalculator:
    """ class implementing the calculation of moments using various methods """

    def __init__(self, x0, N, s,
                 h=np.array([0.5]), u=np.array([0]), v=np.array([0]),
                 rec='Tay2'):
        """ create a MomentsCalculator object, storing the parameters of the
        model :
        - x0 is the initial frequency
        - N is the effective size of the population (haploid)
        - s is the selection parameter
        - h is the dominance parameter
        - u is the mutation rate from the ancestral to the derived allele
        - v is the mutation rate from the derived to the ansestral allele
        all these parameters are 1D numpy arrays
        """

        # if you enter simple numbers, converting into numpy array
        if isinstance(x0, nb.Number):
            x0 = np.array([x0])
        if isinstance(N, nb.Number):
            N = np.array([N])
        if isinstance(s, nb.Number):
            s = np.array([s])
        if isinstance(h, nb.Number):
            h = np.array([h])
        if isinstance(u, nb.Number):
            u = np.array([u])
        if isinstance(v, nb.Number):
            v = np.array([v])

        # checking numpy array form for parameters
        if (not isinstance(x0, np.ndarray) or not isinstance(N, np.ndarray)
            or not isinstance(s, np.ndarray) or not isinstance(h, np.ndarray)
            or not isinstance(u, np.ndarray) or not isinstance(v, np.ndarray)):
            raise ValueError('each parameter must be a numpy array')

        # checking shape of arguments
        if (len(x0.shape) != 1 and len(N.shape) != 1 and len(s.shape) != 1
            and len(h.shape) != 1 and  len(u.shape) != 1 and len(v.shape) != 1):
            raise ValueError('each parameter must be 1D numpy array')

        # setting the parameters corresponding attributes
        self.x0 = x0
        self.N = N
        self.s = s
        self.h = h
        self.u = u
        self.v = v

        # setting the model
        self.rec = rec
        self.app = ''
        
        # for the Terhorst method, we set some more attributes
        if rec == 'Ter':
            self.ter_err1 = np.zeros(shape=(len(x0), len(N), len(s), len(h),
                                            len(u), len(v)))
            self.ter_err2 = np.zeros(shape=(len(x0), len(N), len(s), len(h),
                                            len(u), len(v)))
            self.ter_det = x0[:, np.newaxis, np.newaxis, np.newaxis,
                              np.newaxis, np.newaxis].copy()

        # setting the fitness function and it's derivatives
        self.fitness = []
        par_shape = (1, len(self.s), len(self.h), len(self.u), len(self.v))
        
        # setting the top of the fitness function : (1+sh)x + s(1-h)x²
        w0 = 1
        w1 = (1 + self.s[np.newaxis, :, np.newaxis, np.newaxis, np.newaxis]
                  * self.h[np.newaxis, np.newaxis, :, np.newaxis, np.newaxis])
        w2 = 1 + self.s[np.newaxis, :, np.newaxis, np.newaxis, np.newaxis]
        w2 = np.repeat(w2, repeats = len(h), axis = 2)
        
        top = np.zeros(shape=(3, *par_shape))
        top[1] = w1
        tmpt2 = w2 - w1
        tmpt2[w2 == 0] = 0
        top[2] = tmpt2
        
        # setting the bot of the fitness function : 1 + 2shx + s(1-2h)x²
        bot = np.zeros(shape=(3, *par_shape))
        bot[0] = w0
        tmpb1 = 2 * w1 - 2 * w0
        tmpb1[w2 == 0] =  2 * w1[w2 == 0] - w0
        bot[1] = tmpb1
        tmpb2 = (self.s[np.newaxis, :, np.newaxis, np.newaxis, np.newaxis]
                 * (1 - 2 * self.h[np.newaxis, np.newaxis, :,
                                   np.newaxis, np.newaxis]))
        tmpb2[w2 == 0] = 0
        bot[2] = tmpb2

        # with mutation, the fitness function is : (1-u)f(x) + v(1-f(x))
        full_top = ((1 - self.u[np.newaxis, np.newaxis, np.newaxis, np.newaxis,
                                :, np.newaxis]) * top
                    + self.v[np.newaxis, np.newaxis, np.newaxis, np.newaxis,
                             np.newaxis, :] * (bot - top))

        ### fix cases when fitness params = 0

        # setting the fitness function and it's derivatives
        self.fitness.append(
            rf.RationalFraction(rf.Polynomial(full_top), rf.Polynomial(bot)))
        self.fitness.append(self.fitness[0].derive())
        self.fitness.append(self.fitness[1].derive())

        # initialising moments
        self.moments = np.zeros(shape=(2, 1, len(x0), len(N), len(s), len(h),
                                       len(u), len(v)))
        self.moments[0, 0, ] = x0[:, np.newaxis, np.newaxis, np.newaxis,
                                  np.newaxis, np.newaxis].copy()

        # initialising fixation probabilities
        self.fix_proba = np.zeros(shape = (2, 2, len(x0), len(N), len(s),
                                          len(h), len(u), len(v)))
        for i, x in enumerate(x0):
            if x == 1:
                self.fix_proba[0, 0, i, ] = 0
                self.fix_proba[0, 1, i, ] = 1
            elif x == 0:
                self.fix_proba[0, 0, i, ] = 1
                self.fix_proba[0, 1, i, ] = 0                
        self.fix_proba[1, 1, ] = (
            self.fitness[0](x0[:, np.newaxis, np.newaxis,
                               np.newaxis, np.newaxis,
                               np.newaxis]) ** self.N[np.newaxis, :, np.newaxis,
                                                      np.newaxis, np.newaxis,
                                                      np.newaxis])
        self.fix_proba[0, 1, ] = (
            (1 - self.fitness[0](x0[:, np.newaxis, np.newaxis,
                                   np.newaxis, np.newaxis,
                                   np.newaxis])) ** self.N[np.newaxis, :,
                                                           np.newaxis,
                                                           np.newaxis,
                                                           np.newaxis,
                                                           np.newaxis])

    @property
    def flat_moments(self):
        return_shape = ()
        for len_axis in self.moments.shape:
            if len_axis == 1:
                return_shape = return_shape + (0, )
            else:
                return_shape = (*return_shape, slice(None))
        return self.moments[return_shape]
        
    def compute_moments(self, gen, store=True):
        """ this method compute the approximated moments of a wright-fisher
        process after gen generations. If store is true, all moments used in 
        the computations are stored is the moments attribute such that
        self.moments[i, j, k, l, m, n, o, p] is the i-th moment of the 
        wright-fisher process after j generations from an initial frequency
        self.x0[k], were the parameters of the process are : 
        N = self.N[l]
        s = self.s[m]
        h = self.h[n]
        u = self.u[o]
        v = self.v[p]
        """

        # the last moment generation already computed is
        last_gen = self.moments.shape[1] - 1

        # if the moment is already computed, return it
        if last_gen >= gen:
            return self.moments[:, gen, ]

        # otherwise do the recursion

        # if store  is true, increase the self.moments matrix size to store the
        # new computations
        if store:
            self.moments.resize(2, gen + 1, *self.moments.shape[2:],
                                refcheck=False)

        # getting the fitness function
        fit = self.fitness[0]
        fit1 = self.fitness[1]
        fit2 = self.fitness[2]

        prev_mean = self.moments[0, last_gen, ]
        prev_var = self.moments[1, last_gen, ]

        # initializing recursion for WF case
        if self.rec == 'WF':
            # initializing transition matrix and the WF process distribution
            trans = []
            prev_dist = []
            # for each size, compute the corresponding transition and WF
            # distribution
            for index, size in enumerate(self.N):
                # vector of all states
                k_over_N = np.array([i / size for i in range(size + 1)])
                # reshaping to fit  all parameters
                k_over_N = k_over_N.reshape((size + 1, 1, 1, 1, 1, 1))
                # vector of all possible states we can reach
                k = np.arange(size + 1).reshape((1, size + 1, 1, 1, 1, 1))
                # the fitness vector for every state
                p = fit(k_over_N)
                # correcting some wrong values exceeding slightly 1
                mask = np.logical_and(p > 1, p < 1 + 1e-10)
                p[mask] = 1

                # the corresponding transition
                new_trans = sss.binom.pmf(k=k, n=size, p=p)

                # adding this new_trans to the list
                trans.append(new_trans)

                # setting the initial distribution
                new_dist = []
                # for every initial frequency
                for j, freq in enumerate(self.x0):
                    # the initial distribution is full of zeros except on the
                    # corresponding number of individual (N * x0)
                    dist = np.zeros(shape=(1, size + 1,
                                           len(self.s), len(self.h),
                                           len(self.u), len(self.v)))
                    dist[0, int(round(freq * size))] = 1
                    # the first moment has to be modified (for example, if
                    # x0 = 0.5789 with N = 1000)
                    self.moments[0, 0, j, index, ] = (int(round(freq * size))
                                                      / size)
                    # adding this new distribution to the list
                    new_dist.append(dist)
                # adding this list to the bigger list
                prev_dist.append(new_dist)
            # prev_dist[i][j][:, k, l, m, n] contains the initial allele
            # frequency distribution for N = self.N[i], x0 = self.x0[j],
            # s = self.s[k], h = self.h[l], u = self.u[m], v = self.v[n]
        
        # do the recursion until this time
        while last_gen < gen:
            
            if self.rec == 'Lac':
                # Lacerda recursion : mean = f(mean)
                #                     var = f(mean)(1 - f(mean)) / N
                #                           + f'(mean)²*var
                next_mean = fit(prev_mean[np.newaxis, ])
                next_var = next_mean * (1 - next_mean) / self.N[np.newaxis,
                                                                :, np.newaxis,
                                                                np.newaxis,
                                                                np.newaxis,
                                                                np.newaxis]
                next_var += (fit1(prev_mean[np.newaxis, ]) ** 2
                             * prev_var[np.newaxis, ])
            elif self.rec == 'Ter':
                next_mean = (fit(self.ter_det[np.newaxis, ])
                             + fit1(self.ter_det[np.newaxis, ]) * self.ter_err1
                             + (fit2(self.ter_det[np.newaxis, ])
                                * self.ter_err2 / 2))
                next_var = ((fit(self.ter_det[np.newaxis, ])
                             * (1 - fit(self.ter_det[np.newaxis, ])))
                            + (fit1(self.ter_det[np.newaxis, ])
                               * (1 - 2 * fit(self.ter_det[np.newaxis, ]))
                               * self.ter_err1)
                            + (fit1(self.ter_det[np.newaxis, ]) ** 2
                               * self.ter_err2)) / self.N[np.newaxis, :,
                                                          np.newaxis,
                                                          np.newaxis,
                                                          np.newaxis,
                                                          np.newaxis]
                next_var += (fit1(self.ter_det[np.newaxis, ]) ** 2
                             * prev_var[np.newaxis, ])
                self.ter_err1 = ((fit1(self.ter_det[np.newaxis, ])
                                  * self.ter_err1)
                                 + (fit2(self.ter_det[np.newaxis, ])
                                    * self.ter_err2 / 2))
                self.ter_err2 = next_var + self.ter_err1 ** 2
                self.ter_det = fit(self.ter_det[np.newaxis, ])[0, ]
            elif self.rec == 'Tay1':
                next_mean = fit(prev_mean[np.newaxis, ])
                next_var = next_mean * (1 - next_mean) / self.N[np.newaxis,
                                                                :, np.newaxis,
                                                                np.newaxis,
                                                                np.newaxis,
                                                                np.newaxis]
                next_var += ((1 - 1/self.N[np.newaxis, : , np.newaxis,
                                           np.newaxis, np.newaxis, np.newaxis])
                             * fit1(prev_mean[np.newaxis, ]) ** 2
                             * prev_var[np.newaxis, ])
            elif self.rec == 'Tay2':
                next_mean = (fit(prev_mean[np.newaxis, ])
                             + fit2(prev_mean[np.newaxis, ])
                             * prev_var[np.newaxis, ] / 2)
                next_mean[next_mean > 1] = 1
                next_mean[next_mean < 0] = 0

                next_var = ((1 - 1/self.N[np.newaxis, : , np.newaxis,
                                          np.newaxis, np.newaxis, np.newaxis])
                            * fit1(prev_mean[np.newaxis, ]) ** 2
                            * prev_var[np.newaxis, ])
                next_var[next_var < 0] = 0
                next_var += next_mean * (1 - next_mean) / self.N[np.newaxis,
                                                                 :, np.newaxis,
                                                                 np.newaxis,
                                                                 np.newaxis,
                                                                 np.newaxis]
                next_var[next_var < 0] = 0
                next_var[next_var > 1] = 1
                next_var[
                    np.greater_equal(
                        next_var, next_mean * (1 - next_mean))] = (
                            (next_mean * (1 - next_mean))[
                                np.greater_equal(
                                    next_var, next_mean * (1 - next_mean))])
            elif self.rec == 'WF':
                next_mean = np.full(shape=(1, len(self.x0), len(self.N),
                                           len(self.s), len(self.h),
                                           len(self.u), len(self.v)),
                                    fill_value=-1, dtype=float)
                next_var = np.full(shape=(1, len(self.x0), len(self.N),
                                          len(self.s), len(self.h),
                                          len(self.u), len(self.v)),
                                   fill_value=-1, dtype=np.float)
                for i, size in enumerate(self.N):
                    for j, freq in enumerate(self.x0):
                    # getting the next distribution for the wright fisher
                    # process
                        # making the matrix product
                        inv_axes = list(range(len(prev_dist[i][j].shape)))
                        inv_axes[0] = 1
                        inv_axes[1] = 0
                        next_dist = np.transpose(prev_dist[i][j], inv_axes)
                        next_dist = next_dist * trans[i]
                        next_dist = np.sum(next_dist, axis=0)
                        next_dist = next_dist[np.newaxis, ]

                        assert next_dist.shape == prev_dist[i][j].shape

                        # updating the distribution list
                        prev_dist[i][j] = next_dist

                        # computing moments
                        k_over_N = np.arange(size + 1) / size
                        k_over_N = k_over_N.reshape(1, size + 1, 1, 1, 1, 1)

                        assert len(k_over_N.shape) == len(next_dist.shape)

                        k_over_N_squared = k_over_N ** 2

                        next_mean[0, j, i] = np.sum(k_over_N * next_dist,
                                                    axis=1)
                        
                        next_var[0, j, i] = (np.sum(k_over_N_squared
                                                    * next_dist, axis=1)
                                             - next_mean[0, j, i] ** 2)
                        
            else:
                raise NotImplementedError

            prev_mean = next_mean[0, ]
            prev_var = next_var[0, ]

            last_gen += 1

            if store:
                self.moments[0, last_gen, ] = prev_mean.copy()
                self.moments[1, last_gen, ] = prev_var.copy()

        # we return in the same shape it's stored
        ret_mat = np.empty(shape=(2, *self.moments.shape[2:]))
        ret_mat[0, ] = prev_mean
        ret_mat[1, ] = prev_var

        return ret_mat

    def compute_fixations(self, gen, app = 'beta_tataru',
                          store = True, **kwargs):
        """ this method compute the approximated fixations probabilities of
        a wright-fisher process after gen generations.
        app is a string indicating how to approximate transitions and 
        integrate them in these computations.
        If store is true, all probabilities used in 
        the computations are stored is the fix_proba attribute such that
        self.fix_proba[i, j, k, l, m, n, o, p] is the probability for the 
        wright-fisher process to be fixed (in 0 for i = 0 or in 1 for i = 1) 
        after j generations from an initial frequency
        self.x0[k], were the parameters of the process are : 
        N = self.N[l]
        s = self.s[m]
        h = self.h[n]
        u = self.u[o]
        v = self.v[p]
        """
        # setting the approximation recursion
        self.app = app
        
        # the last fixation probability computed is
        last_gen_fix = self.fix_proba.shape[1] - 1

        # if the probability is already computed, return it
        if last_gen_fix >= gen:
            return self.fix_proba[:, gen, ]
        
        # the last moment generation already computed is
        last_gen_mom = self.moments.shape[1] - 1

        # approximated moments until this time are needed
        if gen > last_gen_mom:
            self.compute_moments(gen = gen, store = True)


        # if store  is true, increase the self.fix_proba matrix size to store
        # the new computations
        if store:
            fix_proba_t = np.full(shape = (2, gen + 1,
                                           *self.fix_proba.shape[2:]),
                                  fill_value = np.nan)
            fix_proba_t[:, :(last_gen_fix + 1), ] = self.fix_proba
            self.fix_proba = fix_proba_t

        # getting the fitness function and it's derivatives
        fit = self.fitness[0]
        fit1 = self.fitness[1]
        fit2 = self.fitness[2]

        prev_p0 = self.fix_proba[0, last_gen_fix].copy()
        prev_p1 = self.fix_proba[1, last_gen_fix].copy()

        # do the recursion until this time
        while last_gen_fix < gen:

            if app == 'beta_tataru':
                with np.errstate(invalid = 'ignore', divide = 'ignore'):
                    scaling = (1 - self.fix_proba[0, last_gen_fix]
                               - self.fix_proba[1, last_gen_fix])
                    cond_mean = (
                        (self.moments[0, last_gen_fix, ]
                         - self.fix_proba[1, last_gen_fix, ])
                        / scaling)
                    cond_var = (
                        (self.moments[1, last_gen_fix, ]
                         + self.moments[0, last_gen_fix, ] ** 2
                         - self.fix_proba[1, last_gen_fix,  ])
                        / scaling
                        - cond_mean ** 2)
                    const = cond_mean * (1 - cond_mean) / cond_var - 1
                const[scaling  <= 0] = 0
                const[cond_var == 0] = 0
                cond_mean[scaling <= 0] = 0
                cond_alpha = cond_mean * const
                cond_beta = (1 - cond_mean) * const
                # this mask capture values where ss.beta is either nan or 0
                mask = (cond_alpha <= 0) | (cond_beta <= 0) | (
                    ss.beta(cond_alpha, cond_beta) == 0)

                # p0n+1 = p0n * (1 - v) ** N + p1n * u ** N
                # + (1 - p0n - p1n) * (1 - u - v) ** N
                # * Beta(cond_alpha, cond_beta + N) / Beta(cond_alpha,
                #                                           cond_beta)
                next_p0 = (prev_p0 * (
                    1 - self.v[np.newaxis, np.newaxis, np.newaxis, np.newaxis,
                               np.newaxis, :]) ** self.N[np.newaxis,
                                                          :, np.newaxis,
                                                          np.newaxis,
                                                          np.newaxis,
                                                          np.newaxis]
                          + prev_p1 * (
                              self.u[np.newaxis, np.newaxis, np.newaxis,
                                     np.newaxis, :,
                                     np.newaxis] ** self.N[np.newaxis,
                                                           :, np.newaxis,
                                                           np.newaxis,
                                                           np.newaxis,
                                                           np.newaxis]))

                with np.errstate(invalid = 'ignore', divide = 'ignore'):
                    next_p0 += (
                        (1 - prev_p1 - prev_p0)
                        * (1 - self.u[np.newaxis, np.newaxis, np.newaxis,
                                      np.newaxis, :, np.newaxis]
                           - self.v[np.newaxis, np.newaxis, np.newaxis,
                                    np.newaxis,
                                    np.newaxis, :]) ** self.N[np.newaxis,
                                                              :,
                                                              np.newaxis,
                                                              np.newaxis,
                                                              np.newaxis,
                                                              np.newaxis]
                        * ss.beta(cond_alpha, cond_beta + self.N[
                            np.newaxis, :, np.newaxis, np.newaxis,
                            np.newaxis, np.newaxis])
                        / ss.beta(cond_alpha, cond_beta))

                next_p0[mask] = prev_p0[mask]
                
                next_p1 = (prev_p0 * (
                    self.v[np.newaxis, np.newaxis, np.newaxis, np.newaxis,
                           np.newaxis, :]) ** self.N[np.newaxis,
                                                     :, np.newaxis,
                                                     np.newaxis,
                                                     np.newaxis,
                                                     np.newaxis]
                          + prev_p1 * (
                              1 - self.u[np.newaxis, np.newaxis, np.newaxis,
                                         np.newaxis, :,
                                         np.newaxis] ** self.N[np.newaxis,
                                                               :, np.newaxis,
                                                               np.newaxis,
                                                               np.newaxis,
                                                               np.newaxis]))
                with np.errstate(invalid = 'ignore', divide = 'ignore'):
                    next_p1 += (
                        (1 - prev_p1 - prev_p0)
                        * (1 - self.u[np.newaxis, np.newaxis, np.newaxis,
                                      np.newaxis, :, np.newaxis]
                           - self.v[np.newaxis, np.newaxis, np.newaxis,
                                    np.newaxis,
                                    np.newaxis, :]) ** self.N[np.newaxis,
                                                              :,
                                                              np.newaxis,
                                                              np.newaxis,
                                                              np.newaxis,
                                                              np.newaxis]
                        * ss.beta(cond_alpha + self.N[np.newaxis,
                                                      :,
                                                      np.newaxis,
                                                      np.newaxis,
                                                      np.newaxis,
                                                      np.newaxis],
                                  cond_beta)
                        / ss.beta(cond_alpha, cond_beta))
                next_p1[mask] = prev_p1[mask]
                
            elif app == 'beta_custom':
                pass
            elif app == 'beta_numerical':
                scaling = (1 - self.fix_proba[0, last_gen_fix]
                           - self.fix_proba[1, last_gen_fix])
                with np.errstate(invalid = 'ignore', divide = 'ignore'):
                    cond_mean = (
                        (self.moments[0, last_gen_fix, ]
                         - self.fix_proba[1, last_gen_fix, ])
                        / scaling)
                    cond_var = (
                        (self.moments[1, last_gen_fix, ]
                         + self.moments[0, last_gen_fix, ] ** 2
                         - self.fix_proba[1, last_gen_fix,  ])
                        / scaling
                        - cond_mean ** 2)
                cond_mean[scaling <= 0] = 0.5
                cond_var[scaling <= 0] = 0
                cond_var[cond_var < 0] = 0

                with np.errstate(divide = 'ignore', invalid = 'ignore'):
                    const = cond_mean * (1 - cond_mean) / cond_var - 1
                const[scaling  <= 0] = 0
                const[cond_var == 0] = 0
                cond_alpha = cond_mean * const
                cond_beta = (1 - cond_mean) * const
                mask = (cond_alpha <= 0) | (cond_beta <= 0) | (
                    ss.beta(cond_alpha, cond_beta) == 0)
                

                next_p0 = (
                    prev_p0 * (
                        1 - fit(np.zeros(shape = (1, 1, 1, 1, 1, 1, 1)))
                    ) ** self.N[np.newaxis, :, np.newaxis, np.newaxis,
                                np.newaxis, np.newaxis]
                    + prev_p1 * (
                        1 - fit(np.ones(shape = (1, 1, 1, 1, 1, 1, 1)))
                    ) ** self.N[np.newaxis, :, np.newaxis, np.newaxis,
                                np.newaxis, np.newaxis])

                x = kwargs['grid'][:, np.newaxis, np.newaxis, np.newaxis,
                                   np.newaxis, np.newaxis, np.newaxis]
                with np.errstate(over = 'ignore'):
                    to_int = np.exp(
                        self.N[np.newaxis, :, np.newaxis, np.newaxis,
                               np.newaxis, np.newaxis] * np.log(
                                   1 - fit(x))
                        + (cond_alpha - 1) * np.log(x)
                        + (cond_beta - 1) * np.log(1 - x)
                        - ss.betaln(cond_alpha, cond_beta))
                    assert np.all((1 - fit(x) < 1))
                    assert np.all((1 - fit(x) > 0))
                                   
                # replacing inf values by the upper bound in numpy float
                to_int[to_int == np.inf] = np.finfo(np.float64).max 

                integrated = np.sum(
                    (to_int[:-1, ] + to_int[1:, ]) / 2
                    * (x[1:, ] - x[:-1, ]),
                    axis = 0)


                next_p0 += (
                    (1 - prev_p0 - prev_p1) * integrated)
                next_p0 = next_p0[0, ]
                
                next_p0[mask] = prev_p0[mask]
                
                next_p1 = (
                    prev_p0 * (
                        fit(np.zeros(shape = (1, 1, 1, 1, 1, 1, 1)))
                    ) ** self.N[np.newaxis, :, np.newaxis, np.newaxis,
                                np.newaxis, np.newaxis]
                    + prev_p1 * (
                        fit(np.ones(shape = (1, 1, 1, 1, 1, 1, 1)))
                    ) ** self.N[np.newaxis, :, np.newaxis, np.newaxis,
                                np.newaxis, np.newaxis])

                x = kwargs['grid'][:, np.newaxis, np.newaxis, np.newaxis,
                                   np.newaxis, np.newaxis, np.newaxis]
                to_int = np.exp(
                    self.N[np.newaxis, :, np.newaxis, np.newaxis,
                           np.newaxis, np.newaxis] * np.log(fit(x))
                    + (cond_alpha - 1) * np.log(x)
                    + (cond_beta - 1) * np.log(1 - x)
                    - ss.betaln(cond_alpha, cond_beta))

                integrated = np.sum(
                    (to_int[:-1, ] + to_int[1:, ]) / 2
                    * (x[1:, ] - x[:-1, ]),
                    axis = 0)
                
                next_p1 += (
                    (1 - prev_p0 - prev_p1) * integrated)
                next_p1 = next_p1[0, ]
                
                next_p1[mask] = prev_p1[mask]

                
            elif app == 'gauss_numerical':
                pass
            elif app == 'wf_exact':
                pass
            else:
                raise NotImplementedError

            prev_p0 = next_p0
            prev_p1 = next_p1

            last_gen_fix += 1

            if store:
                self.fix_proba[0, last_gen_fix, ] = prev_p0.copy()
                self.fix_proba[1, last_gen_fix, ] = prev_p1.copy()

        ret_mat = np.empty(shape = (2, *self.fix_proba.shape[2:]))
        ret_mat[0, ] = prev_p0
        ret_mat[1, ] = prev_p1

        return ret_mat

    def export(self, name, mode = 'w', form = 'csv'):
        if form == 'csv':
            self._export_csv(name = name, mode = mode)
        elif form == 'pic':
            self._export_pic(name = name, mode = mode)
        else:
            raise ValueError('format {f} not supported'.format(f=form))

    def _export_csv(self, name, mode):

        # open the file
        with open(name, mode) as f:
            # the the mode is not append mode
            if mode != 'a':
                header = ['gen', 'x0', 'N', 's', 'h', 'u', 'v', 'rec', 'app',
                          'mom1', 'mom2', 'p0', 'p1']
                f.write(','.join(header) + '\n')
            mom_shape = self.moments.shape
            fix_shape = self.fix_proba.shape
            gen_max = max(mom_shape[1], fix_shape[1])
            gen_min = min(mom_shape[1], fix_shape[1])

            for jx0, jN, js, jh, ju, jv, gen in itt.product(
                    range(len(self.x0)), range(len(self.N)),
                    range(len(self.s)), range(len(self.h)),
                    range(len(self.u)), range(len(self.v)),
                    range(gen_min)):
                line = []
                line.append(str(gen))
                line.append(str(self.x0[jx0]))
                line.append(str(self.N[jN]))
                line.append(str(self.s[js]))
                line.append(str(self.h[jh]))
                line.append(str(self.u[ju]))
                line.append(str(self.v[jv]))
                line.append(self.rec)
                line.append(self.app)
                line.append(str(self.moments[0, gen, jx0, jN, js, jh, ju, jv]))
                line.append(str(self.moments[1, gen, jx0, jN, js, jh, ju, jv]))
                line.append(
                    str(self.fix_proba[0, gen, jx0, jN, js, jh, ju, jv]))
                line.append(
                    str(self.fix_proba[1, gen, jx0, jN, js, jh, ju, jv]))
                f.write(','.join(line) + '\n')
                                         
    def _export_pic(self, name, mode):
        raise NotImplementedError


def main(gen, x, N, Ns, h, u, v, app, out):
    for size in N:
        s = np.array(Ns) / size
        ma = MomentsCalculator(x0 = np.array(x), N = size, s = s,
                               h = np.array(h), u = np.array(u),
                               v = np.array(v), rec = 'Tay2')
        ma.compute_fixations(gen, app = 'beta_tataru')
        if out in os.listdir():
            ma.export(out, mode = 'a')
        else:
            ma.export(out, mode = 'w')

            
if __name__ == '__main__':
    parser = arg.ArgumentParser()

    parser.add_argument('--gen', dest = 'gen', action = 'store',
                        type = int, required = True)
    parser.add_argument('-N', '--popsize', dest = 'N', action = 'store',
                        type = int, nargs = '+', required = True)
    parser.add_argument('--Ns', dest = 'Ns', action = 'store', type = float,
                        nargs = '+', required = True)
    parser.add_argument('-x', '--init_freq', dest = 'x', action = 'store',
                        type = float, nargs = '+', required = True)
    parser.add_argument('-u', dest = 'u', action = 'store', type = float,
                        nargs = '+', default = [0])
    parser.add_argument('-v', dest = 'v', action = 'store', type = float,
                        nargs = '+', default = [0])
    parser.add_argument('-d', '--dominance', dest = 'd', action = 'store',
                        type = float, nargs = '+', default = [0.5])
    parser.add_argument('-a', '--approximation', dest = 'a', action = 'store',
                        type = str, nargs = '+', default = ['beta_tataru'])
    parser.add_argument('-o', '--output', dest = 'o', action = 'store',
                        type = str, required = True)

    args = parser.parse_args()

    main(args.gen, args.x, args.N, args.Ns, args.d, args.u, args.v, args.a,
         args.o)
    

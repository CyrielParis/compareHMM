# encoding: utf-8

import numpy as np
import itertools as it
import scipy.stats as sss
import scipy.special as ss
import warnings as war
from . import moment_approx as dm
from . import rational_fraction as rf
import time as tm

class HMM:
    def __init__(self, times, model = 'spikedbeta', init = 'cunif',
                 nop = 129, **kwargs):
        """ Initialiser for the class HMM :
            - transitions are functions having the following signature :
                f(delta_t, x, x', **kwargs) with each argument is a numpy array
            - times is a numpy array of HMM's times (in generations)
            - nop is the number of points used to compute integrals
         """

        self.nop = nop
        self.T = times
        self.delta_t = times[1:] - times[:-1]

        self.T_size = len(times)
        self.delta_t_size = len(times) - 1

        # in kwargs there is some arrays for model parameters
        for k, v in kwargs.items():
            if isinstance(v, np.ndarray):
                # we add the corresponding attribute
                setattr(self, k, v)
                # we add the length of this parameters vector, and store it in
                # an attribute. for example, len(N) is stored in self.N_size
                setattr(self, k + '_size', len(v))
            if model == 'custom' and k in ['transition_d', 'transition_c']:
                setattr(self, k, v)

        self.params = self._param_order(model)
        self._set_transition(model)
        self._set_init_dist(init)
        self._set_default_X()
        self.Q = self._compute_transition()
        self.mu = self._compute_init()
        self.reset_emission()

    @staticmethod
    def _param_order(model):
        if model == 'neutralnicholgauss':
            return ['N']
        
        elif model == 'nicholgauss':
            return ['N', 's', 'h']
        
        elif model == 'gauss':
            return ['N', 's', 'h']
        
        elif model == 'wrightfisher':
            return ['N', 's', 'h', 'u', 'v']
        
        elif model == 'beta':
            return ['N', 's', 'h']
        elif model == 'spikedbeta':
            return ['N', 's', 'h']
        else:
            raise NotImplementedError(model, ' not implemented')

    def _set_init_dist(self, init='cunif'):
        """this method set the initial distribution of the model
        init parameter is a string
        the function set attributes :
        self.init_cont_dens : initial distribution density
        self.init_cont_supp : initial distribution continuous support
        self.init_disc_prob : initial distribution probability
        self.init_disc_supp : initial distribution discrete support
        """

        if init == 'cunif':
            # this distribution is purely continuous with uniform density over
            # [0, 1]

            density = self._init_cunif_density
            proba = self._init_cunif_proba

            self.init_cont_dens = density
            self.init_cont_supp = np.array([[0, 1]])
            self.init_disc_prob = proba
            self.init_disc_supp = np.array([])

        elif init == 'dunif':
            # this distribution is uniform over the discrete support

            density = self._init_dunif_density
            proba = self._init_dunif_proba

            self.init_cont_dens = density
            self.init_cont_supp = np.array([])
            self.init_disc_prob = proba
            self.init_disc_supp = self.trans_disc_supp
            
        elif init[0:5] == 'dirac':
            freq = float(init[5] + '.' + init[6:])
            density = self._init_dirac_density
            proba = self._init_dirac_proba(freq)

            self.init_cont_dens = density
            self.init_cont_supp = np.array([])
            self.init_disc_prob = proba
            self.init_disc_supp = self.trans_disc_supp
            
        else:
            raise NotImplementedError

    def _set_transition(self, model):
        """ set the transition distribution. the parameter model is a string
        specifying the model used
        """

        if model == 'wrightfisher':
            # the Wright-Fisher model specified in Ewens (2004) - Mathematical
            # population genetics.
            # as the transition matrix size depend on self.N_size, in this model
            # you can only do things with one N (so N_size == 1)
            if self.N_size > 1:
                raise ValueError('In the Wright-Fisher model, you can\'t give'
                                 + 'more than one value for N')
            support = np.arange(self.N[0] + 1) / self.N[0]

            self.trans_cont_dens = lambda *x, **y: 0
            self.trans_cont_supp = np.array([])
            self.trans_disc_prob = self._wrightfisher_proba
            self.trans_disc_supp = support
            
        elif model == 'gauss':
            density = self._gauss_density
            
            self.trans_cont_dens = density
            self.trans_cont_supp = np.array([[0, 1]])
            self.trans_disc_prob = lambda *x, **y: 0
            self.trans_disc_supp = np.array([])

        elif model == 'neutralnicholgauss':
            # In this model, the density between [0, 1] is a gaussian with
            # mean x and for variance F*x*(1-x) with F = (1-(1-1/N)**delta_t)
            # the discrete probabilities are the tails of the gaussian
            # distribution

            density = self._neutralnicholgauss_density
            proba = self._neutralnicholgauss_proba

            self.trans_cont_dens = density
            self.trans_cont_supp = np.array([[0, 1]])
            self.trans_disc_prob = proba
            self.trans_disc_supp = np.array([0, 1])

        elif model == 'nicholgauss':
            # In this model, the density between [0, 1] is a gaussian with
            # mean and variance are computed with delta method
            # the discrete probabilities are the tails of the gaussian
            # distribution

            density = self._nicholgauss_density
            proba = self._nicholgauss_proba

            self.trans_cont_dens = density
            self.trans_cont_supp = np.array([[0, 1]])
            self.trans_disc_prob = proba
            self.trans_disc_supp = np.array([0, 1])            
            
        elif model == 'beta':

            density = self._beta_density
            proba = self._beta_proba

            self.trans_cont_dens = density
            self.trans_cont_supp = np.array([[0.00001, 0.99999]])
            self.trans_disc_prob = lambda *x, **y: 0
            self.trans_disc_supp = np.array([])
            
        elif model == 'spikedbeta':
            density = self._spikedbeta_density
            proba = self._spikedbeta_proba

            self.trans_cont_dens = density
            self.trans_cont_supp = np.array([[0.00001, 0.99999]])
            self.trans_disc_prob = proba
            self.trans_disc_supp = np.array([0, 1])
        else:
            raise NotImplementedError(str(model) + ' model not implemented')

    def _set_default_X(self):
        """ set the default ndarray point grid in which compute densities
        set self.X_size and self.X attributes
        """

        # if there is no continuous part, set the number of points to 0
        if not len(self.trans_cont_supp):
            self.nop = 0
            bmin = 0
            bmax = 1
        else:
            bmin = np.amin(self.trans_cont_supp)
            bmax = np.amax(self.trans_cont_supp)

        # this is the discretisation of the interval
        cont_x = np.linspace(bmin, bmax, self.nop)
        # this is the support of the discrete part
        disc_x = self.trans_disc_supp
        # this is all the x on which we do the computations
        base_x = np.concatenate((cont_x, disc_x))
        self.X_size = len(base_x)

        # the shape of the X matrix is (T_size, X_size, params)
        # (for example : (T_size, X_size, N_size, s_size) )
        shape = (self.T_size, self.X_size,)
        prod = 1
        for p in self.params:
            p_size = p + '_size'
            shape += (getattr(self, p_size),)
            prod *= getattr(self, p_size)

        # duplicating along time axis
        X = np.repeat(base_x[np.newaxis, ], self.T_size, axis = 0)

        # duplicating along others axis
        X = np.repeat(X, prod).reshape(shape)
        self.X = X
        
    def _set_method(self, method):
        self.method = method

    def _compute_init(self):
        """ function computing the initial distribution of the HMM, the object
        calling this method must have corrects attributes :
            - self.X is a ndarray containing points in which compute the
            - initial distribution
            - self can have other parameters, the shapes have to correspond
        returns a matrix
        """

        x_c = self.X[0, :self.nop, ]
        x_d = self.X[0, self.nop:, ]

        params_c = (x_c, )
        params_d = (x_d, )

        params_shape = ()

        for p in self.params:
            par = getattr(self, p)
            params_c += (par, )
            params_d += (par, )
            params_shape += (len(par), )

        mu_shape = (self.X_size, *params_shape)

        mu_c = self.init_cont_dens(*params_c)
        mu_d = self.init_disc_prob(*params_d)

        mu = np.zeros(shape=mu_shape)
        mu[:self.nop, ] = mu_c
        mu[self.nop:, ] = mu_d

        return mu

    def _compute_transition(self):
        """ function computing the whole transition matrix of the HMM,
        the object calling this method must have corrects attributes :
            - self.delta_t is a numpy array recording step durations
            - self.X is a ndarray containing the points in which compute Q
            - self can have other parameters, the shapes have to correspond
        this function returns a matrix Q
        """

        params_shape = self.X.shape[2:]
        Q_shape = (self.delta_t_size, self.X_size, self.X_size, *params_shape)

        x = self.X[:-1, ]
        xprime_c = self.X[1:, :self.nop, ]
        xprime_d = self.X[1:, self.nop:, ]

        params_c = (self.delta_t, x, xprime_c, )
        params_d = (self.delta_t, x, xprime_d, )

        other_params = {}

        for p in self.params:
            other_params[p] = getattr(self, p)

        Q = np.zeros(shape=Q_shape)
        # Continuous part
        Q[:, :, :self.nop, ] = self.trans_cont_dens(*params_c, **other_params)
        # Discrete part
        Q[:, :, self.nop:, ] = self.trans_disc_prob(*params_d, **other_params)
        return Q

    def add_emission(self, y, n):
        """ this method add a new trajectory """

        new_y = y.reshape((1, self.T_size))
        new_n = n.reshape((1, self.T_size))

        self.yk = np.concatenate((self.yk, new_y), axis=0)
        self.nk = np.concatenate((self.nk, new_n), axis=0)

    def reset_emission(self):
        """ reset the observations """
        self.yk = np.array([]).reshape((0, self.T_size))
        self.nk = np.array([]).reshape((0, self.T_size))

    def get_traj(self, index):
        """ method returning a tuple (y, n) where y is the index-th trajectory
        of the HMM, and n is the index-th size of sampling vector
        """

        y = self.yk[index]
        n = self.nk[index]
        return y, n

    def get_traj_list(self):
        """ method returning a tuple (y, n) where y is the list of all
        trajectories and n is the list of sampling sizes
        """

        number_of_traj = self.yk.shape[0]
        y = [self.yk[index] for index in range(number_of_traj)]
        n = [self.nk[index] for index in range(number_of_traj)]
        return y, n

    def _compute_emission(self, y, n, log = False):
        """ return a ndarray containing the probabilities of the observations.
        parameter y is an array containing the observations (so total length is
        self.T_size)
        parameter n is an array containing parameters for emission law
        (supposed here to be binomial)
        one must have len(y) == len(n)
        the return is a ndarray with shape (self.T_size, self.X_size, *p_size)
        where E[i, j, *params] is the probability to observe y[i] given x[j]
        under the model given by *params
        if log parameter is set to True, return the log probability
        """
        shape_E = (self.T_size, self.X_size, )
        for p in self.params:
            p_size = p + '_size'
            shape_E += (getattr(self, p_size), )
        E = np.ones(shape=shape_E)
        for i in range(self.T_size):
            if log:
                if n[i] == 0:
                    E[i, ] = 0
                else:
                    E[i, ] = sss.binom.logpmf(y[i], n[i], self.X[i, ])
                    assert np.all(self.X[i, ] >= 0)
            else:
                if n[i] != 0:
                    E[i, ] = sss.binom.pmf(y[i], n[i], self.X[i, ])
        return E

    def backward(self, E, index = 0, store = True, scale = False):
        """ run the backward algorithm until index (by default the algorithm is
        fully runned so index=0)
        if store is true, the algorithm create and update an attribute beta
        which is the values of backward functions. Otherwise, values are not
        stored.
        method is the method used to compute integrals. method must have the
        following signature : method(f_val, x_val, *args, **kwargs) returning
        an array of integral's values, where x_val is a numpy array and f_val
        the numpy array corresponding to f(x) for x in x_val
        E is the matrix of emission probabilities (obtained with
        _compute_emission)
        if scale is True (option possible only if store is true too). The
        corresponding coefficient is stored in the log form
        """

        # checking args
        if scale and not store:
            raise AttributeError('you can\'t use scale option without store ' +
                                 'option')

        # setting the integration method
        if not hasattr(self, 'method'):
            self._set_method(self._trapezium)

        # getting the shape of beta
        shape_beta = (self.T_size, self.X_size, )
        for p in self.params:
            p_size = p + '_size'
            shape_beta += (getattr(self, p_size), )

        # initialising the beta scaling coefficient
        if scale:
            scale_shape = (shape_beta[0], ) + shape_beta[2:]
            self.beta_scale = np.zeros(shape = scale_shape)

        # initializing the big array containing backward function values
        if store:
            self.beta = np.full(shape = shape_beta, fill_value = np.nan)


        # for one step the corresponding beta has the following shape
        shape_one_beta = shape_beta[1:]

        # the 1st beta is constant to 1
        old_beta = np.ones(shape = shape_one_beta)
        if store:
            self.beta[-1, ] = np.copy(old_beta)

        if index == self.delta_t_size:
            return old_beta


        # initialize the vector which will store the new beta computed
        new_beta = np.full(shape=shape_one_beta, fill_value=np.nan)

        # for each transition we compute the next backward function
        for i in range(self.delta_t_size):
            # the algorithm is backward so at the 1st step we need to get the
            # true index which is, in this case -1
            true_index = - 1 - i

            beta_to_integr = (self.Q[true_index, :, :, ] *
                              old_beta[np.newaxis, :, ] *
                              E[true_index, np.newaxis, :, ])

            new_beta = self.method(beta_to_integr[:, :self.nop, ],
                                   self.X[true_index, :self.nop, ],
                                   axis=1)
            new_beta = new_beta + np.sum(beta_to_integr[:, self.nop:, ],
                                         axis=1)

            # if we scale the backward algorithm
            if scale:
                # copy the current beta
                log_beta = np.copy(new_beta)

                # replace every 0 by np.nan
                log_beta[log_beta == 0] = np.nan

                # take the log of the result
                log_beta = np.log(log_beta)

                # keep min and max values ignoring nan numbers
                mini_beta = np.nanmin(log_beta, axis = 0)
                maxi_beta = np.nanmax(log_beta, axis = 0)
                    
                # compute the scaling coefficient
                i = true_index - 1
                self.beta_scale[i, ] = (maxi_beta + mini_beta) / 2
                
                # scaling beta function
                new_beta = (new_beta * np.exp(- self.beta_scale[i, ]))

            if store:
                self.beta[true_index - 1, ] = np.copy(new_beta)

            if index == self.T_size + true_index:
                return new_beta
            old_beta = new_beta
        return old_beta

    def forward(self, E, index = None, store = True, scale = False,
                log_em = False):

        # checking args
        if scale and not store:
            raise AttributeError('you can\'t use scale option without store ' +
                                 'option')

        # if index == None we run the forward algorithm until the end
        if index is None:
            index = self.T_size

        # setting the integration method
        if not hasattr(self, 'method'):
            self._set_method(self._trapezium)

        shape_alpha = (self.T_size, self.X_size, )
        for p in self.params:
            p_size = p + '_size'
            shape_alpha += (getattr(self, p_size), )
        if store:
            self.alpha = np.full(shape=shape_alpha, fill_value=np.nan)

        # initialising the alpha scaling coefficient
        if scale:
            scale_shape = (shape_alpha[0], ) + shape_alpha[2:]
            self.alpha_scale = np.zeros(shape=scale_shape)

        # for one step, the corresponding alpha has the following shape
        shape_one_alpha = shape_alpha[1:]

        # the 1st alpha is just computed against the 1st emission
        if log_em:
            with np.errstate(divide = 'ignore', invalid = 'ignore'):
                old_alpha = np.log(self.mu)                
                old_alpha += E[0, ]
            old_alpha = np.exp(old_alpha)
        else:
            old_alpha = self.mu * E[0, ]

        if store:
            self.alpha[0, ] = np.copy(old_alpha)
        if index == 0:
            return old_alpha

        # initialize the vector containing the new alpha computed
        new_alpha = np.full(shape=shape_one_alpha, fill_value=np.nan)

        # for each transition, we compute the next forward measure
        for i in range(self.delta_t_size):
            with np.errstate(divide = 'ignore', invalid = 'ignore'):
                alpha_to_integr = (
                    np.log(self.Q[i, :, :, ])
                    + np.log(old_alpha[:, np.newaxis, ]))
                if log_em:
                    alpha_to_integr += E[i + 1, np.newaxis, :]
                else:
                    alpha_to_integr += np.log(E[i + 1, np.newaxis, :])
            alpha_to_integr = np.exp(alpha_to_integr)
                         
            # we need to fit the integration method interface
            number_of_axes = len(alpha_to_integr.shape)
            axes_to_inv = list(range(number_of_axes))
            axes_to_inv[0] = 1
            axes_to_inv[1] = 0
            mod_alpha = np.transpose(alpha_to_integr, axes_to_inv)

            new_alpha = self.method(mod_alpha[:, :self.nop, ],
                                    self.X[i, :self.nop, ],
                                    axis=1)
            new_alpha = new_alpha + np.sum(alpha_to_integr[self.nop:, :, ],
                                           axis = 0)

            # if we scale the forward algorithm
            if scale:
                # to scale, integrate the alpha measure and scale by this
                alpha_to_integr = new_alpha[:, np.newaxis, ]

                # we need to fit the integration method interface
                number_of_axes = len(alpha_to_integr.shape)
                axes_to_inv = list(range(number_of_axes))
                axes_to_inv[0] = 1
                axes_to_inv[1] = 0
                mod_alpha = np.transpose(alpha_to_integr, axes_to_inv)
                mod_alpha = self.method(mod_alpha[:, :self.nop, ],
                                        self.X[i + 1, :self.nop, ],
                                        axis=1)
                mod_alpha = mod_alpha + np.sum(alpha_to_integr[self.nop:, :, ],
                                               axis = 0)
                with np.errstate(divide = 'ignore', invalid = 'ignore'):
                    self.alpha_scale[i + 1, ] = np.log(mod_alpha[0])
                with np.errstate(invalid = 'ignore'):
                    new_alpha = new_alpha * np.exp(- self.alpha_scale[i + 1, ])
            if store:
                self.alpha[i + 1, ] = np.copy(new_alpha)
            if index == i + 1:
                return new_alpha
            old_alpha = new_alpha
        return old_alpha

    def likelihood(self, alpha, beta, x, log = False):
        """ function computing the likelihood from forward and backward
        algorithms given in alpha and beta parameters.
        alpha and beta are not supposed to be scaled
        """

        likelihood_to_int = alpha * beta
        ret = (np.sum(likelihood_to_int[self.nop:, ], axis=0) +
               self.method(likelihood_to_int[np.newaxis, :self.nop, ],
                           x[:self.nop, ]))

        if log:
            return np.log(ret[0, ])
        else:
            return ret[0, ]

    def find_max_L(self, L, axis = ()):
        raise NotImplementedError

    @staticmethod
    def _rectangle(*args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def _trapezium(f_to_int, x, axis = 1, *args, **kwargs):
        """ integration method using 1st order interpolation
        f_to_int is a numpy array containing values of the function to
        integrate.
        x is the values in which f is computed
        """

        f_diff = (f_to_int[:, 1:, ] + f_to_int[:, :-1, ]) / 2
        x_diff = (x[1:, ] - x[:-1, ])
        to_sum = f_diff * x_diff[np.newaxis, :, ]
        return np.sum(to_sum, axis = axis)

    @staticmethod
    def _simpsons(*args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def _wrightfisher_proba(delta_t, x, xprime, N, s, h, u, v):
        """ this function return the array Q containing probabilities such that
        Q[i, j, k, l, m, n, o, p] contains the probability of x' given x during
        delta_t in the selection and mutation Wright-Fisher process where 
        parameters are N, s, h, u, v
        i is the index relative to delta_t
        j is the index relative to x
        k is the index relative to x'
        l is the index relative to N
        m is the index relative to s
        n is the index relative to h
        o is the index relative to u
        p is the index relative to v
        """

        # initializing the returned array
        out_array = np.full(shape = (len(delta_t), x.shape[1], xprime.shape[1],
                                     1, len(s), len(h), len(u), len(v)),
                            fill_value = np.nan)

        for i, t in enumerate(delta_t):
            # giving all parameters broadcasting compatible shapes
            x0 = x[i, :, np.newaxis, :, :, :, :, :]
            x1 = xprime[i, np.newaxis, :, :, :, :, :, :]
            size = N[:, np.newaxis, np.newaxis, np.newaxis, np.newaxis]
            sel = s[np.newaxis, :, np.newaxis, np.newaxis, np.newaxis]
            # giving sel a good shape to use it as a mask
            sel = np.repeat(sel, len(h), axis = 2)
            dom = h[np.newaxis, np.newaxis, :, np.newaxis, np.newaxis]
            d_to_a = u[np.newaxis, np.newaxis, np.newaxis, :, np.newaxis]
            a_to_d = v[np.newaxis, np.newaxis, np.newaxis, np.newaxis, :]

            # set the fitness function without mutation
            top_coef2 = sel * (1 - dom)
            top_coef1 = 1 + sel * dom
            top_coef0 = 0

            bot_coef2 = sel * (1 - 2 * dom)
            bot_coef1 = 2 * sel * dom
            bot_coef0 = 1

            # correcting coefs in 0 fitness cases
            top_coef2[sel == -1] = 0
            bot_coef2[sel == -1] = 0
            bot_coef1[sel == -1] = 1

            # with mutations
            top_coef2 = ((1 - d_to_a) * top_coef2
                         + a_to_d * (bot_coef2 - top_coef2))
            top_coef1 = ((1 - d_to_a) * top_coef1
                         + a_to_d * (bot_coef1 - top_coef1))
            top_coef0 = ((1 - d_to_a) * top_coef0
                         + a_to_d * (bot_coef0 - top_coef0))

            top = np.zeros(shape=(3, *top_coef2.shape))
            bot = np.zeros(shape=(3, *bot_coef2.shape))

            top[0, ] = top_coef0
            top[1, ] = top_coef1
            top[2, ] = top_coef2

            bot[0, ] = bot_coef0
            bot[1, ] = bot_coef1
            bot[2, ] = bot_coef2

            # define the fitness function
            fitness = rf.RationalFraction(rf.Polynomial(top),
                                          rf.Polynomial(bot))

            # in this model we assume len(N) = 1 so this make sense
            scal_size = N[0]

            # building the transition matrix for 1 generation
            k = (np.arange(scal_size + 1).reshape((scal_size + 1, 1,
                                                   1, 1, 1, 1)))
            k_over_N = k / scal_size

            fit = fitness(k_over_N)
            fit[fit>1] = 1

            assert np.max(fit) <= 1
            assert np.min(fit) >= 0
            
            trans = sss.binom.pmf(k[np.newaxis, ],
                                  scal_size,  fit[:, np.newaxis, ])
            # initializing the full transition matrix
            full_trans = np.eye(scal_size + 1)[:, :, np.newaxis, np.newaxis,
                                               np.newaxis, np.newaxis,
                                               np.newaxis]
            for j, val in enumerate(trans.shape[2:]):
                if val != 1:
                    full_trans = np.repeat(full_trans, repeats = val,
                                           axis = j + 2)
            for gen in range(t):
                full_trans = np.einsum('ij..., jk...->ik...',
                                        full_trans, trans)

            # at this point, full_trans contains trans ** t
            out_array[i, ] = full_trans
            
        return out_array
        

        

    @staticmethod
    def _neutralnicholgauss_proba(delta_t, x, xprime, N):
        """ this function returns the array Q containing probabilities such
        that Q[i, j, k, l] contains the corresponding probability of x' given x
        during delta_t in the neutral nicholson model where size is N
        i is the index relative to delta_t
        j is the index relative to x
        k is the index relative to x'
        l is the index relative to N
        """

        # giving all parameters broadcasting compatible shapes
        x0 = x[:, :, np.newaxis, :]
        x1 = xprime[:, np.newaxis, :, :]
        size = N[np.newaxis, np.newaxis, np.newaxis, :]
        t = delta_t[:, np.newaxis, np.newaxis, np.newaxis]

        # computing the standard deviation
        stdev = np.sqrt(x0 * (1 - x0) * (1 - (1 - 1/size)**t))

        # the shape of the array we return
        shape_out = (*x0.shape[:2], x1.shape[2], *x0.shape[3:],)

        # initializing the return matrix
        out_array = np.zeros(shape=shape_out)

        # use mask to set absorption probabilities giving absorption
        mask = ((x0 == 0) * (x1 == 0)) + ((x0 == 1) * (x1 == 1))
        out_array[mask] = 1

        # use mask to set absorption probabilities in 0
        mask = (x0 != 0) * (x0 != 1) * (x1 == 0)
        with np.errstate(divide = 'ignore', invalid = 'ignore'):
            out_array[mask] = sss.norm.cdf(x1, loc = x0, scale=stdev)[mask]

        # use mask to set absorption probabilities in 1
        mask = (x0 != 0) * (x0 != 1) * (x1 == 1)
        with np.errstate(divide='ignore', invalid = 'ignore'):
            out_array[mask] = sss.norm.sf(x1, loc = x0, scale=stdev)[mask]

        return out_array

    @staticmethod
    def _neutralnicholgauss_density(delta_t, x, xprime, N):
        """ this function returns the array Q containing probability density's
        values such that Q[i, j, k, l] contains the corresponding probability
        of x' given x during delta_t in the neutral nicholson model where size
        is N
        i is the index relative to delta_t
        j is the index relative to x
        k is the index relative to x'
        l is the index relative to N
        """

        # giving all parameters broadcasting compatible shapes
        x0 = x[:, :, np.newaxis, :]
        size = N[np.newaxis, np.newaxis, np.newaxis, :]
        t = delta_t[:, np.newaxis, np.newaxis, np.newaxis]
        x1 = xprime[:, np.newaxis, :, :]

        # computing the standard deviation
        stdev = np.sqrt(x0 * (1 - x0) * (1 - (1 - 1/size)**t))

        # some of stdev are 0 so norm.pdf can't be computed properly
        # we just mute warning returns
        with np.errstate(divide='ignore', invalid = 'ignore'):
            out_array = sss.norm.pdf(x1, loc = x0, scale=stdev)

        # if we are absorbed or if we want to go outside [0, 1] return 0
        mask = (x0 == 0) + (x0 == 1) + (x1 < 0) + (x1 > 1)

        out_array[mask] = 0

        return out_array

    @staticmethod
    def _gauss_density(delta_t, x, xprime, N, s, h = np.array([0.5])):
        # giving all parameters broadcasting compatible shapes
        t = delta_t[:, np.newaxis, np.newaxis,
                    np.newaxis, np.newaxis, np.newaxis]
        x0 = x[:, :, np.newaxis, ]
        x1 = xprime[:, np.newaxis, ]

        size = N[np.newaxis, np.newaxis, np.newaxis, :, np.newaxis, np.newaxis]
        sel = s[np.newaxis, np.newaxis, np.newaxis, np.newaxis, :, np.newaxis]
        dom = h[np.newaxis, np.newaxis, np.newaxis, np.newaxis, np.newaxis, :]

        # computing moments
        mom_cal = dm.MomentsCalculator(x0 = x[0, :, 0, 0, 0], N = N, s = s,
                                         h = h, rec = 'Tay2')
        mom_cal.compute_moments(gen = max(delta_t), store = True)
        moments_shape = np.broadcast(t, x0, x1, size, sel, dom).shape
        moments_shape = (2, *moments_shape)
        moments = np.empty(shape = moments_shape)
        for i, time in enumerate(delta_t):
            mom = mom_cal.moments[:, time, :, np.newaxis, :, :, :, 0, 0]
            moments[:, i, ] = mom
        
        mean = moments[0]
        stdev = np.sqrt(moments[1])

        del moments

        # muting some numpy warnings
        with np.errstate(divide='ignore', invalid='ignore'):
            out_array = sss.norm.pdf(x1, loc=mean, scale=stdev)
             
            scaling = (sss.norm.cdf(0, loc=mean, scale=stdev)
                       + sss.norm.sf(1, loc=mean, scale=stdev))

        del mean
        
        scaling[stdev == 0] = 0

        del stdev
        
        out_array = out_array / (1 - scaling)

        del scaling
        
        # if we are absorbed or if we want to go outside [0, 1] return 0
        mask = (x0 == 0) + (x0 == 1) + (x1 < 0) + (x1 > 1)
        out_array[mask] = 0

        return out_array        

    @staticmethod
    def _gauss_proba(delta_t, x, xprime, N, s, h = np.array([0.5])):
        x0 = x[:, :, np.newaxis, ]
        x1 = xprime[:, np.newaxis, ]

        # the shape of the array we return
        shape_out = (*x0.shape[:2], x1.shape[2], *x0.shape[3:], )

        out_array = np.zeros(shape=shape_out)

        mask = ((x0 == 0) * (x1 == 0) + (x0 == 1) * (x1 == 1))
        out_array[mask] = 1

        return out_array
    
    @staticmethod
    def _beta_density(delta_t, x, xprime, N, s, h = np.array([0.5])):
        # giving all parameters broadcasting compatible shapes
        t = delta_t[:, np.newaxis, np.newaxis,
                    np.newaxis, np.newaxis, np.newaxis]
        x0 = x[:, :, np.newaxis, ]
        x1 = xprime[:, np.newaxis, ]

        size = N[np.newaxis, np.newaxis, np.newaxis, :, np.newaxis, np.newaxis]
        sel = s[np.newaxis, np.newaxis, np.newaxis, np.newaxis, :, np.newaxis]
        dom = h[np.newaxis, np.newaxis, np.newaxis, np.newaxis, np.newaxis, :]

        # computing moments
        mom_cal = dm.MomentsCalculator(x0 = x[0, :, 0, 0, 0], N = N, s = s,
                                         h = h, rec = 'Tay2')
        mom_cal.compute_moments(gen = max(delta_t), store = True)
        moments_shape = np.broadcast(t, x0, x1, size, sel, dom).shape
        moments_shape = (2, *moments_shape)
        moments = np.empty(shape = moments_shape)
        for i, time in enumerate(delta_t):
            mom = mom_cal.moments[:, time, :, np.newaxis, :, :, :, 0, 0]
            moments[:, i, ] = mom
        
        mean = moments[0]
        var = moments[1]

        del moments

        # parameters
        # Muting varning related to dividing by 0
        with np.errstate(divide = 'ignore', invalid = 'ignore'):
            const = mean * (1 - mean) / var - 1
        const[var == 0] = 0
        alpha = mean * const
        beta = (1 - mean) * const

        del const

        # the shape of the array we return
        shape_out = (*x0.shape[:2], x1.shape[2], *x0.shape[3:], )

        # initializing the return matrix
        out_array = (alpha - 1) * np.log(x1) + (beta - 1) * np.log(1 - x1)
        out_array -= ss.betaln(alpha, beta)
        out_array = np.exp(out_array)

        # if there is some approximation errors, return 0 (the np.zeros
        # is here only to fit shapes
        mask = (alpha == 0) + (beta == 0) + np.zeros(x1.shape,dtype = bool)
        out_array[mask] = 0

        del alpha
        del beta
        
        # if we are absorbed or if we want to go outside [0, 1] return 0
        mask = (x0 == 0) + (x0 == 1) + (x1 < 0) + (x1 > 1)
        out_array[mask] = 0


        return out_array

    @staticmethod
    def _beta_proba(delta_t, x, xprime, N, s, h = np.array([0.5])):
        x0 = x[:, :, np.newaxis, ]
        x1 = xprime[:, np.newaxis, ]

        # the shape of the array we return
        shape_out = (*x0.shape[:2], x1.shape[2], *x0.shape[3:], )

        out_array = np.zeros(shape = shape_out)

        mask = ((x0 == 0) * (x1 == 0) + (x0 == 1) * (x1 == 1))
        out_array[mask] = 1

        return out_array

    @staticmethod
    def _spikedbeta_density(delta_t, x, xprime, N, s, h = np.array([0.5])):
        # giving all parameters broadcasting compatible shapes
        t = delta_t[:, np.newaxis, np.newaxis,
                    np.newaxis, np.newaxis, np.newaxis]
        x0 = x[:, :, np.newaxis, ]
        x1 = xprime[:, np.newaxis, ]

        size = N[np.newaxis, np.newaxis, np.newaxis, :, np.newaxis, np.newaxis]
        sel = s[np.newaxis, np.newaxis, np.newaxis, np.newaxis, :, np.newaxis]
        dom = h[np.newaxis, np.newaxis, np.newaxis, np.newaxis, np.newaxis, :]

        # computing moments
        mom_cal = dm.MomentsCalculator(x0 = x[0, :, 0, 0, 0], N = N, s = s,
                                         h = h, rec = 'Tay2')
        mom_cal.compute_moments(gen = max(delta_t), store = True)
        mom_cal.compute_fixations(gen = max(delta_t), store = True,
                                  app = 'beta_tataru')
        moments_shape = np.broadcast(t, x0, x1, size, sel, dom).shape
        moments_shape = (2, *moments_shape)
        moments = np.empty(shape = moments_shape)
        fixations = np.empty(shape = moments_shape)
        for i, time in enumerate(delta_t):
            mom = mom_cal.moments[:, time, :, np.newaxis, :, :, :, 0, 0]
            fix = mom_cal.fix_proba[:, time, :, np.newaxis, :, :, :, 0, 0]
            moments[:, i, ] = mom
            fixations[:, i, ] = fix

        mean = moments[0]
        var = moments[1]
        p0 = fixations[0]
        p1 = fixations[1]

        del moments
        del fixations

        scaling = (1 - p1 - p0)

        del p0

        cond_var = var + mean ** 2 - p1

        del var
        
        cond_mean = mean - p1
        
        del mean
        del p1
        
        with np.errstate(invalid = 'ignore', divide = 'ignore'):
            cond_mean /= scaling
            cond_var /= scaling

        cond_mean[scaling == 0] = 0
        cond_var[scaling == 0] = 0
            
        cond_var -= cond_mean ** 2


        # parameters
        # Muting varning related to dividing by 0
        with np.errstate(invalid = 'ignore', divide = 'ignore'):
            const = cond_mean * (1 - cond_mean) / cond_var - 1
        const[cond_var == 0] = 0

        del cond_var
        
        alpha = cond_mean * const
        beta = (1 - cond_mean) * const

        del const
        del cond_mean

        # for numerical reason, alpha and beta may be negative. Fixing it :
        alpha[alpha < 0] = 0
        beta[beta < 0] = 0

        # initializing the return matrix
        out_array = (
            (alpha - 1) * np.log(x1)
            + (beta - 1) * np.log(1 - x1))
        out_array -= ss.betaln(alpha, beta)
        with np.errstate(invalid = 'ignore', divide = 'ignore'):
            out_array += np.log(scaling)
        out_array = np.exp(out_array)

        out_array[scaling == 0] = 0

        del scaling

        # if we are absorbed or if we want to go outside [0, 1] return 0
        mask = (x0 == 0) + (x0 == 1) + (x1 < 0) + (x1 > 1)
        out_array[mask] = 0
        
        del mask

        # if there is some approximation errors, return 0 (the np.zeros
        # is here only to fit shapes
        mask = (alpha == 0) + (beta == 0) + np.zeros(x1.shape, dtype = bool)
        out_array[mask] = 0
        return out_array

    @staticmethod
    def _spikedbeta_proba(delta_t, x, xprime, N, s, h = np.array([0.5])):
        # giving all parameters broadcasting compatible shapes
        t = delta_t[:, np.newaxis, np.newaxis,
                    np.newaxis, np.newaxis, np.newaxis]
        x0 = x[:, :, np.newaxis, ]
        x1 = xprime[:, np.newaxis, ]

        size = N[np.newaxis, np.newaxis, np.newaxis, :, np.newaxis, np.newaxis]
        sel = s[np.newaxis, np.newaxis, np.newaxis, np.newaxis, :, np.newaxis]
        dom = h[np.newaxis, np.newaxis, np.newaxis, np.newaxis, np.newaxis, :]

        # computing moments
        mom_cal = dm.MomentsCalculator(x0 = x[0, :, 0, 0, 0], N = N, s = s,
                                         h = h, rec = 'Tay2')
        mom_cal.compute_moments(gen = max(delta_t), store = True)
        mom_cal.compute_fixations(gen = max(delta_t), store = True,
                                  app = 'beta_tataru')
        moments_shape = np.broadcast(t, x0, x1, size, sel, dom).shape
        moments_shape = (2, *moments_shape)
        fixations = np.empty(shape = moments_shape)
        for i, time in enumerate(delta_t):
            fix = mom_cal.fix_proba[:, time, :, np.newaxis, :, :, :, 0, 0]
            fixations[:, i, ] = fix
        
        p0 = fixations[0]
        p1 = fixations[1]

        del fixations

        # the shape of the array we return
        shape_out = (*x0.shape[:2], x1.shape[2], *x0.shape[3:], )

        out_array = np.full(shape = shape_out, fill_value = np.nan)

        mask = (x1 == 0) + np.zeros(x0.shape, dtype = bool)
        out_array[mask] = p0[mask]
        
        mask = (x1 == 1) + np.zeros(x0.shape, dtype = bool)
        out_array[mask] = p1[mask]

        return out_array
        
        
    @staticmethod
    def _nicholgauss_proba(delta_t, x, xprime, N, s, h = np.array([0.5])):
        # giving all parameters broadcasting compatible shapes
        x0 = x[:, :, np.newaxis, ]
        size = N[np.newaxis, np.newaxis, np.newaxis, :, np.newaxis, np.newaxis]
        t = delta_t[:, np.newaxis, np.newaxis,
                    np.newaxis, np.newaxis, np.newaxis]
        x1 = xprime[:, np.newaxis, ]
        sel = s[np.newaxis, np.newaxis, np.newaxis, np.newaxis, :, np.newaxis]
        dom = h[np.newaxis, np.newaxis, np.newaxis, np.newaxis, np.newaxis, :]

        # computing moments
        mom = dm.moments(t, x0, size, sel, dom)
        mean = mom[0]
        stdev = np.sqrt(mom[1])

        # the shape of the array we return
        shape_out = (*x0.shape[:2], x1.shape[2], *x0.shape[3:],)

        # initializing the return matrix
        out_array = np.zeros(shape = shape_out)

        # use mask to set absorption probabilities giving absorption
        mask = ((x0 == 0) * (x1 == 0)) + ((x0 == 1) * (x1 == 1))
        out_array[mask] = 1

        # use mask to set absorption probabilities in 0
        mask = (x0 != 0) * (x0 != 1) * (x1 == 0)
        with np.errstate(divide = 'ignore', invalid = 'ignore'):
            out_array[mask] = sss.norm.cdf(x1, loc=mean, scale=stdev)[mask]

        # use mask to set absorption probabilities in 1
        mask = (x0 != 0) * (x0 != 1) * (x1 == 1)
        with np.errstate(divide = 'ignore', invalid = 'ignore'):
            out_array[mask] = sss.norm.sf(x1, loc = mean, scale = stdev)[mask]

        return out_array

    @staticmethod
    def _nicholgauss_density(delta_t, x, xprime, N, s, h = np.array([0.5])):
        """ this function returns the array Q containing probability density's
        values such that Q[i, j, k, l, m, n] contains the corresponding
        probability of x' given x during delta_t in the nicholgauss model where
        N is the effective size, s is the selection parameter, h is the
        dominance parameter
        i is the index relative to delta_t
        j is the index relative to x
        k is the index relative to x'
        l is the index relative to N
        m is the index relative to s
        n is the index relative to h
        """

        # giving all parameters broadcasting compatible shapes
        t = delta_t[:, np.newaxis, np.newaxis,
                    np.newaxis, np.newaxis, np.newaxis]
        x0 = x[:, :, np.newaxis, ]
        x1 = xprime[:, np.newaxis, ]

        size = N[np.newaxis, np.newaxis, np.newaxis, :, np.newaxis, np.newaxis]
        sel = s[np.newaxis, np.newaxis, np.newaxis, np.newaxis, :, np.newaxis]
        dom = h[np.newaxis, np.newaxis, np.newaxis, np.newaxis, np.newaxis, :]

        # computing moments
        mom = dm.moments(t, x0, size, sel, dom)
        mean = mom[0]
        stdev = np.sqrt(mom[1])

        # muting some numpy warnings
        with np.errstate(divide = 'ignore', invalid = 'ignore'):
            out_array = sss.norm.pdf(x1, loc = mean, scale = stdev)

        # if we are absorbed or if we want to go outside [0, 1] return 0
        mask = (x0 == 0) + (x0 == 1) + (x1 < 0) + (x1 > 1)
        out_array[mask] = 0

        return out_array

    @staticmethod
    def _init_dirac_density(x, *args):
        """ this function returns the array containing probability density's 
        values corresponding to a dirac (so the continuous part is zero)
        """

        x_size = x.shape[0]
        shape_out = (x_size, )
        for param in args:
            shape_out += (len(param), )

        return np.zeros(shape = shape_out)
        
    @staticmethod
    def _init_dirac_proba(freq):
        """ this function returns the probability function corresponding to a 
        dirac in this frequency (1 on this frequency, 0 on the others)
        """

        def dirac_proba(x, *args):
            """ this function returns the array containing probability density's
            values corresponding to a dirac on freq
            """
            x_size = x.shape[0]
            shape_out = (x_size, )
            for param in args:
                shape_out += (len(param), )

            ret_array = np.zeros(shape=shape_out)
            ret_array[x == freq] = 1

            return ret_array
            
        return dirac_proba
        
    
    @staticmethod
    def _init_dunif_density(x, *args):
        """this function returns the array containing probability density's
        values corresponding to a discrte uniform distribution (so the 
        continuous part is zero
        """
        
        x_size = x.shape[0]
        shape_out = (x_size, )
        for param in args:
            shape_out += (len(param), )

        return np.zeros(shape = shape_out)

    @staticmethod
    def _init_dunif_proba(x, *args):

        x_size = x.shape[0]
        shape_out = (x_size, )
        for param in args:
            shape_out += (len(param), )

        uniform_proba = np.full(shape=shape_out, fill_value = 1 / (x_size - 1))

        return uniform_proba
    
    @staticmethod
    def _init_cunif_density(x, *args):
        """ this function returns the array containing probability density's
        values corresponding to a uniform density over [0, 1]
        """

        x_size = x.shape[0]
        shape_out = (x_size, )
        for param in args:
            shape_out += (len(param), )

        if np.all(x <= 1) and np.all(x >= 0):
            return np.ones(shape = shape_out)
        else:
            raise NotImplementedError

    @staticmethod
    def _init_cunif_proba(x, *args):
        """ this function returns the array containing probability distribution
        values corresponding to a uniform density over [0, 1] (so it's a matrix
        full of 0
        """

        x_size = x.shape[0]
        shape_out = (x_size, )
        for param in args:
            shape_out += (len(param), )

        return np.zeros(shape = shape_out)

if __name__ == '__main__':
    T = np.array([10, 20, 50, 70, 100])
    y = np.array([2, 8, 16, 21, 30])
    y = np.array([15, 15, 15, 15, 15])
    n = np.array([30, 30, 30, 30, 30])
    T = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90])
    y = np.array([15, 14,  8,  5,  6,  4,  6,  1, 13,  9])
    n = np.repeat(30, 10)
    #n = np.array([0, 0, 0, 0, 0])
    N = np.array([100, 500, 1000, 5000, 10000])
    N = np.array([100])
    # N = np.linspace(100,1000,91)
    s = np.linspace(-1, 2, 11)
    # s = np.array([0.2])
    h = np.array([0.2, 0.5, 0.8])
    u = np.array([1e-8])
    v = np.array([1e-8])
    u = v = np.array([0])

    print('Building the HMM...')
    a = HMM(T, model='spikedbeta', init='cunif', h=h, s=s,
            N=N, u=u, v=v, nop=65)
    print('Done')

    E = a._compute_emission(y, n)

    print('Running unscaled backward algorithm...')
    a.backward(E, store=True)
    print('Done')
    print('Running unscaled forward algorithm...')
    a.forward(E, store=True)
    print('Done')

    print('testing alpha_i(beta_i) : ')
    for i in range(len(T)):
        print('i = {i}'.format(i=i),
              a.likelihood(a.alpha[i, :, ], a.beta[i, :, ], a.X[i, ])[0,:,:])

    print('Running scaled backward algorithm...')
    a.backward(E, store=True,  scale=True)
    print('Done')
    print('Running scaled forward algorithm...')
    a.forward(E, store=True, scale=True)
    print('Done')

    print('testing alpha_i(beta_i) : ')
    for i in range(len(T)):
        print('i = {i} scaled likelihood'.format(i=i),
              a.likelihood((a.alpha[i, :, ] *
                            np.exp(np.sum(a.alpha_scale[:(i + 1), ], axis=0))),
                           (a.beta[i, :, ] *
                            np.exp(np.sum(a.beta_scale[i:, ], axis=0))),
                           a.X[i, ])[0,:,:])
        print('i = {i} log likelihood'.format(i=i),
              np.log(a.likelihood((a.alpha[i, :, ] *
                                   np.exp(np.sum(a.alpha_scale[:(i + 1), ],
                                                 axis=0))),
                                  (a.beta[i, :, ] *
                                   np.exp(np.sum(a.beta_scale[i:, ], axis=0))),
                                  a.X[i, ])[0,:,:]))
        print('i = {i} log scaled likelihood'.format(i=i),
              (a.likelihood(a.alpha[i, :, ], a.beta[i, :, ],
                            a.X[i, ], log=True) +
               np.sum(a.beta_scale[i:, ], axis=0) +
               np.sum(a.alpha_scale[:(i + 1), ], axis=0))[0,:,:])
    print('s = ',s)
    

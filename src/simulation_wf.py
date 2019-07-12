# encoding: utf-8

import numpy as np
import numpy.polynomial.polynomial as pol
import scipy.stats as sss
import pickle as pic
import itertools as itt

class WfSimulator:
    """ Simulator of a Wright-Fisher discrete process """

    def __init__(self, x0, N, u=0, v=0, **kwargs):
        """ constructor for WfSimulator object, all arguments given are numbers
        and kwargs must contain one of these options :
        - w0, w1, w2
        - s, h
        - s1, s2
        - s0, s2
        the other parameters correspond to ewens model parameters :
        x0 is the initial frequency, N * x0 must be an integer
        N is the effective size, type : integer
        u and v are mutation rates, u is the mutation rate from the derived
        form to the ancestral form and v is the other one
        """

        self.x0 = x0
        self.N = N
        self.u = u
        self.v = v

        keys = kwargs.keys()

        if 'w0' in keys and 'w1' in keys and 'w2' in keys:
            self.w0 = kwargs['w0']
            self.w1 = kwargs['w1']
            self.w2 = kwargs['w2']
            self.mode = 'w'
        elif 's' in keys and 'h' in keys:
            self.w0 = 1
            self.w1 = 1 + kwargs['s'] * kwargs['h']
            self.w2 = 1 + kwargs['s']
            self.s = kwargs['s']
            self.h = kwargs['h']
            self.mode = 'sh'
        elif 's1' in keys and 's2' in keys:
            self.w0 = 1
            self.w1 = 1 + kwargs['s1']
            self.w2 = 1 + kwargs['s2']
            self.mode = 's1s2'
        elif 's0' in keys and 's2' in keys:
            self.w0 = 1 + kwargs['s0']
            self.w1 = 1
            self.w2 = 1 + kwargs['s2']
            self.mode = 's0s2'
        else:
            raise ValueError('you must give the correct set of fitness '
                             + 'parameters')

        k_over_N = np.arange(N + 1) / N

        # top of the fitness function without mutation
        # for some extreme parameters values, the fitness has a different shape
        if self.w2 != 0:
            if self.w1 != 0:
                if self.w0 != 0:
                    # the most common case
                    top_wm = pol.Polynomial([0, self.w1, self.w2 - self.w1])
                    # bot of the fitness function without mutation
                    bot = pol.Polynomial([self.w0, 2 * (self.w1 - self.w0),
                                          self.w2 - 2 * self.w1 + self.w0])
                else:
                    top_wm = pol.Polynomial([self.w1, self.w2 - self.w1])
                    bot = pol.Polynomial([2 * self.w1, self.w2 - 2 * self.w1])
            else:
                if self.w0 != 0:
                    top_wm = pol.Polynomial([0, 0, self.w2])
                    bot = pol.Polynomial([self.w0, - 2 * self.w0,
                                          self.w2 + self.w0])
                else:
                    top_wm = pol.Polynomial([1])
                    bot = pol.Polynomial([1])
        else:
            # here w2 = 0
            if self.w1 != 0:
                if self.w0 != 0:
                    top_wm = pol.Polynomial([0, self.w1])
                    bot = pol.Polynomial([self.w0, 2 * self.w1 - self.w0])
                else:
                    top_wm = pol.Polynomial([1])
                    bot = pol.Polynomial([2])
            else:
                top_wm = pol.Polynomial([0])
                bot = pol.Polynomial([1])

        # the top of the fitness function with mutation
        top = (1 - u) * top_wm + v * (bot - top_wm)

        # bot is the same with and without mutations

        fitness = top(k_over_N) / bot(k_over_N)

        # some values are slightly greater than 1 or lower than 0 so the pmf
        # function is buggy, and returns nan.
        fitness[fitness > 1] = 1
        fitness[fitness < 0] = 0

        # fitness[i] is the probability to draw a derived allele in a size N
        # population with i individuals carriying the derived allele
        self.fitness = fitness

        # initializing simulations attributes
        self.times = []
        self.simu = []
        self.emis = []
        self.ssize = []

    # @property
    # def s(self):
    #     if self.mode == 'sh':
    #         return self.w2 - 1
    #     else:
    #         raise AttributeError

    # @property
    # def h(self):
    #     if self.mode == 'sh':
    #         if self.s != 0:
    #             return (self.w1 - 1) / (self.w2 - 1)
    #         else:
    #             return 0.5
    #     else:
    #         raise AttributeError

    def __eq__(self, other):
        """ the equality between two simulators is the equality of their
        parameters. These objects does not need to store the same simulations
        """

        if isinstance(other, WfSimulator):
            return (self.x0 == other.x0
                    and self.N == other.N
                    and self.u == other.u
                    and self.v == other.v
                    and self.w0 == other.w0
                    and self.w1 == other.w1
                    and self.w2 == other.w2
                    and self.mode == other.mode)
        else:
            return False

    def _true_eq(self, other):
        """ the true equality between two simulator is stronger than the simple
        equality : simulators need to have the same simulations
        """

        return (self == other
                and len(self.simu) == len(other.simu)
                and not (False in [np.all(self.simu[i] == other.simu[i])
                                   for i in range(len(self.simu))])
                and not (False in [np.all(self.times[i] == other.times[i])
                                   for i in range(len(self.times))])
                and not (False in [np.all(self.emis[i] == other.emis[i])
                                   for i in range(len(self.emis))])
                and not (False in [np.all(self.ssize[i] == other.ssize[i])
                                   for i in range(len(self.ssize))]))

    def __add__(self, other):
        """ merge simulations of two equals simulators, a third simulator
        is returned
        """

        if self == other:
            kwargs = {}
            if self.mode == 'w':
                kwargs['w0'] = self.w0
                kwargs['w1'] = self.w1
                kwargs['w2'] = self.w2
            elif self.mode == 'sh':
                kwargs['s'] = self.w2 - 1
                if self.w2 != 1:
                    kwargs['h'] = (self.w1 - 1) / (self.w2 - 1)
                else:
                    kwargs['h'] = 0
            elif self.mode == 's1s2':
                return NotImplementedError
            elif self.mode == 's0s2':
                return NotImplementedError
            else:
                return ValueError

            x0 = self.x0
            N = self.N
            u = self.u
            v = self.v
            
            new_sim = WfSimulator(x0=x0, N=N, u=u, v=v, **kwargs)

            # adding simulations and sampling of self and other
            new_sim.simu = self.simu + other.simu
            new_sim.times = self.times + other.times
            new_sim.emis = self.emis + other.emis
            new_sim.ssize = self.ssize + other.ssize
            return new_sim
        else:
            raise ValueError('You can only merge two simulators with same'
                             + ' parameters (check with sim1 == sim2)')

    def full_simu(self, n, store=True):
        """ simulating a full trajectory, n is the last generation to be
        simulated.
        This method returns a numpy array containing the frequency trajectory
        """

        # this array will contain the number of individual carrying the
        # derived allele.
        k = np.full(shape=(n + 1), fill_value=-1, dtype=int)
        k[0] = int(round(self.x0 * self.N))

        for gen in range(n):
            k[gen + 1] = sss.binom.rvs(n=self.N, p=self.fitness[k[gen]])
        if store:
            self.times.append(np.arange(n + 1))
            self.simu.append(k / self.N)
            self.emis.append(np.full(shape=(n + 1, ), fill_value=-1,
                                     dtype=int))
            self.ssize.append(np.full(shape=(n + 1, ), fill_value=-1,
                                      dtype=int))
        return k / self.N

    def emission(self, n, index=-1, times=False, store=True):
        """ this method simulate a sampling. self must have simu attribute
        n is a numpy array indicating the sample size
        index argument is the simulation in which sample (the last one by
        default)
        times is a numpy array of times in which sample (these times must have
        already been simulated)
        store is a boolean, set at True is the emission simulated has to be
        added to the emis attribute
        """

        # if sampling times are not given, do sampling on all simulated times
        if not np.any(times):
            times = self.times

        # check args
        assert len(times) == len(n)

        # getting the index corresponding to sampling times
        try:
            indexes = [np.where(self.times[index] == t)[0][0] for t in times]

        # the exception is raised if np.where(self.times == t) returns a void
        # tuple (, ) which mean that one of the times asked isn't simulated
        except IndexError as e:
            print('one of the times given is not simulated yet')
            raise

        # there is one index for each time given so the following must be true
        assert len(indexes) == len(times)

        # do simulations
        emis = sss.binom.rvs(n=n, p=self.simu[index][indexes])

        if store:
            self.emis[index][indexes] = emis
            self.ssize[index][indexes] = n

        return emis

    def make_short(self, hard = True):
        """ this method remove all simulated values which have not been sampled.
        this can be use typically after doing samples before exporting just to
        save memory and readability
        """

        # running the loop from the last element
        for i in range(len(self.simu) - 1, -1, -1):
            # getting the indexes sampled
            indexes = np.where(self.ssize[i] != -1)[0]

            # if there is no sampling
            if hard:
                if len(indexes) == 0:
                    # delete each corresponding part
                    del self.simu[i]
                    del self.times[i]
                    del self.emis[i]
                    del self.ssize[i]
                else:
                    # keeping only the corresponding simulations
                    self.simu[i] = self.simu[i][indexes]
                    self.times[i] = self.times[i][indexes]
                    self.emis[i] = self.emis[i][indexes]
                    self.ssize[i] = self.ssize[i][indexes]
            else:
                if len(indexes) != 0:
                    self.simu[i] = self.simu[i][indexes]
                    self.times[i] = self.times[i][indexes]
                    self.emis[i] = self.emis[i][indexes]
                    self.ssize[i] = self.ssize[i][indexes]

    def reset_emission(self):
        """ this function reset all simulated emissions"""

        self.emis = [np.full(shape=(len(self.simu[i]), ), fill_value=-1)
                     for i in range(len(self.simu))]
        self.ssize = [np.full(shape=(len(self.simu[i]), ), fill_value=-1)
                      for i in range(len(self.simu))]

    def reset_all(self):
        """ this function reset all simulations"""

        self.simu = []
        self.times = []
        self.emis = []
        self.ssize = []

    def export(self, name, mode='pic', short=False):
        """ this method exports the object into a file, the file format is
        specified by the mode argument. The short argument make data shorter
        (with a loss of non informative data)
        """

        if short:
            self.make_short()

        if mode == 'pic':
            self._export_pic(name)
        elif mode == 'csv':
            self._export_csv(name)
        else:
            raise NotImplementedError('the {m} format exportation is not '
                                      + 'supported yet'.format(m=mode))

    @staticmethod
    def load(name, mode='pic'):
        """ this method imports the object from a file. the file format is
        specified by the mode argument
        """

        if mode == 'pic':
            return WfSimulator._load_pic(name)
        elif mode == 'csv':
            return WfSimulator._load_csv(name)
        else:
            raise NotImplementedError('the {m} format importation is not '
                                      + 'supported yet'.format(m=mode))

    def _export_pic(self, name):
        """ this method exports the object into a picke file"""

        with open(name, 'wb') as f:
            pic.dump(self, f)

    @staticmethod
    def _load_pic(name):
        """ this method loads the object from a pickle file"""

        with open(name, 'rb') as f:
            return pic.load(f)

    def _export_csv(self, name):
        """ this method exports the object into a csv file"""

        with open(name, 'w') as f:
            # set the header
            head = 'ID,times,simu,emis,ssize,x0,N'
            if self.mode == 'w':
                head = head + ',w2,w1,w0'
            elif self.mode == 'sh':
                head = head + ',s,h'
            elif self.mode == 's1s2':
                head = head + ',s2,s1'
            elif self.mode == 's0s2':
                head = head + 's2,s0'
            else:
                raise NotImplementedError(self.mode + 'is not available for '
                                          + 'export')

            head = head + ',u,v\n'

            f.write(head)

            # for every simulation
            for sim in range(len(self.simu)):
                for i in range(len(self.simu[sim])):
                    # this list will contain one line
                    list_of_val = []
                    list_of_val.append(str(sim))
                    list_of_val.append(str(self.times[sim][i]))
                    list_of_val.append(str(self.simu[sim][i]))
                    list_of_val.append(str(self.emis[sim][i]))
                    list_of_val.append(str(self.ssize[sim][i]))
                    list_of_val.append(str(self.x0))
                    list_of_val.append(str(self.N))
                    if self.mode == 'w':
                        list_of_val.append(str(self.w2))
                        list_of_val.append(str(self.w1))
                        list_of_val.append(str(self.w0))
                    elif self.mode == 'sh':
                        list_of_val.append(str(self.s))
                        list_of_val.append(str(self.h))
                    elif self.mode == 's1s2':
                        list_of_val.append(str(self.w2 - 1))
                        list_of_val.append(str(self.w1 - 1))
                    elif self.mode == 's0s2':
                        list_of_val.append(str(self.w2 - 1))
                        list_of_val.append(str(self.w0 - 1))
                    else:
                        raise NotImplementedError

                    list_of_val.append(str(self.u))
                    list_of_val.append(str(self.v))
                    f.write(','.join(list_of_val) + '\n')

    @staticmethod
    def _load_csv(name):
        """ this method loads the object from a csv file"""

        with open(name, 'r') as f:
            # getting the header
            line = f.readline()
            head_elements = line.split(',')
            mode = ''.join(head_elements[7:-2])

            # getting the first line
            line = f.readline()

            # if the first line is empty return None
            if line == '':
                return None

            elements = line.split(',')
            x0 = float(elements[5])
            N = int(elements[6])
            u = float(elements[-2])
            v = float(elements[-1])
            sel_par = {}
            for i, key in enumerate(head_elements[7:-2]):
                sel_par[key] = float(elements[i + 7])

            new_simulator = WfSimulator(x0=x0, N=N, u=u, v=v, **sel_par)

            simu = [[float(elements[2])]]
            times = [[int(elements[1])]]
            emis = [[int(elements[3])]]
            ssize = [[int(elements[4])]]

            line = f.readline()

            while line != '':
                # getting values
                elements = line.split(',')

                simu_val = float(elements[2])
                times_val = int(elements[1])
                emis_val = int(elements[3])
                ssize_val = int(elements[4])

                ID = int(elements[0])

                # if we encounter a new ID, extend all lists
                if ID + 1 > len(simu):
                    simu.append([simu_val])
                    times.append([times_val])
                    emis.append([emis_val])
                    ssize.append([ssize_val])
                # otherwise extend the list corresponding to this ID
                else:
                    simu[ID].append(simu_val)
                    times[ID].append(times_val)
                    emis[ID].append(emis_val)
                    ssize[ID].append(ssize_val)

                line = f.readline()

            for i in range(len(simu)):
                new_simulator.simu.append(np.array(simu[i]))
                new_simulator.times.append(np.array(times[i]))
                new_simulator.emis.append(np.array(emis[i]))
                new_simulator.ssize.append(np.array(ssize[i]))

            return new_simulator


class WfSimulatorTimed(WfSimulator):
    """sub class of WfSimulator in which all trajectories are sampled for the
    same generations times
    """

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

    def emission(self, *args, times=False, **kwargs):

        if hasattr(self, 'frozen_times'):
            if np.all(self.frozen_times == times):
                super().emission(*args, times=times, **kwargs)
            else:
                raise ValueError('You can only sample at times you already ' +
                                 'sampled')
        else:
            self.frozen_times = times
            super().emission(*args, times=times, **kwargs)

#! /usr/bin/python3
# encoding: utf-8

import numpy as np
import scipy.stats as sss
import simulation_wf as wfs
import pickle as pic
import pandas as pd
import hmm

class Analyzer:
    """ rewriting of the Analyzer class"""

    def __init__(self):

        self.data = pd.DataFrame(columns=['ID_analyze',
                                          'ID_traj',
                                          'method',
                                          'MLE',
                                          'LMLE',
                                          'L0',
                                          'N',
                                          's',
                                          'h',
                                          'u',
                                          'v',
                                          'freq_ini',
                                          'x0',
                                          'x1',
                                          'xf',
                                          'y0',
                                          'y1',
                                          'yf',
                                          'traj_freq',
                                          'traj_times',
                                          'traj_emis',
                                          'traj_ssize'],
                                 )

    def __add__(self, other):
        return Analyzer.merge_analyzers(self, other, keep_analyze_ID = False)
        
    def merge_analyzers(self, other, keep_analyze_ID=False):


        new_ana = Analyzer()
        if len(self.data) == 0:
            new_ana.data = other.data.copy(deep=True)
            return new_ana
        if len(other.data) == 0:
            new_ana.data = self.data.copy(deep=True)
            return new_ana
        if keep_analyze_ID:
            new_ana.data = pd.concat([self.data, other.data],
                                     ignore_index = True)
        else:
            first = self._clean_ID()
            second = other._clean_ID()
            second.data['ID_analyze'] = (
                second.data['ID_analyze']
                + first.data['ID_analyze'][len(first.data) - 1]
                + 1)
            new_ana.data = pd.concat([first.data, second.data],
                                     ignore_index = True)

        return new_ana

    def _clean_ID(self):
        """ clean ID of one analyzer (basically set them to 0 - N) if there is
        missing indexes
        """

        if len(self.data) <= 0:
            return self
        
        new_ana = Analyzer()
        new_ana.data = self.data.copy(deep=True)
        max_ID = max(new_ana.data['ID_analyze'])

        curr_index = 0  # current index
        for i in range(max_ID + 1):
            if len(new_ana.data['ID_analyze'] == i) > 0:
                new_ana.data.loc[new_ana.data['ID_analyze'] == i,
                                 'ID_analyze'] = curr_index
                curr_index += 1

        return new_ana

    def analyze(self, srange, method, init, simulator, nop=129, begin = 0,
                end = -1):
        """ this method make HMM analysis using method and init as parameters
        for each value of srange.
        simulator is the simulator used to store trajectories and parameters
        """

        method_name = method + '-' + init

        number_of_simu = len(simulator.simu)
        if end == -1:
            true_end = number_of_simu
        else:
            true_end = end + 1
        
        simulator_index = 0
        if len(self.data) > 0:
            simulator_index = self.data['ID_analyze'][len(self.data) - 1] + 1


        # for each simulation done in the simulator, build the HMM and run
        # likelihood computation
        loop_bool = True
        for simu_index in range(begin, true_end):
            # rebuild the HMM object only if this is the 1st step or if
            # sampling times are changing
            if loop_bool or (np.any(simulator.times[simu_index]
                                    != simulator.times[simu_index - 1])):
                loop_bool = False
                hid_model = hmm.HMM(simulator.times[simu_index],
                                    model = method,
                                    init = init,
                                    h = np.array([simulator.h]),
                                    s = srange,
                                    N = np.array([simulator.N]),
                                    u = np.array([simulator.u]),
                                    v = np.array([simulator.v]),
                                    nop = nop)

            # running forward backward algorithm
            n = simulator.ssize[simu_index]

            E = hid_model._compute_emission(simulator.emis[simu_index], n)

            hid_model.forward(E, store = True, scale = True)
            hid_model.backward(E, store = True, scale = True,
                               index = hid_model.delta_t_size)

            L = (hid_model.likelihood(hid_model.alpha[-1, ],
                                      hid_model.beta[-1, ],
                                      hid_model.X[-1, ], log = True)
                 + np.sum(hid_model.beta_scale[-1:, ], axis = 0)
                 + np.sum(hid_model.alpha_scale[:, ], axis = 0))

            mask = L == np.inf
            index = -2
            num_time = len(simulator.times[simu_index])
            num_points = hid_model.X.shape[1]
            while np.any(mask) and abs(index) <= num_time:
                hid_model.backward(E, store = True, scale = True)
                L = (hid_model.likelihood(hid_model.alpha[index, ],
                                          hid_model.beta[index, ],
                                          hid_model.X[index, ], log = True)
                     + np.sum(hid_model.beta_scale[index:, ], axis = 0)
                     + np.sum(hid_model.alpha_scale[:(index + 1), ], axis = 0))
                mask = L == np.inf
                index -= 1
            # at the end of this loop, either there is still inf values but
            # we can't do anything, or there is not inf value anymore
            # if there is still inf values, set them to nan
            mask = mask.reshape(1, *mask.shape)
            mask = np.repeat(mask, num_points, axis = 0)
            L[mask[0]] = np.nan

            L = L.reshape((L.size, ))

            if 0 in srange:
                L0 = L[srange == 0][0]
            else:
                L0 = np.nan
            new_data_line = pd.Series([simulator_index,
                                       simu_index,
                                       method_name,
                                       srange[np.nanargmax(L)],
                                       np.nanmax(L),
                                       L0,
                                       simulator.N,
                                       simulator.s,
                                       simulator.h,
                                       simulator.u,
                                       simulator.v,
                                       simulator.x0,
                                       simulator.simu[simu_index][0],
                                       simulator.simu[simu_index][1],
                                       simulator.simu[simu_index][-1],
                                       (simulator.emis[simu_index][0]
                                        / simulator.ssize[simu_index][0]),
                                       (simulator.emis[simu_index][1]
                                        / simulator.ssize[simu_index][1]),
                                       (simulator.emis[simu_index][-1]
                                        / simulator.ssize[simu_index][-1]),
                                       simulator.simu[simu_index],
                                       simulator.times[simu_index],
                                       simulator.emis[simu_index],
                                       simulator.ssize[simu_index]],
                                      index = ['ID_analyze',
                                               'ID_traj',
                                               'method',
                                               'MLE',
                                               'LMLE',
                                               'L0',
                                               'N',
                                               's',
                                               'h',
                                               'u',
                                               'v',
                                               'freq_ini',
                                               'x0',
                                               'x1',
                                               'xf',
                                               'y0',
                                               'y1',
                                               'yf',
                                               'traj_freq',
                                               'traj_times',
                                               'traj_emis',
                                               'traj_ssize'])
            self.data = self.data.append(new_data_line, ignore_index = True)

        self.data['ID_analyze'] = self.data['ID_analyze'].astype(int)
        self.data['ID_traj'] = self.data['ID_traj'].astype(int)
        self.data['N'] = self.data['N'].astype(int)
                
                                 
    def export(self, name, mode='pic'):
        """ export the Analyzer object into the required format.
        name is a string indicating the file name the Analyzer object is
        exported into
        mode is a string indicating the exporting format
        - 'pic' for pickle
        - 'csv' for csv
        """

        if mode == 'pic':
            self._export_pic(name)
        elif mode == 'csv':
            self._export_csv(name)
        else:
            raise NotImplementedError

    @staticmethod
    def load(name, mode='pic'):
        """ load the Analyzer object from a file. Format file used is
        indicated by the mode argument : 'pic' for pickle format or 'csv' for
        csv format
        """

        if mode == 'pic':
            return Analyzer._load_pic(name)
        elif mode == 'csv':
            return Analyzer._load_csv(name)
        else:
            raise NotImplementedError

    def _export_pic(self, name):

        with open(name, 'wb') as f:
            pic.dump(self, f)

    def _export_csv(self, name):
        self.data.to_csv(name)

    @staticmethod
    def _load_pic(name):

        with open(name, 'rb') as f:
            return pic.load(f)

    @staticmethod
    def _load_csv(name):

        ana = Analyzer()
        with open(name, 'r') as f:
            ana.data = pd.read_csv(f, index_col=0)
        return ana

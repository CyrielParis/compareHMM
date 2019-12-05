#! /usr/bin/python3

import numpy as np
import scipy.stats as sst
import hmm
import pandas as pd

class RealDataAnalyzer:
    """Analyzer class made for analyzing real data"""

    def __init__(self):
        self.data = pd.DataFrame(columns = ['CHR', 'ID', 'method', 'sMLE',
                                            'NMLE',
                                            'LMLE', 'L0', 'CI_low', 'CI_up',
                                            'traj_emis', 'traj_ssize',
                                            'traj_times'],
                                 )
        self.hmm = None
        self.method_name = None

    def build_hmm(self, model, init, srange, Nrange, times, nop = 129):
        """Building the object hmm given parameters"""

        self.method_name = model + '-' + init

        self.hmm = hmm.HMM(times = times,
                           model = model,
                           init = init,
                           h = np.array([0.5]),
                           s = srange,
                           N = Nrange,
                           u = np.array([0]),
                           v = np.array([0]),
                           nop = nop)
                           

    def analyze(self, table, outfile):
        """analyze all trajectories from the table at the frq.strat form.
        This table must have the following columns :
        - CHR
        - SNP 
        - CLST indicates sampling time, must start from 0.
        - MAC indicates the number of derived allele found at this time and at 
        this locus.
        - NCHROBS indicates the total size of the corresponding sampling
        """

        with open(outfile, 'w') as f:
            f.write('CHR,ID,method,sMLE,NMLE,LMLE,L0,CI_low,'
                    + 'CI_up,traj_emis,traj_ssize,traj_times\n')
            for ID in table.SNP.unique():
                sub_table = table.loc[(table.SNP == ID)]
                CHR = np.unique(sub_table.CHR)[0]
                ssize = np.array(sub_table.NCHROBS)
                emis = np.array(sub_table.MAC)
                E = self.hmm._compute_emission(emis, ssize)
                self.hmm.forward(E, store = True, scale = True)
                self.hmm.backward(E, store = True, scale = True,
                                  index = self.hmm.delta_t_size)
                
                L = (self.hmm.likelihood(self.hmm.alpha[-1, ],
                                      self.hmm.beta[-1, ],
                                      self.hmm.X[-1, ], log = True)
                 + np.sum(self.hmm.beta_scale[-1:, ], axis = 0)
                 + np.sum(self.hmm.alpha_scale[:, ], axis = 0))

                mask = L == np.inf
                mask = mask.reshape(1, *mask.shape)
                num_points = self.hmm.X.shape[1]
                mask = np.repeat(mask, num_points, axis = 0)
                if np.any(mask):
                    print('WARNING: some of likelihoods are infinites')
                
                
                L[mask[0]] = np.nan
                L = L.reshape((L.size, ))
                if 0 in self.hmm.s:
                    L0 = L[self.hmm.s == 0][0]
                else:
                    L0 = np.nan


                f.write('{CHR},{ID},{meth},{sMLE},{NMLE},{LMLE},{L0},'.format(
                    CHR = CHR, ID = ID, meth = self.method_name,
                    sMLE = self.hmm.s[np.nanargmax(L)],
                    NMLE = np.unique(self.hmm.N),
                    LMLE = np.nanmax(L),
                    L0 = L0)
                        + '{CIl},{CIu},{te},{ts},{tt}\n'.format(
                            CIl = np.nan, CIu = np.nan, te = str(emis),
                            ts = str(ssize), tt = str(self.hmm.T)))
                
                        
                    
                new_data_line = pd.Series([CHR, ID, self.method_name,
                                           self.hmm.s[np.nanargmax(L)],
                                           np.unique(self.hmm.N),
                                           np.nanmax(L),
                                           L0, np.nan, np.nan,
                                           str(emis), str(ssize),
                                           str(self.hmm.T)],
                                          index = ['CHR', 'ID', 'method',
                                                   'sMLE',
                                                   'NMLE', 'LMLE', 'L0',
                                                   'CI_low','CI_up',
                                                   'traj_emis', 'traj_ssize',
                                                   'traj_times'])
                self.data = self.data.append(new_data_line, ignore_index = True)
                self.data['NMLE'] = self.data['NMLE'].astype(int)
                                                      

        

    @staticmethod
    def _build_traj_table_frq_strat(table):
        """Building the arguments for analyze methods from frq.strat data.
        input table must be under the frq.strat format, involving differents
        columns :
        - SNP 
        - CLST indicates sampling time, must start from 0.
        - MAC indicates the number of derived allele found at this time and at 
        this locus.
        - NCHROBS indicates the total size of the corresponding sampling
        """

        list_CLST = list(table.CLST.unique())
        list_CLST.sort()
        list_keys_emis = ['ID'] + ['Y' + str(i) for i in list_CLST]
        list_keys_ssize = ['ID'] + ['N' + str(i) for i in list_CLST]        
        emis_table = pd.DataFrame(columns = list_keys_emis)
        ssize_table = pd.DataFrame(columns = list_keys_ssize)
        return emis_table, ssize_table

class RDA(RealDataAnalyzer):
    """Just a shortcut name for the class"""
    pass

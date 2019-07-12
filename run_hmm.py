#encoding: utf-8

## Loading standard packages
import numpy as np
import pandas as pd

## Loading custom packages
from src import simulation_wf as wfs
from src import hmm
from src import analyzer as san


def analyze_one_simu():
    sim = wfs.WfSimulator.load('data/example_one_simulation.csv', mode = 'csv')
    method_name = 'spikedbeta-cunif'
    method = 'spikedbeta'
    init = 'cunif'
    nosr = 101
    nop = 65
    srange = np.concatenate(
        (np.linspace(-1, 0, nosr), np.linspace(0, 3, nosr)))
    srange = np.unique(srange)
    nos = len(sim.simu)

    loop_bool = True

    data_out = pd.DataFrame(
        columns = ['ID', 's', 'L', 'traj_times', 'traj_ssize', 'traj_emis',
                   'traj_simu'])
    
    for simu_index in range(nos):
        # rebuild the HMM object only if this is the 1st step or if
        # sampling times are changing
        if loop_bool or (np.any(sim.times[simu_index]
                                != sim.times[simu_index - 1])):
            loop_bool = False
            hid_model = hmm.HMM(sim.times[simu_index],
                                model = method,
                                init = init,
                                h = np.array([sim.h]),
                                s = srange,
                                N = np.array([sim.N]),
                                u = np.array([sim.u]),
                                v = np.array([sim.v]),
                                nop = nop)
            
        # running forward backward algorithm
        n = sim.ssize[simu_index]

        E = hid_model._compute_emission(sim.emis[simu_index], n)
        
        hid_model.forward(E, store = True, scale = True)
        hid_model.backward(E, store = True, scale = True,
                           index = hid_model.delta_t_size)
        
        L = (hid_model.likelihood(hid_model.alpha[-1, ],
                                  hid_model.beta[-1, ],
                                  hid_model.X[-1, ], log = True)
             + np.sum(hid_model.beta_scale[-1:, ], axis = 0)
             + np.sum(hid_model.alpha_scale[:, ], axis = 0))

        L = L.reshape((L.size, ))
        new_data_dict = {
            'ID': np.repeat(simu_index, len(srange)),
            's': srange,
            'L': L,
            'traj_times': np.repeat(str(sim.times[simu_index]), len(srange)),
            'traj_ssize': np.repeat(str(sim.ssize[simu_index]), len(srange)),
            'traj_emis': np.repeat(str(sim.emis[simu_index]), len(srange)),
            'traj_simu': np.repeat(str(sim.simu[simu_index]), len(srange)),
        }
        data_out = data_out.append(pd.DataFrame(new_data_dict),
                                   ignore_index = True)
    data_out.to_csv('analyzed/example_one_ana.csv')

        
def analyze_fixing_simu():

    method_name = 'spikedbeta-cunif'
    method = 'spikedbeta'
    init = 'cunif'
    nosr = 101
    nop = 65
    srange = np.concatenate(
        (np.linspace(-1, 0, nosr), np.linspace(0, 3, nosr)))
    srange = np.unique(srange)

    file_list = ['fixing_from_high', 'fixing_from_low',
                 'fixing_from_middle']

    for file_pref in file_list:
        sim = wfs.WfSimulator.load(
            'data/' + file_pref + '.csv', mode = 'csv')
        nos = len(sim.simu)

        loop_bool = True

        data_out = pd.DataFrame(
            columns = ['ID', 's', 'L', 'traj_times', 'traj_ssize', 'traj_emis',
                       'traj_simu'])
    
        for simu_index in range(nos):
            # rebuild the HMM object only if this is the 1st step or if
            # sampling times are changing
            if loop_bool or (np.any(sim.times[simu_index]
                                    != sim.times[simu_index - 1])):
                loop_bool = False
                hid_model = hmm.HMM(sim.times[simu_index],
                                    model = method,
                                    init = init,
                                    h = np.array([sim.h]),
                                    s = srange,
                                    N = np.array([sim.N]),
                                    u = np.array([sim.u]),
                                    v = np.array([sim.v]),
                                    nop = nop)
            
            # running forward backward algorithm
            n = sim.ssize[simu_index]
            
            E = hid_model._compute_emission(sim.emis[simu_index], n)
            
            hid_model.forward(E, store = True, scale = True)
            hid_model.backward(E, store = True, scale = True,
                               index = hid_model.delta_t_size)
            
            L = (hid_model.likelihood(hid_model.alpha[-1, ],
                                      hid_model.beta[-1, ],
                                      hid_model.X[-1, ], log = True)
                 + np.sum(hid_model.beta_scale[-1:, ], axis = 0)
                 + np.sum(hid_model.alpha_scale[:, ], axis = 0))
            
            L = L.reshape((L.size, ))
            new_data_dict = {
                'ID': np.repeat(simu_index, len(srange)),
                's': srange,
                'L': L,
                'traj_times': np.repeat(str(sim.times[simu_index]),
                                        len(srange)),
                'traj_ssize': np.repeat(str(sim.ssize[simu_index]),
                                        len(srange)),
                'traj_emis': np.repeat(str(sim.emis[simu_index]),
                                       len(srange)),
                'traj_simu': np.repeat(str(sim.simu[simu_index]),
                                       len(srange)),
            }
            data_out = data_out.append(pd.DataFrame(new_data_dict),
                                       ignore_index = True)
            data_out.to_csv('analyzed/' + file_pref + '_ana.csv')


def analyze_multiple_simu():
    nos = 101
    ana = san.Analyzer()
    sim = wfs.WfSimulator.load('data/multiple_simu.csv', mode = 'csv')

    srange = np.concatenate(
        (np.linspace(-1, 0, nos), np.linspace(0, 3, nos)))
    srange = np.unique(srange)

    ana.analyze(srange = srange, method = 'spikedbeta', init = 'cunif',
                simulator = sim, nop = 129)

    ana.export('analyzed/multiple_simu_ana.csv', mode = 'csv')

def analyze_real_dataset():
    pass



if __name__ == '__main__':
    analyze_one_simu()
    analyze_fixing_simu()
    analyze_multiple_simu()
    analyze_real_dataset()
            

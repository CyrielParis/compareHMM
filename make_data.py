#encoding: utf-8

## Loading standard packages
import numpy as np
import scipy.stats 

## Loading custom packages
from src import simulation_wf as wfs

### Making a simple simulation
def make_one_simu():
    nex = 3
    n = np.repeat(30, 10)
    times = 10 * np.arange(10)

    simulator = wfs.WfSimulator(x0 = 0.5, N = 100, u = 0, v = 0,
                                h = 0.5, s = 0)
    for _ in range(nex):
        simulator.full_simu(n = 100, store = True)
        simulator.emission(n = n, times = times, store = True)
    simulator.make_short()

    simulator.export('data/example_one_simulation.csv', mode = 'csv')

### Example of fixing in one step
def make_fixing_simu():
    n = np.repeat(30, 10)
    times = 10 * np.arange(10)

    ## Starting from middle
    simulator = wfs.WfSimulator(x0 = 0.5, N = 100, u = 0, v = 0,
                                h = 0.5, s = 0)
    simulator.full_simu(n = 100, store = True)
    simulator.emission(n = n, times = times, store = True)
    simulator.make_short()
    simulator.emis[-1] = np.array([15, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    simulator.simu[-1] = np.array([0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    simulator.full_simu(n = 100, store = True)
    simulator.emission(n = n, times = times, store = True)
    simulator.make_short()
    simulator.emis[-1] = np.array([15, 30, 30, 30, 30, 30, 30, 30, 30, 30])
    simulator.simu[-1] = np.array([0.5, 1, 1, 1, 1, 1, 1, 1, 1, 1])

    simulator.export('data/fixing_from_middle.csv', mode = 'csv')

    ## Starting from low frequency
    simulator = wfs.WfSimulator(x0 = 0.1, N = 100, u = 0, v = 0,
                                h = 0.5, s = 0)
    simulator.full_simu(n = 100, store = True)
    simulator.emission(n = n, times = times, store = True)
    simulator.make_short()
    simulator.emis[-1] = np.array([3, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    simulator.simu[-1] = np.array([0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    simulator.full_simu(n = 100, store = True)
    simulator.emission(n = n, times = times, store = True)
    simulator.make_short()
    simulator.emis[-1] = np.array([3, 30, 30, 30, 30, 30, 30, 30, 30, 30])
    simulator.simu[-1] = np.array([0.1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

    simulator.export('data/fixing_from_low.csv', mode = 'csv')

    ## Starting from high frequency
    simulator = wfs.WfSimulator(x0 = 0.9, N = 100, u = 0, v = 0,
                                h = 0.5, s = 0)
    simulator.full_simu(n = 100, store = True)
    simulator.emission(n = n, times = times, store = True)
    simulator.make_short()
    simulator.emis[-1] = np.array([27, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    simulator.simu[-1] = np.array([0.9, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    simulator.full_simu(n = 100, store = True)
    simulator.emission(n = n, times = times, store = True)
    simulator.make_short()
    simulator.emis[-1] = np.array([27, 30, 30, 30, 30, 30, 30, 30, 30, 30])
    simulator.simu[-1] = np.array([0.9, 1, 1, 1, 1, 1, 1, 1, 1, 1])

    simulator.export('data/fixing_from_high.csv', mode = 'csv')    

### Making multiple simu
def make_multiple_simu():
    nsim = 100
    ngen = 100
    n = np.repeat(30, 10)
    times = 10 * np.arange(10)

    simulator = wfs.WfSimulator(x0 = 0.5, N = 100, u = 0, v = 0,
                                h = 0.5, s = 0)
    for _ in range(nsim):
        simulator.full_simu(n = ngen, store = True)
        simulator.emission(n = n, times = times, store = True)

    simulator.make_short()
    simulator.export('data/multiple_simu.csv', mode = 'csv')

### Making real data
def make_real_dataset():
    pass


if __name__ == '__main__':
    np.random.seed(seed = 123456789)
    print('Making one random simulation...')
    make_one_simu()
    np.random.seed(seed = 123456789)
    print('Creating fixed trajectories...')
    make_fixing_simu()
    np.random.seed(seed = 123456789)
    print('Generating random simulation dataset...')
    make_multiple_simu()
    np.random.seed(seed = 123456789)
    print('Generating random real dataset...')
    make_real_dataset()

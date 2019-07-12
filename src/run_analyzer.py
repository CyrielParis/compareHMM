#! /usr/bin/python3
# encoding: utf-8

import analyzer as san
import simulation_wf as wfs
import numpy as np
import argparse as arg
import sys


# Options are :
# -s -h -u -v -N -x0 -m -i -n -t -o -srange

def print_args(f):
    def _f(*args, **kwargs):
        print('positional arguments :', *args)
        print('keyword arguments :', kwargs)
        return f(*args, **kwargs)
    return _f
    
def simulator_creation(x0, N, s, h, u, v, t, gen, n, ssize):
    sim = wfs.WfSimulatorTimed(x0 = x0, N = N, s = s, h = h, u = u, v = v)
    for _ in range(n):
        sim.full_simu(n = gen, store = True)
        sim.emission(n = ssize, times = t, store = True)

    sim.make_short()

    return sim
    
def analyzer_creation(input_file):
    if input_file == '':
        ana = san.Analyzer()
    else:
        ana = san.Analyzer.load(name = input_file, mode = 'csv')
 
    return ana
    

def main(x0, N, ssim, h, u, v, t, nsim, ssize, gen,
         method_list, init_list, srange, output_file, input_file = '',
         input_sim = '', nop = 129, ID_sim_beg = 0, ID_sim_end = -1):

    srange = np.array(srange)
    if input_sim == '':
        sim = simulator_creation(x0 = x0, N = N, s = ssim, h = h, u = u, v = v,
                                 t = np.array(t), gen = gen, n = nsim,
                                 ssize = np.array(ssize))
    else:
        sim = wfs.WfSimulatorTimed.load(input_sim, mode = 'pic')
    ana = analyzer_creation(input_file)
    for method, init in zip(method_list, init_list):
        ana.analyze(srange = srange, method = method, init = init,
                    simulator = sim, nop = nop, begin = ID_sim_beg,
                    end = ID_sim_end)
    ana.export(output_file + '.csv', mode = 'csv')

if __name__ == '__main__':
    parser = arg.ArgumentParser(description = '',
                                epilog = '',
                                add_help = True,
                                allow_abbrev = True)
    
    parser.add_argument('-x', '--freq', dest = 'x0', nargs = '?',
                        action = 'store', type = float, required = True,
                        help = ('The initial allele frequency, must be a '
                                + 'float in [0,1]'))
    parser.add_argument('-N', '--popSize', dest = 'N', nargs = '?',
                        action = 'store', type = int, required = True,
                        help = 'Effective haploid population size')
    parser.add_argument('--ssim', dest = 'ssim', nargs = '?', action = 'store',
                        type = float, default = 0,
                        help = 'Selection parameter used in simulations')
    parser.add_argument('-d', '--dominance', dest = 'h', nargs = '?',
                        action = 'store', type = float, default = 0.5,
                        help = 'Dominance parameter used in simulations')
    parser.add_argument('-u', dest = 'u', nargs = '?', action = 'store',
                        type = float, default = 0,
                        help = 'Mutation parameter')
    parser.add_argument('-v', dest = 'v', nargs = '?', action = 'store',
                        type = float, default = 0,
                        help = 'Mutation parameter')
    parser.add_argument('--nsim', dest = 'nsim', nargs = '?', action = 'store',
                        type = int, required = False,
                        help = 'Number of simulations to run')
    parser.add_argument('--gen', dest = 'gen', nargs = '?', action = 'store',
                        type = int, required = False,
                        help = 'The number of generation to simulate')
    parser.add_argument('-t', dest = 't', nargs = '+', action = 'store',
                        type = int, required = False,
                        help = 'Sampling dates (in generations)')
    parser.add_argument('--ssize', dest = 'ssize', nargs = '+',
                        action='store', type = int, required = False,
                        help = 'Sampling sizes')
    parser.add_argument('--model_list', dest = 'model_l', nargs = '+',
                        action = 'store', type = str, required = True,
                        help = ('List of transition model among spikedbeta,'
                                + ' beta, gauss or wrightfisher'))
    parser.add_argument('--init_list', dest = 'init_l', nargs = '+',
                        action = 'store', type = str, required = True,
                        help = ('Initial frequency distribution among '
                                + 'cunif for continuous uniform distribution '
                                + 'dunif for discrete uniform distribution '))
    parser.add_argument('--input_file', dest = 'input_file', nargs = 1,
                        action = 'store', type = str, default = [''],
                        required = False,
                        help = 'Unused argument')
    parser.add_argument('--input_sim', dest = 'input_sim', action = 'store',
                        type = str, default = '',
                        help = 'Simulation file to analyze')
    parser.add_argument('--output_file', dest = 'output_file', nargs = 1,
                        action = 'store', type = str, required = True,
                        help = 'Output filename')
    parser.add_argument('--srange', dest = 'srange', nargs = '+',
                        action = 'store', type = float, required = True,
                        help = ('s parameter grid in which compute the '
                                + 'likelihood'))
    parser.add_argument('--nop', dest = 'nop', action = 'store', type = int,
                        default = 129,
                        help = 'Number of integration point')
    parser.add_argument('--sim_beg', dest='ID_sim_beg', action = 'store',
                        default = 0, type = int,
                        help = 'index of the first simulation to analyze')
    parser.add_argument('--sim_end', dest='ID_sim_end', action = 'store',
                        default = -1, type = int,
                        help = 'index of the last simulation to analyze')

    args = parser.parse_args()

    main(args.x0, args.N, args.ssim, args.h, args.u, args.v, args.t, args.nsim,
         args.ssize, args.gen, args.model_l, args.init_l, args.srange,
         args.output_file[0], args.input_file[0], args.input_sim, args.nop,
         args.ID_sim_beg, args.ID_sim_end)

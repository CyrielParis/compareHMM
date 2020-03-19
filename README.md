# Presentation of the software

This software was designed to compare different transition distributions.

# Quick user guide

This section gives instructions to start quickly simulate Wright-Fisher
process, analyze it with a dedicated class but also analyze real dataset
at the plink frq.strat format. If only interested by the real data analysis,
jump directly to the corresponding part.

## Simulate a Wright-Fisher process under selection

The Wright-Fisher model used in these simulations is described in Mathematical
population genetics (Ewens, 2004) and in the article related to this work.
The script use to simulate this process is simulation_wf.py in the src
directory. It needs few classical python packages : numpy, scipy, pickle and
itertools. The simulator is designed with a python class. This class can
simulate the allele frequency evolution, a random sampling at differents times
and be exported to be plotted or analyzed.

These operations can be done in a python console or in a python script.
First, create a simulator object : 
wfs = simulation_wf.WfSimulator(x0 = 0.5, N = 100, s = 0.2, h = 0.5)

Then simulate the complete allele frequency evolution with the full_simu
method, indicating the total number of generation to simulate with the n
argument:
wfs.full_simu(n = 50)

You can simulate the sampling giving sampling sizes and sampling times by
giving to the method emission, the arguments n for the sample size and
times to give the generations sampling
wfs.emission(n = 30, times = numpy.array([0, 10, 25, 30]))

An optional method but saving memory is the make_short method, deleting any
simulation not sampled, keeping only times sampled with the emission method.
wfs.make_short()

Given you did al this, you can access to simulations by calling following
attributes :
- wfs.simu[-1] : the allele frequencies
- wfs.times[-1] : the corresponding times
- wfs.emis[-1] : the sampling simulations (-1 means no sampling simulated)
- wfs.ssize[-1] : the sampling sizes

Finally you can export this object either in classic csv format or in pickle
format :
- wfs.export('name_file', mod = 'csv')
- wfs.export('name_file', mod = 'pic')

If you need to load a previously stored object, use the load method :
- wfs = simulation_wf.load('name_file', mode = 'csv')
- wfs = simulation_wf.load('name_file', mode = 'pic')

## Run an analyze on simulations

## Run an analyze on real data set

### Input format

### Output format

### Run the script

# compareHMM
comparator of models to detect selection in HMM framework


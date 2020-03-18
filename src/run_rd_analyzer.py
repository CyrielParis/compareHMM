#encoding: utf-8

# Loading standard packages
import numpy as np
import pandas as pd
import argparse as arg

# Loading custom packages
import rd_analyzer as rda

def main(N, CHR, times, infile, outfile):
    df = pd.read_csv(infile, sep = '\s+')
    namefile = outfile
    if CHR != -1:
        df = df.loc[(df.CHR == CHR)]
        namefile = 'CHR_{CHR}_'.format(CHR = CHR) + namefile
    rd = rda.RDA()
    rd.build_hmm('spikedbeta', 'cunif',
                 srange = np.unique(np.hstack((
                     np.linspace(-1, 0, 100),
                     np.linspace(0, 1, 100)))),
                 Nrange = np.array([N]),
                 times = times, 
                 nop = 129)

    rd.analyze(table = df, outfile = namefile)

if __name__ == '__main__':
    parser = arg.ArgumentParser(
        description = '''

Software to analyze real data sets based on the frq.strat format. 
---------------------------------------------------------------------------
- The CHR column is a label to group SNPs (not necessarily the chromosome
        number). 
- The SNP column is an ID for each time serie to analyse (usually the SNP
        name).
- The CLST column is the generation corresponding to this sampling (the
        first one beeing zero). The order in the file must be the chronological
        order. 
- The A1, A2 and MAF columns are unused, but for one time series each
        sample must have the same reference allele (same A1 and A2).
- The MAC column is the counting of the allele at corresponding generation. 
- The NCHORBS column is the sampling size at this generation.
---------------------------------------------------------------------------''',
        formatter_class = arg.RawTextHelpFormatter)
    parser.add_argument('-N', dest = 'N', action = 'store', type = int,
                        help = "Effective population size (haploid)")
    parser.add_argument(
        '--CHR', dest = 'CHR', action = 'store', type = int,
        default = -1,
        help =
        "Analyze only the corresponding CHR in infile (optional)")
    parser.add_argument('--infile', dest = 'inf', action = 'store', type = str,
                        help = "The input .frq.strat file")
    parser.add_argument('--outfile', dest = 'ouf', action = 'store', type = str,
                        help =
                        "The output file name (existing file is overwritten)")
    parser.add_argument('--times', dest = 'times', action = 'store', type = int,
                        nargs = '+',
                        help =
                        "The list of generation sampled (must start with 0)")

    args = parser.parse_args()

    main(N = args.N, CHR = args.CHR, times = np.array(args.times),
         infile = args.inf, outfile = args.ouf)

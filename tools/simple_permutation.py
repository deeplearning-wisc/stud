# -*- coding: utf-8 -*-
import argparse
from tqdm import trange
import numpy as np
import itertools
from scipy.spatial.distance import cdist
import os

parser = argparse.ArgumentParser(description='Train network on Imagenet')
parser.add_argument('--classes', default=24, type=int,
                    help='Number of permutations to select')
parser.add_argument('--selection', default='max', type=str,
                    help='Sample selected per iteration based on hamming distance: [max] highest; [mean] average')
args = parser.parse_args()

if __name__ == "__main__":
    outname = 'permutations/permutations_hamming_%s_%d' % (
    args.selection, args.classes)
    os.makedirs(os.path.dirname(outname), exist_ok=True)

    P_hat = np.array(list(itertools.permutations(list(range(2)), 2)))
    np.save(outname, P_hat)
    print('file created --> ' + outname)

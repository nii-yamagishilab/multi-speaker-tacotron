# ==============================================================================
# Copyright (c) 2021, Yamagishi Laboratory, National Institute of Informatics
# Author: Erica Cooper (ecooper@nii.ac.jp)
# All rights reserved.
# ==============================================================================

# coding: utf-8
"""
Preprocess data for synthesis
usage: preprocess.py [options] <in_dir> <out_dir>

options:
    --hparams=<parmas>       Hyper parameters. [default: ].
    --gender=gender          Gender of the target speaker. F or M
    --speakerID=sid          Speaker ID of the target speaker.
    --source-only            Process source only.
    --target-only            Process target only.
    -h, --help               Show help message.

"""

import csv
import os
import numpy as np
import json
from pyspark import SparkContext
from docopt import docopt
from hparams import hparams, hparams_debug_string

if __name__ == "__main__":
    args = docopt(__doc__)
    in_dir = args["<in_dir>"]
    out_dir = args["<out_dir>"]
    gender = args["--gender"]
    speakerID = args["--speakerID"]
    source_only = args["--source-only"]
    target_only = args["--target-only"]

    hparams.parse(args["--hparams"])
    print(hparams_debug_string())

    if source_only:
        process_source = True
        process_target = False
    elif target_only:
        process_source = False
        process_target = True
    else:
        process_source = True
        process_target = True

    from datasets.synthesize_data import Synthesize
    instance = Synthesize(in_dir, out_dir, hparams, gender, speakerID)

    sc = SparkContext()

    record_rdd = sc.parallelize(instance.list_files())

    if process_source:
        keys = instance.process_sources(record_rdd).collect()

    if process_target:
        target_rdd = instance.process_targets(record_rdd)
        keys = target_rdd.keys()
        average = target_rdd.average()
        stddev = np.sqrt(target_rdd.moment2() - np.square(average))

    with open(os.path.join(out_dir, 'list.csv'), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for path in keys:
            writer.writerow([path])

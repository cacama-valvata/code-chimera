# This file is part of the Reproducible Open Benchmarks for Data Analysis
# Platform (ROB).
#
# Copyright (C) 2019 NYU.
#
# ROB is free software; you can redistribute it and/or modify it under the
# terms of the MIT License; see LICENSE file for more details.

"""Submission for the ROB Number Predictor Benchmark. Uses the difference
between last two values in a given sequenc to determine the result.
"""

from __future__ import absolute_import, division, print_function

import argparse
import errno
import os
import json
import sys


def predict(values):
    """Add difference between last two numbers to last number as the prediction
    result.

    Parameters
    ----------
    values: list(int)
        List of numbers

    Returns
    -------
    int
    """
    return values[-1] + (values[-1] - values[-2])


def main(inputfile, outputfile):
    """Read input sequence. For each sequence the predict() function is called
    to get the prediction result. Results are written to the output file.

    Parameters
    ----------
    inputfile: file
    outputfile: file
    """
    # Read input lines. Each line is expected to start with the sequence
    # identifier followed by ':' and a comma-separated list of values.
    inputs = dict()
    with open(inputfile, 'r') as f:
        for line in f:
            line = line.strip()
            id, values = line.split(':')
            inputs[int(id)] = [int(v) for v in values.split(',')]
    # Call the user-provided predict function for each input line to create
    # the results
    results = dict()
    for seq_id in inputs:
        results[seq_id] = predict(inputs[seq_id])
    # Ensure that output directory exists:
    # influenced by http://stackoverflow.com/a/12517490
    dir_name = os.path.dirname(outputfile)
    if dir_name != '':
        if not os.path.exists(dir_name):
            try:
                os.makedirs(dir_name)
            except OSError as exc:  # guard against race condition
                if exc.errno != errno.EEXIST:
                    raise
    # Write analytics results. The file format is sequence id ':' prediction.
    with open(outputfile, 'w') as f:
        for seq_id in results:
            f.write('{}:{}\n'.format(seq_id, results[seq_id]))


if __name__ == '__main__':
    args = sys.argv[1:]

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--inputfile", required=True)
    parser.add_argument("-o", "--outputfile", required=True)

    parsed_args = parser.parse_args(args)

    main(inputfile=parsed_args.inputfile, outputfile=parsed_args.outputfile)

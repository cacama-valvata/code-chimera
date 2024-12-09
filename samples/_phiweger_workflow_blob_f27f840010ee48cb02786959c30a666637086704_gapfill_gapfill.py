#!/usr/bin/env python3

'''gapfill

This tiny script takes as input two files, one some form of text and one yaml, 
replacing the keys in the text file with the values in the yaml.

Requires Python 3.X.

Usage: yamlexpand.py -i <infile> -o <outfile> -y <yaml>

Options:
  -h --help     Show this screen.
  -v --version  Show version.
  -i --infile   /path/to/infile
  -o --outfile  /path/to/outfile
  -y --yaml     /path/to/yaml 

'''

import yaml
# http://docopt.org/
from docopt import docopt

if __name__ == '__main__':
    arguments = docopt(__doc__, version='gapfill 0.1')
    # print(arguments)

# Import YAML file and parse to dict.
# http://stackoverflow.com/questions/1773805/how-can-i-parse-a-yaml-file
with open(arguments['<yaml>'], 'r') as stream:
    d = yaml.load(stream)

# Read in the markdown file, replace the keys, and save to a new file.
# http://stackoverflow.com/questions/4617034/how-can-i-open-multiple-files-using-with-open-in-python
with open(arguments['<infile>'], 'r') as infile, \
    open(arguments['<outfile>'], 'a+') as outfile:
    for line in infile:
        buffer = line
        for key, value in d.items():
            if '&' + key in buffer:
                buffer = buffer.replace('&' + str(key), str(value))
        # print(buffer, end='')
        outfile.write(buffer)

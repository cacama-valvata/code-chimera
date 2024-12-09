'''gapfill

This tiny script takes a template and a yaml, and will use the latter to 
fill fields in the former. This is coordinated through corresponding keys,
i.e. a "{{ variable }}" in the template will be filled with its value as 
specified in the yaml "variable: value" entry.

Requires Python 3.X if used with non-ASCII characters in the yaml.

Usage: gapfill.py -t <template> -o <outfile> -y <yaml>

Options:
  -h --help     Show this screen.
  -v --version  Show version.
  -t --template   /path/to/template
  -o --outfile  /path/to/outfile
  -y --yaml     /path/to/yaml 

'''

from docopt import docopt # http://docopt.org/
import yaml
import re
from jinja2 import Template
import sys

if __name__ == '__main__':
    arguments = docopt(__doc__, version='gapfill 0.1')

#                                                   ---------------------------
# Import YAML file into type dictionary.

# http://stackoverflow.com/questions/1773805/how-can-i-parse-a-yaml-file
with open(arguments['<yaml>'], 'r') as stream:
    d = yaml.load(stream)

# load markdown file as string and transform into a template like
with open(arguments['<template>'], 'r') as file:
    s = file.read()
template = Template(s)

#                                                   ---------------------------
# Check that all fields in the template have a corresponding yaml entry. 

# regexp search for placeholder {{ variable }} or {{ index[0] }}
# http://stackoverflow.com/questions/4697882/how-can-i-find-all-matches-to-a-regular-expression-in-python
m = set(re.findall('\{\{\s?(\w+).*?\}\}', s))
# \{ .. curly brackets need escaping
# \s? .. a space may or may not be present
# (\w+) .. a word is put in a group and will thus be the only thing printed
# .* followed by more signs (a space, brackets, ...)
# ? .. non-greedy matching 

# compare keys present in yaml and template
diff = m.difference(d.keys())
if diff: # if set is not empty, i.e. bool({}) evaluates to False
    # http://stackoverflow.com/questions/73663/terminating-a-python-script
    sys.exit('missing key(s) in yaml: ' + ', '.join(diff))

#                                                   ---------------------------
# Write to file.
with open(arguments['<outfile>'], 'w+') as outfile:
    outfile.write(template.render(d))







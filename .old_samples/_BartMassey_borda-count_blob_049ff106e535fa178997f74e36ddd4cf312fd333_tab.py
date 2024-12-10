#!/usr/bin/python3
# Copyright (c) 2013 Bart Massey

# Tabulate Borda Count votes.

from sys import stdin

votes = { c:0 for c in 'abcdef-' }

for line in stdin:
    for i, c in enumerate(line.rstrip()):
        votes[c] += 6 - i

# http://stackoverflow.com/questions/613183
for k in sorted(votes, key=votes.get, reverse=True):
    if k != '-':
        print('{}: {}'.format(k, votes[k]))

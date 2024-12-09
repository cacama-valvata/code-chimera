#!/usr/bin/python3
# Copyright (c) 2013 Bart Massey
# Tabulate Borda Count votes

from sys import stdin

votes = {}
for c in "abcdef-":
    votes[c] = 0

for line in stdin:
    v = 6
    for c in line.rstrip():
        votes[c] += v
        v -= 1

# http://stackoverflow.com/questions/613183
for k in sorted(votes, key=votes.get, reverse=True):
    if k == '-':
        continue
    print("%c: %d" % (k, votes[k]))

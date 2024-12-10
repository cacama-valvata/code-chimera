#!/usr/bin/env python3
# https://adventofcode.com/2023/day/1
import re

with open("input.txt", "r", encoding="utf8") as f:
    lines = f.read().splitlines()


def get_cals(lines):
    return sum([int(v[0] + v[-1]) for v in [[c for c in l if c.isdigit()] for l in lines]])


print(f"Part 1: {get_cals(lines)}")

# Unique starting and ending chars:
# e f n o r s t x
# Starting and ending chars appearing at least twice:
# e f n o s t
n_map = {
    "one": "o1e",
    "two": "t2o",
    "three": "t3e",
    "four": "f4",
    "five": "f5e",
    "six": "s6",
    "seven": "s7n",
    "eight": "e8t",
    "nine": "n9e",
}

# Method 1: Regex (lookaheads + substitution)

# See https://stackoverflow.com/a/5616910
# exp = re.compile(f"(?=({'|'.join(n_map.keys())}))")
# lines = [re.sub(exp, lambda n: n_map[str(n.group(1))], l) for l in lines]

# Method 2: Replace

for i in range(len(lines)):
    for n in n_map.keys():
        lines[i] = lines[i].replace(n, n_map[n])

print(f"Part 2: {get_cals(lines)}")

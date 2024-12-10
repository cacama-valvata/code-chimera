#!/usr/bin/env python3

import re
from operator import itemgetter


def word_counts(text):
    """
    Search words in a given text.
    Adapted from Python 2 PyTagCloud counter.py get_word_counts
    https://github.com/atizo/PyTagCloud
    """
    # https://docs.python.org/3/howto/regex.html
    words = map(lambda x: x.lower(), re.findall(r'\w+', text, re.ASCII))

    counted = {}

    for word in words:
        if len(word) > 1:
            if word in counted:
                counted[word] += 1
            else:
                counted[word] = 1

    # sort by word count. itemgetter(1) is dictionary key:value pair value
    return sorted(counted.items(), key=itemgetter(1), reverse=True)


def word_counts_from_file(filename):
    """
    read input_filename as a single string
    returns word_counts
    """

    # read text_file to a string
    # throws UnicodeDecodeError: 'utf-8' codec can't decode byte
    textfile = open(filename, 'r', encoding='utf-8')
    # text is a string containing the entire text file
    text = textfile.read()
    textfile.close()

    return word_counts(text)


def write_word_counts(input_filename, output_filename):
    """
    reads input_filename as a single string
    writes to output_filename
    """

    counts = word_counts_from_file(input_filename)

    # write results to file
    with open(output_filename, 'w') as outfile:
        for item in counts:
            # use print instead of write
            # https://stackoverflow.com/questions/6159900/correct-way-to-write-line-to-file-in-python#6159912
            print(str(item), file=outfile)


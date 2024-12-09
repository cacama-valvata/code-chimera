import pandas as pd
import argparse
import os.path

from utils import format_cols
from search import feature_search


def is_valid_file(parser, arg):
    if not os.path.exists(arg):
        parser.error("The file %s does not exist!" % arg)
    else:
        return pd.read_csv('Ver_2_CS170_Fall_2021_Small_data__65.txt', header=None, delim_whitespace=True)  # return an open file handle




if __name__ == "__main__": 
    # https://stackoverflow.com/questions/20063/whats-the-best-way-to-parse-command-line-arguments

    parser = argparse.ArgumentParser(description='Program that tests different features to see which one works the best with K-nearest neighbors.', 
                                    formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('input_file', type=str,
                        help='Required: Pass in .txt from Professor Keogh')


    args = parser.parse_args()
    df = is_valid_file(parser, args.input_file)
    cols = format_cols(df)
    feature_search(df=df, cols=cols)
"""Utility to load datasets."""
import pandas as pd
import os


def load_webkb():
    filename = os.path.join(os.path.dirname(__file__), "text/webkb-train-stemmed.txt")
    train = pd.read_csv(filename, header=None, sep='\t', index_col=0).dropna()
    train.columns = ['words']

    filename = os.path.join(os.path.dirname(__file__), "text/webkb-test-stemmed.txt")
    test = pd.read_csv(filename, header=None, sep='\t', index_col=0).dropna()
    test.columns = ['words']

    return train, test

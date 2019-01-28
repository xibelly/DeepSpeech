from __future__ import absolute_import, division, print_function

import codecs
import re

import numpy as np

from six.moves import range

class Alphabet(object):
    def __init__(self, config_file):
        self._config_file = config_file
        self._size = 256

    def string_from_label(self, label):
        assert False

    def label_from_string(self, string):
        assert False

    def decode(self, labels):
        return bytes(labels).decode('utf-8', errors='replace')

    def size(self):
        return self._size

    def config_file(self):
        return self._config_file


def text_to_char_array(original):
    r"""
    Given a Python string ``original``, remove unsupported characters, map characters
    to integers and return a numpy array representing the processed string.
    """
    return np.frombuffer(original.encode('utf-8'), np.uint8).astype(np.int32)


# The following code is from: http://hetland.org/coding/python/levenshtein.py

# This is a straightforward implementation of a well-known algorithm, and thus
# probably shouldn't be covered by copyright to begin with. But in case it is,
# the author (Magnus Lie Hetland) has, to the extent possible under law,
# dedicated all copyright and related and neighboring rights to this software
# to the public domain worldwide, by distributing it under the CC0 license,
# version 1.0. This software is distributed without any warranty. For more
# information, see <http://creativecommons.org/publicdomain/zero/1.0>

def levenshtein(a, b):
    "Calculates the Levenshtein distance between a and b."
    n, m = len(a), len(b)
    if n > m:
        # Make sure n <= m, to use O(min(n,m)) space
        a, b = b, a
        n, m = m, n

    current = list(range(n+1))
    for i in range(1, m+1):
        previous, current = current, [i]+[0]*n
        for j in range(1, n+1):
            add, delete = previous[j]+1, current[j-1]+1
            change = previous[j-1]
            if a[j-1] != b[i-1]:
                change = change + 1
            current[j] = min(add, delete, change)

    return current[n]

# Validate and normalize transcriptions. Returns a cleaned version of the label
# or None if it's invalid.
def validate_label(label):
    # For now we can only handle [a-z ']
    if re.search(r"[0-9]|[(<\[\]&*{]", label) is not None:
        return None

    label = label.replace("-", "")
    label = label.replace("_", "")
    label = label.replace(".", "")
    label = label.replace(",", "")
    label = label.replace("?", "")
    label = label.replace("\"", "")
    label = label.strip()
    label = label.lower()

    return label if label else None

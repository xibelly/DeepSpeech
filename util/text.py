from __future__ import absolute_import, division, print_function

import codecs
import re

import numpy as np

from six.moves import range

class Alphabet(object):
    def __init__(self, _):
        pass

    @staticmethod
    def string_from_label(_):
        assert False

    @staticmethod
    def label_from_string(_):
        assert False

    @staticmethod
    def decode(labels):
        return bytes(np.asarray(labels, np.uint8) + 1).decode('utf-8', errors='replace')
        # return bytes(labels)

    @staticmethod
    def size():
        return 255

    @staticmethod
    def config_file():
        return ''


def text_to_char_array(series):
    r"""
    Given a Pandas Series containing transcript string, map characters to
    integers and return a numpy array representing the processed string.
    """
    try:
        series['transcript'] = np.frombuffer(series['transcript'].replace(' ', '').encode('utf-8'), np.uint8).astype(np.int32) - 1
    except KeyError as e:
        # Provide the row context (especially wav_filename) for alphabet errors
        raise ValueError(str(e), series)

    if series['transcript'].shape[0] == 0:
        raise ValueError("Found an empty transcript! You must include a transcript for all training data.", series)

    return series


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

    label = label.replace("-", " ")
    label = label.replace("_", " ")
    label = re.sub("[ ]{2,}", " ", label)
    label = label.replace(".", "")
    label = label.replace(",", "")
    label = label.replace("?", "")
    label = label.replace("\"", "")
    label = label.strip()
    label = label.lower()

    return label if label else None

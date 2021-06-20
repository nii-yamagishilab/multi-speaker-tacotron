# Copyright (c) 2017 Keith Ito
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
# ==============================================================================
# Copyright (c) 2018, Yamagishi Laboratory, National Institute of Informatics
# Author: Yusuke Yasuda (yasuda@nii.ac.jp)
# All rights reserved.
# ==============================================================================
"""  """

from datasets.cleaners import english_cleaners

_characters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz!\'\"()[],-.:;?` '

symbols = list(_characters)

# Mappings from symbol to numeric ID and vice versa:
# Reserve 0 for silence
_symbol_to_id = {s: i + 1 for i, s in enumerate(symbols)}
_id_to_symbol = {i + 1: s for i, s in enumerate(symbols)}


def text_to_sequence(text, cleaner):
    clean_text = cleaner(text)
    return _symbols_to_sequence(clean_text), clean_text


def _symbols_to_sequence(symbols):
    # Add leading and trailing silence symbols
    # skip symbols that don't convert
    result = [0]
    for s in symbols:
        try:
            result.append(_symbol_to_id[s])
        except:
            continue
    result += [0]
    return result

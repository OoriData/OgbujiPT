# SPDX-FileCopyrightText: 2023-present Uche Ogbuji <uche@ogbuji.net>
# SPDX-License-Identifier: Apache-2.0
# ogbujipt.word_loom

'''
Routines to help with processing text in the word loom convention:
https://github.com/OoriData/OgbujiPT/wiki/Word-Loom:-A-format-for-managing-language-for-AI-LLMs-(including-prompts)
'''

import tomli

def read(fp):
    '''
    Read a word loom and return the tables as top-level result mapping
    Just a shallow wrapper around tomli.load(), for now. Might do more processing later.

    >>> with open("myprompts.toml", mode="rb") as fp:
    >>>     loom = tomli.load(fp)
    '''
    loom = tomli.load(fp)
    return loom

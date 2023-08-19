# SPDX-FileCopyrightText: 2023-present Uche Ogbuji <uche@ogbuji.net>
# SPDX-License-Identifier: Apache-2.0
# ogbujipt.word_loom

'''
Routines to help with processing text in the word loom convention:
https://github.com/OoriData/OgbujiPT/wiki/Word-Loom:-A-format-for-managing-language-for-AI-LLMs-(including-prompts)
'''

import tomli

# XXX Defaulting to en leaves a bit too imperialist a flavor, really
def load(fp, lang='en'):
    '''
    Read a word loom and return the tables as top-level result mapping
    Loads the TOML, then selects text by given language

    >>> from ogbujipt import word_loom
    >>> with open('myprompts.toml', mode='rb') as fp:
    >>>     loom = word_loom.load(fp)
    '''
    loom_raw = tomli.load(fp)
    # Select text by language
    # FIXME: Only top level, for now. Presumably we'll want to support scoping
    texts = {}
    default_lang = loom_raw.get('lang', None)
    for k, v in loom_raw.items():
        if not isinstance(v, dict):
            # Skip top-level items
            continue
        val = v.get(lang)
        if not v and lang == default_lang:
            val = v.get('text')
        texts[k] = val
    return texts

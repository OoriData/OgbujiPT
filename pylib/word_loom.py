# SPDX-FileCopyrightText: 2023-present Oori Data <info@oori.dev>
# SPDX-License-Identifier: Apache-2.0
# ogbujipt.word_loom

'''
Routines to help with processing text in the word loom convention:
https://github.com/OoriData/OgbujiPT/wiki/Word-Loom:-A-format-for-managing-language-for-AI-LLMs-(including-prompts)
'''

import io
import tomli


class text_item(str):
    '''
    Text or template for use with LLM tools
    Largely keeps metadata around language, template markers, etc.

    >>> from ogbujipt.word_loom import T
    >>> t = T('spam', lang='en)
    'spam'
    >>> t.lang
    'en'
    '''

    def __new__(cls, value, lang, markers=None):
        assert isinstance(value, str)
        self = super(text_item, cls).__new__(cls, value)
        self.lang = lang
        self.markers = markers
        return self

    def __repr__(self):
        return u'T(' + str(self) + ')'


T = text_item


# XXX Defaulting to en leaves a bit too imperialist a flavor, really
def load(fp_or_str, lang='en'):
    '''
    Read a word loom and return the tables as top-level result mapping
    Loads the TOML, then selects text by given language

    fp_or_str - file-like object or string containing TOML
    lang - select oly texts in this language (default: 'en')

    >>> from ogbujipt import word_loom
    >>> with open('myprompts.toml', mode='rb') as fp:
    >>>     loom = word_loom.load(fp)
    '''
    # Ensure we have a file-like object
    if isinstance(fp_or_str, str):
        fp_or_str = io.BytesIO(fp_or_str.encode('utf-8'))
    elif isinstance(fp_or_str, bytes):
        fp_or_str = io.BytesIO(fp_or_str)
    # Load TOML
    loom_raw = tomli.load(fp_or_str)
    # Select text by language
    # FIXME: Only top level, for now. Presumably we'll want to support scoping
    texts = {}
    default_lang = loom_raw.get('lang', None)
    for k, v in loom_raw.items():
        if not isinstance(v, dict) or 'text' not in v:
            # Skip top-level items
            continue
        if v.get('lang') == lang or ('lang' not in v and lang == default_lang):
            texts[k] = T(v['text'], lang, markers=v.get('markers'))
    return texts

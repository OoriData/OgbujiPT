# SPDX-FileCopyrightText: 2023-present Oori Data <info@oori.dev>
# SPDX-License-Identifier: Apache-2.0
# ogbujipt.word_loom

'''
Routines to help with processing text in the word loom convention:
https://github.com/OoriData/OgbujiPT/wiki/Word-Loom:-A-format-for-managing-language-for-AI-LLMs-(including-prompts)
'''

import io
import tomli
import warnings


class text_item(str):
    '''
    Text or template for use with LLM tools
    Largely keeps metadata around language, template markers, etc.

    >>> from ogbujipt.word_loom import T
    >>> t = T('spam', lang='en')
    >>> t
    'spam'
    >>> t.lang
    'en'
    >>> t = T('spam', lang='en', altlang={'fr': 'jambon'})
    >>> t.altlang['fr']
    'jambon'
    '''

    def __new__(cls, value, deflang, altlang=None, meta=None, markers=None):
        assert isinstance(value, str)
        self = super(text_item, cls).__new__(cls, value)
        self.lang = deflang  # Default language
        self.meta = meta or {}
        self.markers = markers or {}
        self.altlang = altlang or {}
        return self

    def __repr__(self):
        return u'T(' + repr(str(self)) + ')'

    def in_lang(self, lang):
        return self.altlang.get(lang)

T = text_item


# XXX Defaulting to en leaves a bit too imperialist a flavor, really
def load(fp_or_str, lang='en'):
    '''
    Read a word loom and return the tables as top-level result mapping
    Loads the TOML, then selects text by given language

    fp_or_str - file-like object or string containing TOML
    lang - select oly texts in this language (default: 'en')

    >>> from ogbujipt import word_loom
    >>> with open('demo/language.toml', mode='rb') as fp:
    >>>     loom = word_loom.load(fp)
    >>> loom['test_prompt_joke'].meta
    {'tag': 'humor', 'updated': '2024-01-01'}
    >>> str(loom['test_prompt_joke'])
    'Tell me a funny joke about {topic}\n'
    >>> loom['test_prompt_joke'].in_lang('fr')
    'Dites-moi une blague drôle sur {topic}\n'
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
        if not isinstance(v, dict):
            # Skip top-level items
            continue
        if 'text' in v:
            warnings.warn('Deprecated attribute "text". Use "_" instead')
            text = v['text']
        else:
            text = v.get('_')
        if text is None: # Skip items without text
            continue
        markers = v.get('_m')
        if 'markers' in v:
            warnings.warn('Deprecated attribute "marker". Use "_m" instead')
            markers = v['markers']
        else:
            markers = v.get('_m')
        if v.get('lang') == lang or ('lang' not in v and lang == default_lang):
            altlang = {kk.lstrip('_'): vv for kk, vv in v.items() if (kk.startswith('_') and kk not in ('_', '_m'))}
            meta = {kk: vv for kk, vv in v.items() if (not kk.startswith('_') and kk not in ('text', 'markers'))}
            texts[k] = T(text, lang, altlang=altlang, meta=meta, markers=markers)
    return texts

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


class language_item(str):
    '''
    Text or template for use with LLM tools
    Largely keeps metadata around language, template markers, etc.

    >>> from ogbujipt.word_loom import T
    >>> t = T('spam', deflang='en')
    >>> t
    'spam'
    >>> t.lang
    'en'
    >>> t = T('spam', deflang='en', altlang={'fr': 'jambon'})
    >>> t.altlang['fr']
    'jambon'
    '''

    def __new__(cls, value, deflang, altlang=None, meta=None, markers=None):
        '''
        Construct a new text item

        value - text value in the default language
        deflang - default language - made mandatory to avoid sloppy language assumptions
        altlang - dictionary of text values in alternative languages
        meta - dictionary of metadata
        markers - used to specify values that can be set, with the text value is treated as a template
        '''
        assert isinstance(value, str)
        self = super(language_item, cls).__new__(cls, value)
        self.lang = deflang  # Default language
        self.meta = meta or {}
        self.markers = markers or {}
        self.altlang = altlang or {}
        return self

    def __repr__(self):
        return u'T(' + repr(str(self)) + ')'

    def in_lang(self, lang):
        return self.altlang.get(lang)

    def clone(self, value=None, deflang=None, altlang=None, meta=None, markers=None):
        '''
        Clone the text item, with optional overrides

        >>> from ogbujipt.word_loom import T
        >>> t = T('spam', deflang='en', meta={'tag': 'food'})
        >>> t, t.meta
        'spam', {'tag': 'food'}
        >>> t_cloned = t.clone(meta={'tag': 'protein'})
        >>> t_cloned, t_cloned.meta
        'spam', {'tag': 'protein'}
        >>> t_cloned = t.clone('eggs')
        >>> t_cloned, t_cloned.meta
        'spam', {'tag': 'food'}
        '''
        value = str(self) if value is None else value
        deflang = self.lang if deflang is None else deflang
        altlang = self.altlang if altlang is None else altlang
        meta = self.meta if meta is None else meta
        markers = self.markers if markers is None else markers
        return language_item(value, deflang, altlang=altlang, meta=meta, markers=markers)


# Following 2 lines are deprecated
T = language_item
text_item = language_item

LI = language_item  # Alias for language_item


# XXX Defaulting to en leaves a bit too imperialist a flavor, really
def load(fp_or_str, lang='en', preserve_key=False):
    '''
    Read a word loom and return the tables as top-level result mapping
    Loads the TOML

    Return a dict of the language items, indexed by the TOML key as well as its default language text

    fp_or_str - file-like object or string containing TOML
    lang - select oly texts in this language (default: 'en')
    preserve_key - if True, the key in the TOML is preserved in each item's metadata

    >>> from ogbujipt import word_loom
    >>> with open('demo/language.toml', mode='rb') as fp:
    >>>     loom = word_loom.load(fp)
    >>> loom['test_prompt_joke'].meta
    {'tag': 'humor', 'updated': '2024-01-01'}
    >>> actual_text = loom['test_prompt_joke']
    >>> str(actual_text)
    'Tell me a funny joke about {topic}\n'
    >>> str(loom[str(actual_text)])
    'Tell me a funny joke about {topic}\n'
    >>> loom[str(actual_text)].in_lang('fr')
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
            if preserve_key:
                meta['_key'] = k
            if k in texts:
                warnings.warn(f'Key {k} duplicates an existing item, which will be overwritten')
            texts[k] = T(text, lang, altlang=altlang, meta=meta, markers=markers)
            # Also index by literal text
            if text in texts:
                warnings.warn(
                    f'Item default language text {text[:20]} duplicates an existing item, which will be overwritten')
            texts[text] = T(text, lang, altlang=altlang, meta=meta, markers=markers)
    return texts

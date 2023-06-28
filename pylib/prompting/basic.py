# SPDX-FileCopyrightText: 2023-present Uche Ogbuji <uche@ogbuji.net>
#
# SPDX-License-Identifier: Apache-2.0
# ogbujipt.prompting.basic

'''
Routines for building prompts for LLM query

Can be used instead of, e.g. some of the more basic langchain chains,
but designed to be much more transparent

Also, trying to avoid more complex templating such as Jinja or Mako
'''

from enum import Enum


class pdelim(Enum):
    '''
    Types of delimiters used in prompts. Prompt delimiters can be anything
    from simple spacing/formatting aids to control text such as the
    '### User:\n' but in Alpaca-style prompting, for example
    '''
    PREAMBLE = 1  # Post preamble
    INTERCONTEXT = 2  # Between multiple context sections
    PREQUERY = 3  # e.g. '### User:\n\n' in Alapaca
    POSTQUERY = 4  # e.g. '### Assistant:\n\n' in Alapaca


def context_build(query, preamble='', contexts=None, delimiters=None):
    '''
    Build a full LLM prompt out of the actual human/user query, an optional
    preamble (e.g. system message to condition the LLM's responses),
    zero or more context items (either simple text or tuples of text/metadata
    dict), and optional delimiters

    >>> from ogbujipt.prompting.basic import context_build
    >>> context_build('How are you?', preamble='You are a friendly AI who loves conversation') 
'You are a friendly AI who loves conversation\n\nHow are you?\n'
    '''
    contexts = contexts or []
    delimiters = delimiters or {}
    parts = [preamble] if preamble + delimiters.get(
        pdelim.PREAMBLE, '\n') else []
    if contexts:
        for c in contexts:
            parts.append(str(c))
            parts.append(delimiters.get(pdelim.INTERCONTEXT, '\n'))
        del parts[-1]  # Final intercontext not needed
    parts.append(delimiters.get(pdelim.PREAMBLE, ''))
    parts.append(query)
    parts.append(delimiters.get(pdelim.POSTQUERY, ''))
    full_context = '\n'.join(parts)
    return full_context

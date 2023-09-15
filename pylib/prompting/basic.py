# SPDX-FileCopyrightText: 2023-present Oori Data <info@oori.dev>
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
    '### Instruction:\n' but in Alpaca-style prompting, for example
    '''
    FIXED_PREAMBLE = 1  # Preamble that's always set
    PRE_PREAMBLE = 2  # Set as prefix to a preamble (e.g. ### System:)
    POST_PREAMBLE = 3  # Set as prefix to a preamble (e.g. ### System:)
    INTERCONTEXT = 4  # Between multiple context sections
    PREQUERY = 5  # e.g. '### Instruction:\n' in Alpaca
    POSTQUERY = 6  # e.g. '### Response:\n' in Alpaca
    PRECONTEXT = 7  # e.g. '### Input:\n' in Alpaca instruct variation
    POSTCONTEXT = 8
    PRE_ALL_CONTEXT = 9
    POST_ALL_CONTEXT = 10
    PRE_CONTEXT_METADATA = 11   #Mostly for Airoboros (BEGINCONTEXT [..] ENDCONTEXT) format
    POST_CONTEXT_METADATA = 12  #"       "
    META_ORDERING = 100


class ordering(Enum):
    '''
    Ordering of complex prompt compoents. Of course preamble always comes 1st
    '''
    QUERY_CONTEXT = 1
    CONTEXT_QUERY = 2


def format(query, preamble='', contexts=None, delimiters=None, context_with_metadata=False):
    '''
    Build a full LLM prompt out of the actual human/user query, an optional
    preamble (e.g. system message to condition the LLM's responses),
    zero or more context items (either simple text or tuples of text/metadata
    dict), and optional delimiters

    When context_with_metadata is True, this will assume contexts is a list of 2-item tuple of strings, where
    the first item is the metadata for the context (a string) and the second item is the context string itself

    >>> from ogbujipt.prompting.basic import format
    >>> format('How are you?', preamble='You are a friendly AI who loves conversation') 
    'You are a friendly AI who loves conversation\n\nHow are you?\n'
    '''
    contexts = contexts or []
    if isinstance(contexts, str):
        contexts = [contexts]
    delimiters = delimiters or {}

    parts = []
    fixed = delimiters.get(pdelim.FIXED_PREAMBLE)
    if fixed:
        parts.append(fixed)
    if preamble:
        parts.extend([delimiters.get(pdelim.PRE_PREAMBLE, ''), preamble, '\n'])

    # XXX Maybe find a more efficient way than closures (re: function-call overhead)
    def add_context():
        '''
        Append the context info to the output parts
        '''
        if contexts:
            parts.append(delimiters.get(pdelim.PRE_ALL_CONTEXT, ''))
            for c in contexts:
                # Some prompt conventions might use a pre & post context convention
                # Some use internal delimiters only
                parts.append(delimiters.get(pdelim.PRECONTEXT, ''))
                if context_with_metadata:
                    metadata, val = c
                    parts.append(delimiters.get(pdelim.PRE_CONTEXT_METADATA, '') + '\n')
                    parts.append(metadata + '\n')
                    parts.append(delimiters.get(pdelim.POST_CONTEXT_METADATA, '') + '\n')
                if context_with_metadata:
                    metadata, val = c
                    parts.append(str(val))
                else:
                    parts.append(str(c))
                parts.append(delimiters.get(pdelim.POSTCONTEXT, ''))
                parts.append(delimiters.get(pdelim.INTERCONTEXT, '\n'))
            del parts[-1]  # Final intercontext (might be empty anyway) not needed
            parts.append(delimiters.get(pdelim.POST_ALL_CONTEXT, ''))

    def add_query():
        '''
        Append the main query to the output parts
        '''
        parts.append(delimiters.get(pdelim.PREQUERY, ''))
        parts.append(query)

    # 
    if delimiters.get(pdelim.META_ORDERING, ordering.CONTEXT_QUERY) \
            == ordering.CONTEXT_QUERY:
        add_context()
        add_query()
    else:
        add_query()
        add_context()

    # XXX: Feels a bit weird that the post-query bit must be outside the query
    # closure. Maybe needs a rename?
    parts.append(delimiters.get(pdelim.POSTQUERY, '\n'))
    full_context = ''.join(parts)
    return full_context


# Support former name, for now
context_build = format

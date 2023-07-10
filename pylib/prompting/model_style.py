# SPDX-FileCopyrightText: 2023-present Uche Ogbuji <uche@ogbuji.net>
# SPDX-License-Identifier: Apache-2.0
# ogbujipt.prompting.model_style

'''
Delimiters for common LLM model prompting styles

Plain Alpaca style, e.g.:

* WizardLM

Alpaca-instruct style, e.g.

* Nous-Hermes

Vicuña style, e.g.

* Robin

Also includes Orca & Airoboros

Useful collection of Alpaca demo prompts: https://huggingface.co/datasets/tatsu-lab/alpaca
'''

from ogbujipt.prompting.basic import pdelim, ordering

VICUNA_DELIMITERS = {
    pdelim.PREQUERY: '### USER:',
    pdelim.POSTQUERY: '### ASSISTANT:',
}

ALPACA_DELIMITERS = {
    pdelim.POSTQUERY: '### Response:',
}

ALPACA_INSTRUCT_DELIMITERS = {
    pdelim.PREQUERY: '### Instruction:',
    pdelim.POSTQUERY: '### Response:',
}

ALPACA_INSTRUCT_INPUT_DELIMITERS = {
    pdelim.PREQUERY: '### Instruction:',
    pdelim.POSTQUERY: '### Response:',
    # Expect a single context item, to treat as the input:
    pdelim.PRE_ALL_CONTEXT: '### Input:',
    pdelim.META_ORDERING: ordering.QUERY_CONTEXT
}

ORCA_DELIMITERS = {
    pdelim.FIXED_PREAMBLE: '### System:\nYou are an AI assistant that follows instruction extremely well. Help as much as you can.',  # noqa E501
    pdelim.PREQUERY: '### User:',
    pdelim.POSTQUERY: '### Response:',
    # Expect a single context item, to treat as the input:
    pdelim.PRE_ALL_CONTEXT: '### Input:',
    pdelim.META_ORDERING: ordering.QUERY_CONTEXT
}


'''
Model style for airoboros: https://huggingface.co/jondurbin/airoboros-13b-gpt4

Nice, meaty example of context-obedient QA prompting: https://huggingface.co/datasets/jondurbin/airoboros-gpt4/blob/main/full-example.md

https://www.reddit.com/r/LocalLLaMA/comments/1408ued/airoboros_gpt4_instructed_contextobedient/

Example:

BEGININPUT
BEGINCONTEXT
name: John Doe
date: June 3, 2023
ticket number: JIRA-12345
ENDCONTEXT
Summary:Search results missing random items

Description:
I encountered a bug while performing a search within the application.
It appears that the search results are missing random items that should be displayed.
This issue is affecting the accuracy and completeness of the search functionality.

Steps to Reproduce:
1. Log in to the application.
2. Navigate to the search feature.
3. Enter a search query that should return multiple results.
4. Observe the displayed search results.

Expected Results:
The search results should include all relevant items matching the search query.

Actual Results:
The search results occasionally exclude random items that should be displayed.
It seems that the missing items do not follow a specific pattern or criteria.
Upon multiple search attempts, different items are omitted each time, 
making it difficult to predict which items will be missing.
ENDINPUT

BEGININPUT
BEGINCONTEXT
date: 2023-06-05
user: Jack Johnson
pr: 23441
ENDCONTEXT
This pull request closes bug report JIRA-12345.

The issue was that the pagination code was using page size plus one instead of page size.
ENDINPUT

BEGININSTRUCTION
Do we have any bug reports related to search results?  If so, were they fixed?  Source?
ENDINSTRUCTION


'''

# Closed-context prompting
AIROBOROS_OBEDIENT_DELIMITERS = {
    pdelim.PREQUERY: 'BEGININSTRUCTION',
    pdelim.POSTQUERY: 'ENDINSTRUCTION\nASSISTANT:',
    pdelim.PRECONTEXT: 'BEGININPUT',
    pdelim.POSTCONTEXT: 'ENDINPUT',
    pdelim.PRE_ALL_CONTEXT: 'USER:',
}

# If you're not using the closed-context/obedient prompting, it's just Vicuña style
AIROBOROS_DELIMITERS = VICUNA_DELIMITERS

# XXX: Should this just be a FIXED_PREAMBLE?
AIROBOROS_SUGGESTED_PREAMBLE = 'A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user\'s questions, and doesn\'t make up answers if it doesn\'t know.'  # noqa

AIR_CONOB_INPUT_PRETMPL = '''\
BEGINCONTEXT
{context}
ENDCONTEXT
{text}
'''


def concat_input_prompts(context_content_pairs):
    '''
    Turn a sequence of context / text pairs into a composite
    Input set for airoboros
    Context is just the key, value pairs in a text string
    '''
    parts = [
        (AIR_CONOB_INPUT_PRETMPL.format(context=ctx, text=text))
        for ctx, text in context_content_pairs
        # for ix, (ctx, text) in enumerate(context_content_pairs)
    ]
    # pre_tmpl = ''.join(parts)
    # return PromptTemplate.from_template(''.join(parts))
    return ''.join(parts)

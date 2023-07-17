# SPDX-FileCopyrightText: 2023-present Uche Ogbuji <uche@ogbuji.net>
# SPDX-License-Identifier: Apache-2.0
# ogbujipt.prompting.model_style

'''
Delimiters for common LLM model prompting styles.

For details see https://github.com/uogbuji/OgbujiPT/wiki/Prompt-templates
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


# Closed-context prompting
AIROBOROS_OBEDIENT_DELIMITERS = {
    pdelim.PREQUERY: 'BEGININSTRUCTION',
    pdelim.POSTQUERY: 'ENDINSTRUCTION\nASSISTANT:',
    pdelim.PRECONTEXT: 'BEGININPUT',
    pdelim.POSTCONTEXT: 'ENDINPUT',
    pdelim.PRE_ALL_CONTEXT: 'USER:',
}

# If not using the closed-context/obedient prompting, it's just Vicuña style
AIROBOROS_DELIMITERS = VICUNA_DELIMITERS

# XXX: Should this just be a FIXED_PREAMBLE?
AIROBOROS_SUGGESTED_PREAMBLE = 'A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user\'s questions, and doesn\'t make up answers if it doesn\'t know.'  # noqa E501

AIR_CONOB_INPUT_PRETMPL = '''\
BEGINCONTEXT
{context}
ENDCONTEXT
{text}
'''

# Define delimeters in ChatGPT style
CHATGPT_DELIMITERS = {
    pdelim.PREQUERY: '### USER:',
    pdelim.POSTQUERY: '### ASSISTANT:',
}


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

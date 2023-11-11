# SPDX-FileCopyrightText: 2023-present Oori Data <info@oori.dev>
# SPDX-License-Identifier: Apache-2.0
# ogbujipt.prompting.model_style

'''
Delimiters for common LLM model prompting styles.

For details see https://github.com/uogbuji/OgbujiPT/wiki/Prompt-templates
'''

import re
from typing import List
from enum import Enum

from ogbujipt.prompting.basic import pdelim, ordering

# __all__ = ['hosted_model_openai']

# Quick cheat sheet:
# Alpaca style = [A…] \n ### Response:
# Alpaca/Instruct style = ### Instruction:\n [A…] ### Response:
# Alpaca/Instruct style, with input = ### Instruction:\n [A…] ### Input:\n ### Response:
# Vicuña style = ### Human:\n [A…] ### Assistant:
# Wizard style = USER:\n [A…] ### ASSISTANT:

UNKNOWN = 'UNKNOWN'


class style(Enum):
    '''
    Marker of different prompring styles. When LLMs are trained, they're tuned
    to expect prompts according to a specific convention, and can become very
    erratic if you use the wrong one
    '''
    ALPACA = 1
    ALPACA_INSTRUCT = 2
    VICUNA = 3
    WIZARD = 4


VICUNA_DELIMITERS = {
    pdelim.PREQUERY: '### USER: ',
    pdelim.POSTQUERY: '\n### ASSISTANT: ',
}

VICUNA_NOHASH_DELIMITERS = {
    pdelim.PREQUERY: 'USER: ',
    pdelim.POSTQUERY: '\n\nASSISTANT:\n',
}

ALPACA_DELIMITERS = {
    pdelim.POSTQUERY: '\n\n### Response:\n',
}

ALPACA_INSTRUCT_DELIMITERS = {  # remove? seems redundant with ALPACA_INSTRUCT_INPUT_DELIMITERS
    pdelim.PREQUERY: '### Instruction:\n',
    pdelim.POSTQUERY: '\n### Response:\n',
}

ALPACA_INSTRUCT_INPUT_DELIMITERS = {
    pdelim.PREQUERY: '### Instruction:\n',
    pdelim.POSTQUERY: '\n\n### Response:\n',
    # Expect a single context item, to treat as the input:
    pdelim.PRE_ALL_CONTEXT: '\n\n### Input:\n',
    pdelim.META_ORDERING: ordering.QUERY_CONTEXT
}

ORCA_DELIMITERS = {
    pdelim.FIXED_PREAMBLE: '### System:\nYou are an AI assistant that follows instruction extremely well. Help as much as you can.',  # noqa E501
    pdelim.PREQUERY: '\n\n### User:\n',
    pdelim.POSTQUERY: '\n\n### Response:\n',
    # Expect a single context item, to treat as the input:
    pdelim.PRE_ALL_CONTEXT: '\n\n### Input:\n',
    pdelim.META_ORDERING: ordering.QUERY_CONTEXT
}

LLAMA_INSTRUCT = {
    pdelim.PRE_PREAMBLE: '### System:\n',
    pdelim.PREQUERY: '\n\n### User:\n',
    pdelim.POSTQUERY: '\n\n### Assistant:\n',
}

AIROBOROS_SUMMARIZATION_DELIMITERS = {
    pdelim.PRE_ALL_CONTEXT: 'User:\nBEGININPUT\nBEGINCONTEXT\n',
    pdelim.PREQUERY: '\nBEGININSTRUCTION\n',
    pdelim.POSTQUERY: '\nENDINSTRUCTION\nASSISTANT: ',
    pdelim.POST_ALL_CONTEXT: '\nENDCONTEXT\nENDINPUT\n'
}

# Closed-context prompting
AIROBOROS_OBEDIENT_DELIMITERS = {
    pdelim.PRECONTEXT: 'BEGININPUT\n',
    pdelim.POSTCONTEXT: '\nENDINPUT',
    pdelim.PRE_CONTEXT_METADATA: 'BEGINCONTEXT',
    pdelim.POST_CONTEXT_METADATA: 'ENDCONTEXT',
    pdelim.PREQUERY: '\nBEGININSTRUCTION\n',
    pdelim.POSTQUERY: '\nENDINSTRUCTION\n'
}

MISTRAL_INSTRUCTION_DELIMITERS = {
    pdelim.PRE_ALL_CONTEXT: '<s>[INST]',
    pdelim.POST_ALL_CONTEXT: '\n[/INST]',
}

CHATML_DELIMITERS = {
    pdelim.PRE_PREAMBLE: '<|im_start|>system\n',
    pdelim.PREQUERY: '<|im_end|>\n<|im_start|>user\n',
    pdelim.POSTQUERY: '<|im_end|>\n<|im_start|>assistant',
}

# If not using the closed-context/obedient prompting, it's just Vicuña style
AIROBOROS_DELIMITERS = VICUNA_DELIMITERS

# XXX: Should this just be a FIXED_PREAMBLE?
AIROBOROS_SUGGESTED_PREAMBLE = 'A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user\'s questions, and doesn\'t make up answers if it doesn\'t know.'  # noqa E501

AIR_CONOB_INPUT_PRETMPL = '\nBEGINCONTEXT\n{metadata}\nENDCONTEXT\n{context}'

# Define delimeters in OpenAI GPT style
OPENAI_GPT_DELIMITERS = {
    pdelim.PREQUERY: '### USER:\n',
    pdelim.PRECONTEXT: '\n"""\n',
    pdelim.POSTCONTEXT: '\n"""',
    pdelim.POSTQUERY: '\n\n### ASSISTANT:\n',
    pdelim.META_ORDERING: ordering.QUERY_CONTEXT
}

GORILLA_DELIMITERS = {}  # Gorilla doesn't use delimiters, just natural language prompts

# LLAMA_DELIMITERS?, see https://www.reddit.com/r/LocalLLaMA/comments/155po2p/get_llama_2_prompt_format_right/
# or https://huggingface.co/blog/llama2#how-to-prompt-llama-2
'''
<s>[INST] <<SYS>>
{{ system_prompt }}
<</SYS>>

{{ user_message }} [/INST]
'''

# XXX: Rethink in light of word loom
def concat_input_prompts(context_content_pairs):
    '''
    Turn a sequence of context / text pairs into a composite
    Input set for airoboros
    Context is just the key, value pairs in a text string
    '''
    parts = [
        (AIR_CONOB_INPUT_PRETMPL.format(metadata=metas, context=context))
        for metas, context in context_content_pairs
        ]
    return ''.join(parts)


MODEL_NAME_SEARCH_PATTERNS = {
    re.compile('nous-hermes'): [style.ALPACA_INSTRUCT],
    re.compile('wizardlm'): [style.WIZARD],
    # re.compile(''): [style.ALPACA_INSTRUCT],
}


def model_style_from_name(mname: str) -> List[str]:
    '''
    Uses heuristics to figure out the prompting/model style from its name

    >>> from ogbujipt.prompting.model_style import model_style_from_name
    >>> model_style_from_name('path/wizardlm-13b-v1.0-uncensored.ggmlv3.q6_K.bin')
    [<style.WIZARD: 4>]
    >>> from ogbujipt.llm_wrapper import openai_api
    >>> llm_api = openai_api(base_url='http://localhost:8000')
    # Model style hosted via the API
    >>> model_style_from_name(llm_api.hosted_model()[0])
    '''
    # Use the final slash portion, in case it's a full path
    mname = mname.split('/')[-1]
    for pat, styles in MODEL_NAME_SEARCH_PATTERNS.items():
        if pat.search(mname):
            mstyles = styles
            break
    else:
        mstyles = UNKNOWN
    return mstyles

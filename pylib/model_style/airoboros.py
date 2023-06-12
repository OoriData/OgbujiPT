# SPDX-FileCopyrightText: 2023-present Uche Ogbuji <uche@ogbuji.net>
#
# SPDX-License-Identifier: Apache-2.0
# ogbujipt.model_styles.airoboros

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
Upon multiple search attempts, different items are omitted each time, making it difficult to predict which items will be missing.
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

# import re
# from functools import partial
# from typing import Optional, List, Mapping, Any, Union

# from langchain.schema import AgentAction, AgentFinish
# from langchain.agents import Tool, LLMSingleActionAgent, AgentOutputParser

# Composing prompts from input sequences, rather than fixed, composite structure, so don't actually need PipelinePromptTemplate
# https://python.langchain.com/en/latest/modules/prompts/prompt_templates/examples/prompt_composition.html
# from langchain.prompts.pipeline import PipelinePromptTemplate
from langchain.prompts.prompt import PromptTemplate

# Prompts for airoboros using the context-obedient question answering approach
# as specified in the model card.
AIR_CONOB_OUTER_TMPL = '''\
A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.

USER:
{given_inputs}
BEGININSTRUCTION
{instructions}
ENDINSTRUCTION
Don't make up answers if you don't know.
ASSISTANT:
'''

AIR_CONOB_INPUT_PRETMPL = '''\
BEGININPUT
BEGINCONTEXT
{context}
ENDCONTEXT
{text}
ENDINPUT
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


AIR_CONOB_PROMPT_TMPL = PromptTemplate(
    input_variables=['given_inputs', 'instructions'],
    template=AIR_CONOB_OUTER_TMPL,
)

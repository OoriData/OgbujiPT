'''
pytest test

or

pytest test/test_alpaca_prompting.py
'''
# import pytest

from ogbujipt.prompting.basic import format
from ogbujipt.prompting.model_style import (
    ALPACA_DELIMITERS,
    ALPACA_INSTRUCT_INPUT_DELIMITERS
    )

def alpaca_prompt_building(PROMPTING_USER_QUERY):
    EXPECTED_PROMPT = 'what\'s nine plus ten?\n\n### Response:\n'  # noqa: E501

    prompt = format(
        PROMPTING_USER_QUERY,
        delimiters=ALPACA_DELIMITERS
        )
    assert prompt == EXPECTED_PROMPT


def alpaca_instruct_prompt_building(PROMPTING_USER_QUERY):
    EXPECTED_PROMPT = '### Instruction:\nwhat\'s nine plus ten?\n\n### Response:\n'  # noqa: E501

    prompt = format(
        PROMPTING_USER_QUERY,
        delimiters=ALPACA_INSTRUCT_INPUT_DELIMITERS
        )
    assert prompt == EXPECTED_PROMPT


def alpaca_instruct_input_prompt_building(PROMPTING_USER_QUERY, PROMPTING_CONTEXT):
    EXPECTED_PROMPT = '### Instruction:\nwhat\'s nine plus ten?\n\n### Input:\nThe entirety of "Principia Mathematica" by Isaac Newton\n\n### Response:\n'  # noqa: E501

    prompt = format(
        PROMPTING_USER_QUERY,
        contexts=PROMPTING_CONTEXT,
        delimiters=ALPACA_INSTRUCT_INPUT_DELIMITERS
        )
    assert prompt == EXPECTED_PROMPT


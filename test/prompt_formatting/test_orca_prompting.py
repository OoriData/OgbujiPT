'''
pytest test

or

pytest test/test_orca_prompting.py
'''
# import pytest

from ogbujipt.prompting.basic import format
from ogbujipt.prompting.model_style import ORCA_DELIMITERS


def orca_prompt_building(PROMPTING_USER_QUERY):
    EXPECTED_PROMPT = '### System:\nYou are an AI assistant that follows instruction extremely well. Help as much as you can.\n\n### User:\nwhat\'s nine plus ten?\n\n### Response:\n'  # noqa: E501

    prompt = format(
        PROMPTING_USER_QUERY,
        delimiters=ORCA_DELIMITERS
        )
    assert prompt == EXPECTED_PROMPT


def orca_input_prompt_building(PROMPTING_USER_QUERY, PROMPTING_CONTEXT):
    EXPECTED_PROMPT = '### System:\nYou are an AI assistant that follows instruction extremely well. Help as much as you can.\n\n### User:\nwhat\'s nine plus ten?\n\n### Input:\nThe entirety of "Principia Mathematica" by Isaac Newton\n\n### Response:\n'  # noqa: E501

    prompt = format(
        PROMPTING_USER_QUERY,
        contexts=PROMPTING_CONTEXT,
        delimiters=ORCA_DELIMITERS
        )
    assert prompt == EXPECTED_PROMPT

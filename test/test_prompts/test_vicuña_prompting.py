'''
pytest test

or

pytest test/test_vicuña_prompting.py
'''
# import pytest

from ogbujipt.prompting.basic import format
from ogbujipt.prompting.model_style import VICUNA_DELIMITERS


def vicuna_prompt_building(PROMPTING_USER_QUERY):
    EXPECTED_PROMPT = '### USER:\nwhat\'s nine plus ten?\n\n### ASSISTANT:\n'  # noqa: E501

    prompt = format(
        PROMPTING_USER_QUERY,
        delimiters=VICUNA_DELIMITERS
        )
    assert prompt == EXPECTED_PROMPT

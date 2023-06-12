'''
pytest test

or

pytest test/test_vicuna_prompting.py
'''

from ogbujipt.model_style.vicuna import VICUNA_PROMPT_TMPL


def test_basic_prompt():
    EXPECTED_PROMPT = '### USER: \n\nWhat is the capital of Cross River state?\n\n### ASSISTANT:\n'  # noqa

    prompt = VICUNA_PROMPT_TMPL.format(
        query='What is the capital of Cross River state?')

    assert prompt == EXPECTED_PROMPT

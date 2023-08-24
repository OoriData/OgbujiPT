'''
pytest test

or

pytest test/test_vicuña_prompting.py
'''
# import pytest

from ogbujipt.prompting.basic import format
from ogbujipt.prompting.model_style import VICUNA_DELIMITERS


def test_basic_prompt_substyles(BAD_XML_CODE):
    EXPECTED_PROMPT = '### USER:\nCorrect the following XML to make it well-formed\n<earth>\n<country><b>Russia</country></b>\n<capital>Moscow</capital>\n</Earth>\n### ASSISTANT:'  # noqa

    prompt = format(
        'Correct the following XML to make it well-formed\n' + BAD_XML_CODE,
        delimiters=VICUNA_DELIMITERS
    )
    # 'You are a friendly AI who loves conversation\n\nHow are you?\n'

    assert prompt == EXPECTED_PROMPT

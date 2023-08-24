'''
pytest test

or

pytest test/test_alpaca_prompting.py
'''
# import pytest

from ogbujipt.prompting.basic import format
from ogbujipt.prompting.model_style import (
    ALPACA_DELIMITERS,
    ALPACA_INSTRUCT_DELIMITERS,
    ALPACA_INSTRUCT_INPUT_DELIMITERS
    )


def test_basic_prompt_substyles(BAD_XML_CODE):
    # EXPECTED_PROMPT = 'Correct the following XML to make it well-formed\n\n\n<earth>\n<country><b>Russia</country></b>\n<capital>Moscow</capital>\n</Earth>\n### Response:'  # noqa
    EXPECTED_PROMPT = '\nCorrect the following XML to make it well-formed\n\n\n\n\n<earth>\n<country><b>Russia</country></b>\n<capital>Moscow</capital>\n</Earth>\n### Response:\n'  # noqa

    prompt = format(
        BAD_XML_CODE,
        preamble='Correct the following XML to make it well-formed\n',
        delimiters=ALPACA_DELIMITERS
    )
    # 'You are a friendly AI who loves conversation\n\nHow are you?\n'

    assert prompt == EXPECTED_PROMPT

    EXPECTED_PROMPT = '\nYou are a helpful assistant.\n\n\n### Instruction:\nCorrect the following XML to make it well-formed\n\n<earth>\n<country><b>Russia</country></b>\n<capital>Moscow</capital>\n</Earth>\n### Response:\n'  # noqa

    prompt = format(
        'Correct the following XML to make it well-formed\n\n' + 
        BAD_XML_CODE,
        preamble='You are a helpful assistant.',
        delimiters=ALPACA_INSTRUCT_DELIMITERS
    )

    assert prompt == EXPECTED_PROMPT

    # EXPECTED_PROMPT = 'Have a look at the following XML\n### Instruction:\nPlease correct this XML to make it well-formed\n### Input:\n\n<earth>\n<country><b>Russia</country></b>\n<capital>Moscow</capital>\n</Earth>\n\n\n### Response:'  # noqa
    EXPECTED_PROMPT = '\nHave a look at the following XML\n\n\n### Instruction:\nPlease correct this XML to make it well-formed\n### Input:\n\n<earth>\n<country><b>Russia</country></b>\n<capital>Moscow</capital>\n</Earth>\n\n\n### Response:\n'  # noqa

    prompt = format(
        'Please correct this XML to make it well-formed',
        preamble='Have a look at the following XML',
        contexts=BAD_XML_CODE,
        delimiters=ALPACA_INSTRUCT_INPUT_DELIMITERS
    )

    # print(prompt)
    assert prompt == EXPECTED_PROMPT


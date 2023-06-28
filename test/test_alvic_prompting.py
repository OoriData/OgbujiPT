'''
pytest test

or

pytest test/test_alvic_prompting.py
'''
import pytest

from ogbujipt.model_style.alvic import make_prompt, sub_style


def test_basic_prompt_substyles():
    BAD_XML_CODE = '''\
<earth>
<country><b>Russia</country></b>
<capital>Moscow</capital>
</Earth>'''

    EXPECTED_PROMPT = 'Correct the following XML to make it well-formed\n### Inputs:\n<earth>    \n<country><b>Russia</country></b>\n<capital>Moscow</capital>\n</Earth>\n\n\n### Response:\n'  # noqa
    # using the default sub-style, so Alpaca
    prompt = make_prompt(
        'Correct the following XML to make it well-formed',
        inputs=BAD_XML_CODE,
        )

    assert prompt == EXPECTED_PROMPT

    # Explicitly state the Alpaca style. EXPECTED doesn't change
    prompt = make_prompt(
        'Correct the following XML to make it well-formed',
        inputs=BAD_XML_CODE,
        sub=sub_style.ALPACA
        )
    assert prompt == EXPECTED_PROMPT

    EXPECTED_PROMPT = '### Instruction:\n\nCorrect the following XML to make it well-formed\n### Inputs:\n<earth>    \n<country><b>Russia</country></b>\n<capital>Moscow</capital>\n</Earth>\n\n\n### Response:\n'  # noqa
    prompt = make_prompt(
        'Correct the following XML to make it well-formed',
        inputs=BAD_XML_CODE,
        sub=sub_style.ALPACA_INSTRUCT
        )
    assert prompt == EXPECTED_PROMPT

    EXPECTED_PROMPT = '### USER:\n\nWhat is the capital of Cross River state?\n\n### ASSISTANT:\n'  # noqa
    with pytest.warns(UserWarning):
        prompt = make_prompt(
            'What is the capital of Cross River state?',
            inputs='NONSENSE',  # Meaningless for Vicuña
            sub=sub_style.VICUNA)

    prompt = make_prompt('What is the capital of Cross River state?',
        sub=sub_style.VICUNA)
    assert prompt == EXPECTED_PROMPT

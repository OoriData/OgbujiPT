'''
pytest test

or

pytest test/test_alpaca_prompting.py
'''

from ogbujipt.model_style.alpaca import prep_instru_inputs, ALPACA_PROMPT_TMPL


def test_basic_prompt():
    EXPECTED_PROMPT = '### Instruction:\n\nCorrect the following XML to make it well-formed\n### Inputs:\n<earth>    \n<country><b>Russia</country></b>\n<capital>Moscow</capital>\n</Earth>\n\n\n### Response:\n'  # noqa
    BAD_XML_CODE = '''\
<earth>    
<country><b>Russia</country></b>
<capital>Moscow</capital>
</Earth>'''

    instru_inputs = prep_instru_inputs(
        'Correct the following XML to make it well-formed',
        inputs=BAD_XML_CODE
        )

    prompt = ALPACA_PROMPT_TMPL.format(instru_inputs=instru_inputs)

    assert prompt == EXPECTED_PROMPT


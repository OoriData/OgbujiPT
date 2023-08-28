'''
pytest test

or

pytest test/test_ogbujipt.py
'''
# ruff: noqa: E501

# import pytest

from ogbujipt import oapi_first_choice_text

def test_oapi_first_choice_text(OPENAI_RESPONSE_OBJECT):
    text1 = oapi_first_choice_text(OPENAI_RESPONSE_OBJECT)
    assert text1 == 'The fitness gram pacer test is a multi stage aerobic capacity test that progressively gets more difficult as it continues. The 20 meter pacer test will begin in 30 seconds. Line up at the start.'
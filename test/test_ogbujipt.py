# SPDX-FileCopyrightText: 2023-present Oori Data <info@oori.dev>
# SPDX-License-Identifier: Apache-2.0
# test/test_ogbujipt.py
'''
pytest test

or

pytest test/test_ogbujipt.py
'''
# ruff: noqa: E501

# import pytest

from ogbujipt.llm_wrapper import llm_response  #, openai_chat_api

def test_oapi_first_choice_text(OPENAI_TEXT_RESPONSE_OBJECT):
    text1 = llm_response.from_openai_chat(OPENAI_TEXT_RESPONSE_OBJECT).first_choice_text
    assert text1 == '…is an exceptional employee who has made significant contributions to our company.'

def test_oapi_first_choice_message(OPENAI_MSG_RESPONSE_OBJECT):
    msg1 = llm_response.from_openai_chat(OPENAI_MSG_RESPONSE_OBJECT).first_choice_text
    assert msg1 == '…is an exceptional employee who has made significant contributions to our company.'


if __name__ == '__main__':
    raise SystemExit("Attention! Run with pytest")

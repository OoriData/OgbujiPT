# SPDX-FileCopyrightText: 2023-present Oori Data <info@oori.dev>
# SPDX-License-Identifier: Apache-2.0
# test/test_llm_wrapper.py

import os
import json

import pytest

from ogbujipt import config
from ogbujipt.llm_wrapper import openai_api, openai_chat_api, prompt_to_chat # , DUMMY_MODEL
from ogbujipt.llm_wrapper import llama_cpp_http, llama_cpp_http_chat

try:
    import respx
except ImportError:
    raise RuntimeError('respx not installed. Please see wiki for setup: https://github.com/OoriData/OgbujiPT/wiki/Notes-for-contributors')  # noqa: E501

from httpx import Response

if 'OPENAI_API_KEY' in os.environ:
    del os.environ['OPENAI_API_KEY']

@pytest.mark.asyncio
@respx.mock
async def test_openai_llm_wrapper():
    # httpx_mock.add_response(
    #     url='http://127.0.0.1:8000/v1/models',
    #     json={'object':'list','data':[{'id':'model1','object':'model','owned_by':'me','permissions':[]}]})
    # httpx_mock.add_response(
    #     url='https://api.openai.com/v1/models',
    #     json={'object':'list','data':[{'id':'model1','object':'model','owned_by':'me','permissions':[]}]})

    respx.get('http://127.0.0.1:8000/v1/models').mock(return_value=Response(200))
    # FIXME: Figure out the right patterns for checking the HTTP requests
    # route1 = respx.get('http://127.0.0.1:8000/v1/models').mock(return_value=Response(200))

    host = 'http://127.0.0.1'
    api_key = 'jsbdflkajsdhfklajshdfkljalk'
    port = '8000'
    debug = True
    # model = DUMMY_MODEL
    rev = 'v1'
    base_url = f'{host}:{port}/{rev}'

    test_model = openai_chat_api(base_url=base_url, api_key=api_key, debug=debug)
    # assert route1.called

    assert test_model.api_key == api_key
    assert test_model.parameters.debug == debug
    assert test_model.model is None

    # Not OpenAI
    test_model = openai_api(base_url=base_url, debug=debug)

    assert test_model.api_key == config.OPENAI_KEY_DUMMY
    assert test_model.parameters.debug == debug
    assert test_model.model is None

    test_model = openai_chat_api(base_url=base_url, debug=debug)

    assert test_model.api_key == config.OPENAI_KEY_DUMMY
    assert test_model.parameters.debug == debug
    assert test_model.model is None
    assert test_model.base_url == base_url

    test_model = openai_chat_api(base_url=base_url, api_key=api_key, debug=debug)

    assert test_model.api_key == api_key
    assert test_model.parameters.debug == debug
    assert test_model.model is None


@pytest.mark.asyncio
@respx.mock
async def test_openai_llama_cpp_http():
    # httpx_mock.add_response(
    #     url='http://127.0.0.1:8000/v1/models',
    #     json={'object':'list','data':[{'id':'model1','object':'model','owned_by':'me','permissions':[]}]})
    # httpx_mock.add_response(
    #     url='https://api.openai.com/v1/models',
    #     json={'object':'list','data':[{'id':'model1','object':'model','owned_by':'me','permissions':[]}]})

    respx.get('http://127.0.0.1:8000/v1/models').mock(return_value=Response(200))
    route_chat = respx.post(
        'http://127.0.0.1:8000/v1/chat/completions').mock(
            return_value=Response(200, text=json.dumps(OPENAI_STYLE_CHAT_RESPONSE)))
    route_nonchat = respx.post(
        'http://127.0.0.1:8000/completion').mock(
            return_value=Response(200, text=json.dumps(OPENAI_STYLE_RESPONSE)))
    # FIXME: Figure out the right patterns for checking the HTTP requests
    # route1 = respx.get('http://127.0.0.1:8000/v1/models').mock(return_value=Response(200))

    host = 'http://127.0.0.1'
    api_key = 'jsbdflkajsdhfklajshdfkljalk'
    port = '8000'
    # model = DUMMY_MODEL
    base_url = f'{host}:{port}'

    test_endpoint = llama_cpp_http_chat(base_url=base_url, apikey=api_key)

    assert test_endpoint.apikey == api_key
    assert test_endpoint.model is None

    _ = await test_endpoint(prompt_to_chat('test'))
    assert route_chat.called

    test_endpoint = llama_cpp_http(base_url=base_url, apikey=api_key)

    assert test_endpoint.apikey == api_key
    assert test_endpoint.model is None

    _ = await test_endpoint('test')
    assert route_nonchat.called


def test_prompt_to_chat():
    chatified = prompt_to_chat('test')
    assert chatified == [{'content': 'test', 'role': 'user'}]


OPENAI_STYLE_CHAT_RESPONSE = {'choices': [{'finish_reason': 'stop',
   'index': 0,
   'message': {'content': 'Hello! How can I help you today? If you have any questions or need assistance, feel free to ask.',  # noqa: E501
    'role': 'assistant'}}],
 'created': 1710171748,
 'id': 'chatcmpl-YOGhMGqIh3qlkOzOCayFRSU3GYKY6NPJ',
 'model': 'unknown',
 'object': 'chat.completion',
 'usage': {'completion_tokens': 24, 'prompt_tokens': 11, 'total_tokens': 35},
 'prompt_tokens': 11,
 'generated_tokens': 24,
 'first_choice_text': 'Hello! How can I help you today? If you have any questions or need assistance, feel free to ask.'}  # noqa: E501

OPENAI_STYLE_RESPONSE = {'id': 'chatcmpl-Fvua40LBorK82dyfDD5QTDNl7DmeLggg',
 'choices': [{'finish_reason': 'stop',
   'index': 0,
   'message': {'content': 'Hello! How can I help you today? If you have any questions or need assistance, feel free to ask.',  # noqa: E501
    'role': 'assistant',
    'function_call': None,
    'tool_calls': None}}],
 'created': 1710172279,
 'model': 'unknown',
 'object': 'chat.completion',
 'system_fingerprint': None,
 'usage': {'completion_tokens': 24, 'prompt_tokens': 11, 'total_tokens': 35},
 'prompt_tokens': 11,
 'generated_tokens': 24,
 'first_choice_text': 'Hello! How can I help you today? If you have any questions or need assistance, feel free to ask.'}  # noqa: E501

if __name__ == '__main__':
    raise SystemExit("Attention! Run with pytest")

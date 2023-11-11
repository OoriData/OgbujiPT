import os
from ogbujipt import config
from ogbujipt.llm_wrapper import openai_api, openai_chat_api, prompt_to_chat # , DUMMY_MODEL

import respx

from httpx import Response

del os.environ['OPENAI_API_KEY']

@respx.mock
def test_openai_llm_wrapper():
    # httpx_mock.add_response(
    #     url='http://127.0.0.1:8000/v1/models',
    #     json={'object':'list','data':[{'id':'model1','object':'model','owned_by':'me','permissions':[]}]})
    # httpx_mock.add_response(
    #     url='https://api.openai.com/v1/models',
    #     json={'object':'list','data':[{'id':'model1','object':'model','owned_by':'me','permissions':[]}]})

    route1 = respx.get('http://127.0.0.1:8000/v1/models').mock(return_value=Response(200))

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


def test_prompt_to_chat():
    chatified = prompt_to_chat('test')
    assert chatified == [{'content': 'test', 'role': 'user'}]

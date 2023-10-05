from ogbujipt import config
from ogbujipt.llm_wrapper import openai_api, openai_chat_api, prompt_to_chat


def test_openai_llm_wrapper():
    host = 'http://127.0.0.1'
    api_key = "jsbdflkajsdhfklajshdfkljalk"
    port = '8000'
    debug = True
    model = ''
    rev = 'v1'
    api_base = f'{host}:{port}/{rev}'

    test_model = openai_chat_api(api_key=api_key, debug=debug, model=model)

    assert test_model.api_key == api_key
    assert test_model.parameters.debug == debug
    assert test_model.model == model

    # Not OpenAI
    test_model = openai_api(debug=debug, model=model)

    assert test_model.api_key == config.OPENAI_KEY_DUMMY
    assert test_model.parameters.debug == debug
    assert test_model.model == model

    test_model = openai_chat_api(api_base=api_base, debug=debug, model=model)

    assert test_model.api_key == config.OPENAI_KEY_DUMMY
    assert test_model.parameters.debug == debug
    assert test_model.model == model
    assert test_model.api_base == api_base

    test_model = openai_chat_api(api_key=api_key, debug=debug, model=model)

    assert test_model.api_key == api_key
    assert test_model.parameters.debug == debug
    assert test_model.model == model


def test_prompt_to_chat():
    chatified = prompt_to_chat('test')
    assert chatified == [{'content': 'test', 'role': 'user'}]

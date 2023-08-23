from ogbujipt import config
import unittest


host = 'http://127.0.0.1'
api_key = "jsbdflkajsdhfklajshdfkljalk"
port = '8000'
debug = True
model = ''
rev = 'v1'
api_base = f'{host}:{port}/{rev}'

def test_config():
    test_model = config.openai_live(apikey=api_key, debug=debug, model=model)

    assert test_model.api_key == api_key
    assert test_model.debug == debug
    assert test_model.model == model

    test_model = config.openai_emulation(host=host, port=port, apikey=api_key, debug=debug, model=model)

    assert test_model.api_key == api_key
    assert test_model.debug == debug
    assert test_model.model == model
    assert test_model.api_base == api_base
# SPDX-FileCopyrightText: 2023-present Oori Data <info@oori.dev>
# SPDX-License-Identifier: Apache-2.0
# ogbujipt.config

# Given the nature of a config module, we don't use __all__
# & avoid top-level imports where possible, and at the end of the module
# del any symbols we don't want exposed
# pylint: disable=wrong-import-position

'''
Configuration & globally-relevant values
'''

# Really just a bogus name for cases when OpenAPI is being emulated
# OpenAI API requires the model be specified, but many compaitble APIs
# have a model predetermined by the host
HOST_DEFAULT_MODEL = HOST_DEFAULT = 'HOST-DEFAULT'
OPENAI_KEY_DUMMY = 'OPENAI_DUMMY'


class attr_dict(dict):
    # XXX: Should unknown attr access return None rather than raise?
    # If so, this line should be: __getattr__ = dict.get
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def openai_live(apikey=None, debug=True, model=''):
    '''
    Set up to use OpenAI proper. If you don't pass in an API key, the
    environment variable OPENAI_API_KEY will be checked

    Side note: a lot of OpenAI tutorials suggest that you embed your
    OpenAI private key into the code, which is a horrible, no-good idea

    Extra reminder: If you set up your environment via .env file, make sure
    it's in .gitignore or equivalent so it never gets accidentally committed!

    Args:
        apikey (str, optional): OpenAI API key to use for authentication

        debug (bool, optional): Debug flag

    Returns:
        openai_api (openai): Prepared OpenAI API
    '''
    from warnings import warn
    warn.warn(
        'Use llm_wrapper.openai_api() instead. openai_live() will be removed in 0.6.0',
        DeprecationWarning, stacklevel=2)
    import os
    import openai as openai_api

    # openai_api.api_version
    openai_api.debug = debug
    openai_api.api_key = apikey or os.getenv('OPENAI_API_KEY')
    openai_api.model = model
    return openai_api


def openai_emulation(
        host='http://127.0.0.1',
        port='8000',
        apikey='BOGUS',
        debug=True, model=''):
    '''
    Set up emulation, to use a alternative, OpenAI API compatible service
    Port 8000 for llama-cpp-python, Port 5001 for Oobabooga

    Args:
        host (str, optional): Host address

        port (str, optional): Port to use at "host"

        apikey (str, optional): Unused standin for OpenAI API key

        debug (bool, optional): Debug flag

    Returns:
        openai_api (openai): Prepared (emulated) OpenAI API
    '''
    from warnings import warn
    warn.warn(
        'Use llm_wrapper.openai_api() instead. openai_live() will be removed in 0.6.0',
        DeprecationWarning, stacklevel=2)
    import openai as openai_api

    rev = 'v1'
    openai_api.api_key = apikey
    openai_api.api_base = f'{host}:{port}/{rev}'
    openai_api.debug = debug
    openai_api.model = model
    return openai_api

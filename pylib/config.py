# SPDX-FileCopyrightText: 2023-present Uche Ogbuji <uche@ogbuji.net>
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
HOST_DEFAULT = 'HOST-DEFAULT'


class attr_dict(dict):
    # XXX: Should unknown attr access return None rather than raise?
    # If so, this line should be: __getattr__ = dict.get
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def openai_live(
        apikey=None,
        debug=True
        ):
    '''
    Set up to use OpenAI proper. If you don't pass in an API key, the
    environment variable OPENAI_API_KEY will be checked

    Side note: a lot of OpenAI tutorials suggest that you embed your
    OpenAI private key into the code, which is a horrible, terrible idea

    Extra reminder: If you set up your environment via .env file, make sure
    it's in .gitignore or equivalent so it never gets accidentally committed!

    Args:
        apikey (str, optional): OpenAI API key to use for authentication

        debug (bool, optional): Debug flag

    Returns:
        openai_api (openai): Prepared OpenAI API
    '''
    import openai as openai_api
    from dotenv import load_dotenv

    load_dotenv()
    # openai_api.api_version
    openai_api.debug = debug
    openai_api.params = attr_dict(
        api_key=apikey,
        api_base=openai_api.api_base,
        debug=debug
        )

    return openai_api


def openai_emulation(
        host='http://127.0.0.1',
        port='8000',
        rev='v1',
        model=HOST_DEFAULT,
        apikey='BOGUS', oaitype='open_ai', debug=True):
    '''
    Set up emulation, to use a alternative, OpenAI API compatible service
    Port 8000 for llama-cpp-python, Port 5001 for Oobabooga

    Args:
        host (str, optional): Host address

        port (str, optional): Port to use at "host"

        rev (str, optional): OpenAI revision to use

        apikey (str, optional): API key to use for authentication

        oaitype (str, optional): OpenAI type to use

        debug (bool, optional): Debug flag

    Returns:
        openai_api (openai): Prepared (emulated) OpenAI API
    '''
    import openai as openai_api

    openai_api.api_key = apikey
    openai_api.api_type = oaitype
    openai_api.api_base = f'{host}:{port}/{rev}'
    openai_api.debug = debug

    openai_api.params = attr_dict(
        api_key=apikey,
        api_type=oaitype,
        api_base=openai_api.api_base,
        model=model,
        debug=debug
        )

    return openai_api

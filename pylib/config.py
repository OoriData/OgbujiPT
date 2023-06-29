# SPDX-FileCopyrightText: 2023-present Uche Ogbuji <uche@ogbuji.net>
#
# SPDX-License-Identifier: Apache-2.0
# ogbujipt.config

# Given the nature of a config module, we don't use __all__
# & avoid top-level imports where possible, and at the end of the module
# del any symbols we don't want exposed
# pylint: disable=wrong-import-position

'''
Configuration & globally-relevant values
'''

import openai as openai_api


def openai_live(
        debug=True,
        apikey=None):
    '''
    Set up to use OpenAI proper. If you don't pass in an API key, the
    environment variable OPENAI_API_KEY will be checked

    Side note: a lot of OpenAI tutorials suggest that you embed your
    OpenAI private key into the code, which is a terrible idea

    Extra reminder: If you set up your environment via .env file, make sure
     it's in .gitignore or equivalent so it never gets accisentally committed!
    '''
    from dotenv import load_dotenv
    load_dotenv()
    if debug:
        os.environ['OPENAI_DEBUG'] = 'true'

    return openai_api


def openai_emulation(
        host='http://127.0.0.1',
        port='8000',  # llama-cpp-python; for Ooba, use '5001'
        rev='v1',
        apikey='BOGUS',
        oaitype='open_ai', debug=True):
    '''
    Set up emulation, to use a alternative, OpenAI API compatible service
    '''
    import os

    # apikey = apikey or os.getenv('OPENAI_API_KEY')
    # os.environ['OPENAI_API_KEY'] = oaikey  # Dummy val, so NBD

    openai_api.api_key = apikey
    openai_api.api_type = oaitype
    openai_api.api_base = f'{host}:{port}/{rev}'
    openai_api.debug = debug

    return openai_api

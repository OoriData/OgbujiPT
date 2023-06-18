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


def openai_live(debug=True):
    '''
    Set up to use OpenAI proper. Assumes your APi key is in a .env file
    Make sure you have the .env file in .gitignore or equivalent
    '''
    import os
    from dotenv import load_dotenv
    load_dotenv()
    if debug:
        os.environ['OPENAI_DEBUG'] = 'true'


def openai_emulation(
        host='http://127.0.0.1',
        port='8000',  # llama-cpp-python; for Ooba, use '5001'
        rev='v1',
        oaikey='BOGUS', oaitype='open_ai', debug=True):
    '''
    Set up emulation, to use a alternative, OpenAI API compatible service
    '''
    import os

    # Side note: a lot of OpenAI tutorials suggest that you embed your
    # OpenAI private key into the code, which is a terrible idea
    os.environ['OPENAI_API_KEY'] = oaikey  # Dummy val, so NBD

    os.environ['OPENAI_API_TYPE'] = oaitype
    os.environ['OPENAI_API_BASE'] = f'{host}:{port}/{rev}'

    # Tools such s langchain will check just for the presence of OPENAI_DEBUG
    if debug:
        os.environ['OPENAI_DEBUG'] = 'true'

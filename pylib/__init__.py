# SPDX-FileCopyrightText: 2023-present Oori Data <info@oori.dev>
#
# SPDX-License-Identifier: Apache-2.0
# ogbujipt

# ruff: noqa: F401,F403

from .__about__ import __version__


def oapi_first_choice_text(response):
    '''
    Given an OpenAI-compatible API simple completion response, return the first choice text
    '''
    import warnings
    warnings.warn(DeprecationWarning('Use the openai_api class\'s first_choice_text() method instead'))
    return response['choices'][0]['text']


def oapi_chat_first_choice_message(response):
    '''
    Given an OpenAI-compatible API chat completion response, return the first choice message content
    '''
    import warnings
    warnings.warn(DeprecationWarning('Use the openai_chat_api class\'s first_choice_message() method instead'))
    return response['choices'][0]['message']['content']

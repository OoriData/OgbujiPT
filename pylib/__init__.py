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
    return response['choices'][0]['text']


def oapi_chat_first_choice_message(response):
    '''
    Given an OpenAI-compatible API chat completion response, return the first choice message content
    '''
    return response['choices'][0]['message']['content']

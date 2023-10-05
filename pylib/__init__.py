# SPDX-FileCopyrightText: 2023-present Oori Data <info@oori.dev>
#
# SPDX-License-Identifier: Apache-2.0
# ogbujipt

# ruff: noqa: F401,F403

from .__about__ import __version__


def oapi_first_choice_text(response):
    '''
    Given an OpenAI-compatible API response, return the first choice response text
    '''
    return response['choices'][0]['text']


def oapi_first_choice_content(response):
    '''
    Given an OpenAI-compatible API response, return the first choice response content
    '''
    return response['choices'][0]['message']['content']
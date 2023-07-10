# SPDX-FileCopyrightText: 2023-present Uche Ogbuji <uche@ogbuji.net>
#
# SPDX-License-Identifier: Apache-2.0
# ogbujipt

def oapi_choice1_text(response):
    '''
    Given an OpenAI-compatible API response, return the first choice response text
    '''
    return response['choices'][0]['text']

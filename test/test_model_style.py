# SPDX-FileCopyrightText: 2023-present Oori Data <info@oori.dev>
# SPDX-License-Identifier: Apache-2.0
# test/test_model_style.py
# FIXME: For consistency let's migrate this to pytest
from unittest.mock import patch, Mock

from ogbujipt.prompting import model_style
from ogbujipt.llm_wrapper import openai_api

import unittest

# Should no longer be necessary, with mocking
#     try:
#         import httpx  # noqa
#     except ImportError:
#         raise RuntimeError('Needs httpx installed. Try pip install httpx')


class Test_hosted_model_openAI(unittest.TestCase):

    @patch('httpx.get')
    @patch('openai.base_url', 'https://api.openai.com/v1')  # Use your actual API URL
    def test_hosted_model_openai(self, mock_get):
        mock_response = Mock()
        mock_response.json.return_value = {
            'data': [{'id': 'model_1'}, {'id': 'model_2'}]
        }
        mock_get.return_value = mock_response
        
        llm = openai_api()
        result = llm.hosted_model()

        self.assertEqual(result, 'model_1')
        
        result = llm.available_models()

        self.assertEqual(result, ['model_1', 'model_2'])

    def test_model_style_from_name(self):

        result1 = model_style.model_style_from_name('path/wizardlm-13b-v1.0-uncensored.ggmlv3.q6_K.bin')
        self.assertEqual(str(result1), "[<style.WIZARD: 4>]")


if __name__ == '__main__':
    raise SystemExit("Attention! Run with pytest")

from ogbujipt.prompting import model_style
from unittest.mock import patch, Mock

import unittest

# def test_hosted_model_openai():
#     try:
#         import httpx  # noqa
#     except ImportError:
#         raise RuntimeError('Needs httpx installed. Try pip install httpx')
#     import openai
#     test_resp = httpx.get(f'{openai.api_base}/models').json()
#     test_fullmodel = [i['id'] for i in test_resp['data']]
#     assert test_fullmodel == model_style.hosted_model_openai()


class Test_hosted_model_openAI(unittest.TestCase):

    @patch('httpx.get')
    @patch('openai.api_base', 'https://api.openai.com/v1')  # Use your actual API URL
    def test_hosted_model_openai(self, mock_get):
        mock_response = Mock()
        mock_response.json.return_value = {
            'data': [{'id': 'model_1'}, {'id': 'model_2'}]
        }
        mock_get.return_value = mock_response
        
        result = model_style.hosted_model_openai()

        self.assertEqual(result, ['model_1', 'model_2'])
        

    def test_model_style_from_name(self):

        result1 = model_style.model_style_from_name('path/wizardlm-13b-v1.0-uncensored.ggmlv3.q6_K.bin')
        self.assertEqual(str(result1), "[<style.WIZARD: 4>]")



if __name__ == '__main__':
    unittest.main()
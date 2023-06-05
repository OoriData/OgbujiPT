# ogbujipt.config

'''
Configuration & globally-relevant values
'''

import os

DEFAULT_API_HOST = 'http://127.0.0.1'  # Running on the same server
DEFAULT_API_PORT = '5001'  # or 5001?

DEBUG_API = True

# Note: a lot of OpenAI tutorials tell you to do this & it leads to people
# stuffing private APi keys into their code. Bad idea. In this case, these
# Are dummy values, so NBD
os.environ['OPENAI_API_TYPE'] = 'open_ai'
os.environ['OPENAI_API_KEY'] = '123'
os.environ['OPENAI_API_BASE'] = f'{DEFAULT_API_HOST}:{DEFAULT_API_PORT}/v1'

# The check is actually the presence of OPENAI_DEBUG
if DEBUG_API:
    os.environ['OPENAI_DEBUG'] = 'true'

# Don't include in import *. Could use __all__, but tricky with the nature of a config module
del os

# SPDX-FileCopyrightText: 2023-present Uche Ogbuji <uche@ogbuji.net>
#
# SPDX-License-Identifier: Apache-2.0
# ogbujipt.model_styles.alpaca

'''
Model style for Alpaca, Instruction based.

Works, for example, with:

* https://huggingface.co/NousResearch/Nous-Hermes-13b
'''

# from functools import partial
from typing import Optional, List, Mapping, Any, Union

# XXX Try out preambles to instructions, e.f. for jailbreaks?
ALPACA_PROMPT_TMPL = '''\
### Instruction:

{instru_inputs}

### Response:
'''


def prep_instru_inputs(instru, inputs=''):
    # ROundabout method needed pre Python 3.12 because of escaping limitations
    cr = '\n'
    return f'{instru}\n{"### Inputs:" + cr + inputs if inputs else "" }\n'

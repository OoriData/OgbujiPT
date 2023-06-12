# SPDX-FileCopyrightText: 2023-present Uche Ogbuji <uche@ogbuji.net>
#
# SPDX-License-Identifier: Apache-2.0
# ogbujipt.model_styles.vicuna

'''
Model style for Vicuña.

Works, for example, with Wizard-Vicuna models
'''

# from functools import partial

# XXX Try out preambles to instructions, e.f. for jailbreaks?
VICUNA_PROMPT_TMPL = '''\
### USER: 

{query}

### ASSISTANT:
'''

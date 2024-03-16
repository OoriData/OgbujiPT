# SPDX-FileCopyrightText: 2023-present Oori Data <info@oori.dev>
# SPDX-License-Identifier: Apache-2.0
# demo/function_calling.py
'''
Demonstrate the use of OpenAI-style function calling with OgbujiPT

python demo/function_calling.py

Requires `OPENAI_API_KEY` in the environment

Hard-codes the function spec, but you can generate it from a PyDantic schema

```py
from typing import List
from pydantic import BaseModel

# PyDantic schema from which we'll generate function spec
class ExecuteStepByStepPlan(BaseModel):
    title: str
    steps: List[str]

# Generate the function spec
FUNC_SPEC = ExecuteStepByStepPlan.schema()
```

What about non-OpenAI LLM hosts? There is ongoing work in several areas.
It requires properly fine-tuned models, the right systems prompts and also suppot by the host code
Useful discussion re llama-cpp-python: https://github.com/abetlen/llama-cpp-python/discussions/397
'''

from ogbujipt.llm_wrapper import openai_chat_api, prompt_to_chat

FUNC_SPEC = {
    'title': 'ExecuteStepByStepPlan',
    'type': 'object',
    'properties': {
        'headline': {'headline': 'Headline', 'type': 'string'},
        'steps': {'title': 'Steps', 'type': 'array', 'items': {'type': 'string'}},
    },
    'required': ['headline', 'steps'],
}

# Requires OPENAI_API_KEY in environment
llm_api = openai_chat_api(model='gpt-3.5-turbo')

messages = prompt_to_chat('Explain how to poach an egg, step by step')

functions=[
        {
          'name': 'handle_steps_from_user_query',
          'description': 'Respond to a user query by specifying a series of steps',
          'parameters': FUNC_SPEC
        }
]

function_call={'name': 'handle_steps_from_user_query'}

resp = llm_api.call(messages=messages, functions=functions, function_call=function_call)
fc = resp.choices[0].message.function_call

if fc:
    print('Function to be called: ' + fc.name)
    print('Function call arguments: ' + fc.arguments)
else:
    print('No function call issued')

# SPDX-FileCopyrightText: 2023-present Oori Data <info@oori.dev>
# SPDX-License-Identifier: Apache-2.0
# ogbujipt/demo/function_calling.py
'''
Demonstrate the use of OpenAI-style function calling with OgbujiPT


python demo/function_calling.py --apibase=http://localhost:8000

You can alternatively use OpenAI by using the --openai param

from typing import List
from pydantic import BaseModel

# PyDantic schema from which we'll generate function spec
class ExecuteStepByStepPlan(BaseModel):
    title: str
    steps: List[str]

# Generate the function spec
FUNC_SPEC = ExecuteStepByStepPlan.schema()
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

llm_api = openai_chat_api(model='gpt-4')

messages = prompt_to_chat('Explain how to poach an egg')

functions=[
        {
          'name': 'handle_steps_from_user_query',
          'description': 'Respond to a user query by specifying a series of steps',
          'parameters': FUNC_SPEC
        }
]

function_call={'name': 'handle_steps_from_user_query'}

resp = llm_api(messages=messages, functions=functions, function_call=function_call)
# print(resp.choices[0].message.function_call)

print('Function to be called: ' + resp.choices[0].message.function_call.name)
print('Function call arguments: ' + resp.choices[0].message.function_call.arguments)

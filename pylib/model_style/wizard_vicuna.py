# SPDX-FileCopyrightText: 2023-present Uche Ogbuji <uche@ogbuji.net>
#
# SPDX-License-Identifier: Apache-2.0
# ogbujipt.model_styles.wizard_vicuna

import re
from functools import partial
from typing import Optional, List, Mapping, Any, Union

from langchain.prompts import StringPromptTemplate
from langchain.schema import AgentAction, AgentFinish
from langchain.agents import Tool, LLMSingleActionAgent, AgentOutputParser


WV_TEMPLATE = '''\
USER:Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of {tool_names}
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input  or the final conclusion to your thoughts


Begin!

Question: {input}
ASSISTANT: {agent_scratchpad}'''


class wv_prompt_template(StringPromptTemplate):
    template: str
    tools: List[Tool]

    def format(self, **kwargs) -> str:
        # Get the intermediate steps (AgentAction, Observation tuples)
        # Format them for Wizard-Vicuna
        intermediate_steps = kwargs.pop('intermediate_steps')
        thoughts = ''
        for action, observation in intermediate_steps:
            thoughts += '\Thought:'+action.log
            thoughts += f'\nObservation: {observation}\nThought: '
        # Set the agent_scratchpad variable to that value
        kwargs['agent_scratchpad'] = thoughts
        # Format the provided list of tools
        kwargs['tools'] = '\n Thought: '.join(
            [f'{tool.name}: {tool.description}' for tool in self.tools])
        # Get just the tool names
        kwargs['tool_names'] = ', '.join([tool.name for tool in self.tools])
        return self.template.format(**kwargs)


wv_prompt = partial(
    wv_prompt_template,
    template=WV_TEMPLATE,
    # Omit `agent_scratchpad`, `tools`, & `tool_names`, which are generated dynamically
    # Include the needed `intermediate_steps`
    input_variables=['input', 'intermediate_steps']
)


class wv_output_parser(AgentOutputParser):
    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        llm_output = split_text(str(llm_output))
        # Check if agent should finish
        if 'Final Answer:' in llm_output:
            return AgentFinish(
                # Return values is generally always a dictionary with a single `output` key
                # It is not recommended to try anything else at the moment :)
                return_values={'output': llm_output.split('Final Answer:')[-1].strip()},
                log=llm_output,
            )

        # Parse out the action and action input
        # FIXME: Switch to a regex implementation that's less susceptible to injection attack
        regex = r'Action\s*\d*\s*:(.*?)\nAction\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)'
        match = re.search(regex, llm_output, re.DOTALL)
        if not match:
            # raise ValueError(f'Could not parse LLM output: `{llm_output}`')
            return AgentFinish(
                # Return values is generally always a dictionary with a single `output` key
                # It is not recommended to try anything else at the moment :)
                return_values={'output': llm_output.split('Final Answer:')[0].strip()},
                log=llm_output,
            )
        else:
            action = match.group(1).strip()
            action_input = match.group(2)
        # Return the action and action input
        return AgentAction(tool=action, tool_input=action_input.strip().strip('"'), log=llm_output)


# Wizard-Vicuna-style models need the 'Observation' stop token
wv_agent_factory = partial(LLMSingleActionAgent, stop=['\nObservation'])


def split_text(text):
    blocks = text.split('Page:')
    print('blocks', blocks)
    if len(blocks) < 1:
        if len(blocks) < 2:
            first_block = blocks[0].strip()+'\n Page: '+blocks[1].strip()
        else:
            first_block = blocks[0].strip
            print('first_block', first_block)
    else:
        first_block = text
        print('first_block', first_block)
    return first_block


# SPDX-FileCopyrightText: 2023-present Uche Ogbuji <uche@ogbuji.net>
#
# SPDX-License-Identifier: Apache-2.0
# ogbujipt.model_styles.async_helper

'''
Coroutines to make it a little easier to multitask LLM access
using Python asyncio
'''

import asyncio
import concurrent.futures
from functools import partial

import openai


async def schedule_llm_call(callable, *args, **kwargs):
    '''
    Schedule task long-running/blocking LLM requests in a separate process,
    wrapped to work well in an asyncio event loop

    Basically hides away a bunch of the multiprocessing webbing

    e.g. `llm_task = asyncio.create_task(schedule_llm_call(llm, prompt))`

    Can then use asyncio.wait(), asyncio.gather(), etc. with `llm_task`
    '''
    # Link up the current async event loop for multiprocess execution
    loop = asyncio.get_running_loop()
    executor = concurrent.futures.ProcessPoolExecutor()
    # Need to partial execute to get in any kwargs for the target callable
    prepped_callable = partial(callable, **kwargs)
    # Spawn a separate process for the LLM call
    response = await loop.run_in_executor(
        executor, prepped_callable, *args)
    return response


# FIXME: Add all arguments for OpenAI API generation functions here
def openai_api_surrogate(prompt, api_func=openai.Completion.create, **kwargs):
    '''
    Wrapper around OpenAI API generation functions. Needed for use
    in multiprocessing because it seems when the openai library gets
    re-imported after the process fork, important attributes such as
    api_base & api_key get reset
    '''
    # Reset API details, relevant when emulating OpenAI
    openai.api_base = kwargs['api_base']
    openai.api_key = kwargs['api_key']
    # Send other, provided args to the generation function
    return api_func(prompt=prompt, **kwargs)

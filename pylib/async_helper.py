# SPDX-FileCopyrightText: 2023-present Uche Ogbuji <uche@ogbuji.net>
# SPDX-License-Identifier: Apache-2.0
# ogbujipt.model_styles.async_helper

'''
Coroutines to make it a little easier to multitask LLM access
using Python asyncio
'''
import sys

import asyncio
import concurrent.futures
from functools import partial


async def schedule_callable(callable, *args, **kwargs):
    '''
    Schedule long-running/blocking function call in a separate process,
    wrapped to work well in an asyncio event loop

    Basically hides away a bunch of the multiprocessing webbing

    e.g. `llm_task = asyncio.create_task(schedule_callable(llm, prompt))`

    Can then use asyncio.wait(), asyncio.gather(), etc. with `llm_task`

    Args:
        callable (callable): Callable to be scheduled

    Returns:
        response: Response object
    '''
    # Link up the current async event loop for multiprocess execution
    loop = asyncio.get_running_loop()
    executor = concurrent.futures.ProcessPoolExecutor()
    # Need to partial execute to get in any kwargs for the target callable
    prepped_callable = partial(callable, **kwargs)
    # Spawn a separate process for the LLM call
    response = await loop.run_in_executor(executor, prepped_callable, *args)
    return response


# FIXME: Add all arguments for OpenAI API generation functions here
def openai_api_surrogate(prompt, api_func=None, **kwargs):
    '''
    Wrapper around OpenAI API generation functions. Needed for use
    in multiprocessing because it seems when the openai library gets
    re-imported after the process fork, important attributes such as
    api_base & api_key get reset

    Args:
        prompt (str): Prompt string for the LLM to ingest

        api_func: API function to utilize

    Returns:
        api_func: Result of OpenAI API call
    '''
    import openai

    api_func = api_func or openai.Completion.create

    trimmed_kwargs = {}
    for k in kwargs:
        if k in OPENAI_GLOBALS:
            setattr(openai, k, kwargs[k])
        else:
            trimmed_kwargs[k] = kwargs[k]
    # Send other, provided args to the generation function
    return api_func(prompt=prompt, **trimmed_kwargs)


# Extracted from https://github.com/openai/openai-python/blob/main/openai/__init__.py
OPENAI_GLOBALS = ['api_key', 'api_key_path', 'api_base', 'organization', 'api_type', 'api_version',
                 'proxy', 'app_info', 'debug', 'log']


def save_openai_api_params():
    '''
    openai package uses globals for a lot of its parameters, including the mandatory api_key.
    In some circs, e.g. multiprocessing, these should be saved for re-set when the module is re-imported.
    '''
    import openai

    params = {}
    # model also carried as a user convenience
    for k in OPENAI_GLOBALS + ['model']:
        if hasattr(openai, k):
            params[k] = getattr(openai, k)
    return params


async def console_progress_indicator(pause=0.5, file=sys.stderr):
    '''
    Simple progress indicator for the console. Just prints dots.

    pause - seconds between each dot printed to console, default half a sec

    file - file for dots output, default STDERR
    '''
    while True:
        print('.', end='', flush=True, file=file)
        await asyncio.sleep(pause)

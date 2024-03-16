# SPDX-FileCopyrightText: 2023-present Oori Data <info@oori.dev>
# SPDX-License-Identifier: Apache-2.0
# ogbujipt/demo/multiprocess.py
'''
Advanced demo showing quick chat with LLMs, with 3 simultaneous requests.
It can be configured to chat with 3 separate LLMs, or 3 instances of the same LLM.
Includes a progress indicator dislay while the LLM instances are generating.
Takes advantage of Python's asyncio, and also multiprocess, which requires some finesse,
mostly handled by OgbujiPT. Works even when the LLM host in use doesn't suport asyncio.

```sh
python demo/multiprocess.py --apibase=http://my-llm-host:8000
```

Also allows you to use the actual OpenAI ChatGPT service, by specifying --openai
'''
import asyncio
from pathlib import Path

import click

from ogbujipt import word_loom
from ogbujipt.llm_wrapper import openai_chat_api, prompt_to_chat
from ogbujipt.async_helper import console_progress_indicator


def file_path_here():
    '''Cross-platform Python trick to get the path to this very file'''
    from inspect import getsourcefile
    from os.path import abspath

    return abspath(getsourcefile(lambda: 0))


# Load language material, such as prompt templates, from an external file—word loom format
HERE = Path(file_path_here()).parent
with open(HERE / Path('language.toml'), mode='rb') as fp:
    TEXTS = word_loom.load(fp)


async def async_main(requests_info):
    '''
    Main entry point for asyncio
    model_topics - list of (llm_wrapped, topic, temperature) tuples
    llmtemp - LLM temperature; hard-coded across all (see https://beta.openai.com/docs/api-reference/completions/create)
    '''
    # Pro tip: When creating asyncio tasks be mindful to not lose references to tasks,
    # lest they get garbage collected, which sows chaos. asyncio.TaskGroup (new in Python 3.11)
    # is often a better alternative, but waits for all tasks to complete whereas we're done once
    # the LLM generation tasks are complete
    indicator_task = asyncio.create_task(console_progress_indicator())
    llm_tasks = [asyncio.create_task(llm(prompt_to_chat(msg), temperature=temp, max_tokens=1024))
                 for (llm, msg, temp) in requests_info]
    llm_messages = [msg for (llm, msg, temp) in requests_info]
    # Need to gather to make sure all LLM tasks are completed
    gathered_llm_tasks = asyncio.gather(*llm_tasks)
    done, _ = await asyncio.wait((indicator_task, gathered_llm_tasks), return_when=asyncio.FIRST_COMPLETED)

    # Completed task will from gather() of llm_tasks; results in original task arg order
    results = zip(llm_messages, next(iter(done)).result())
    for msg, resp in results:
        print(f'Result re {msg}')
        # resp is an instance of openai.openai_object.OpenAIObject, with lots of useful info
        print('\nFull response data from LLM:\n', resp)
        # Just the response text
        print('\nResponse text from LLM:\n\n', resp.first_choice_text)
        print('-'*80)


# Command line arguments defined in click decorators
@click.command()
@click.option('--apibase', default='http://127.0.0.1:8000', help='OpenAI API base URL')
@click.option('--llmtemp', default='0.9', type=float, help='LLM temperature')
@click.option('--openai', is_flag=True, default=False, type=bool,
              help='Use live OpenAI API. If you use this option, you must have '
              '"OPENAI_API_KEY" defined in your environmnt')
@click.option('--model', default='', type=str, 
              help='OpenAI model to use (see https://platform.openai.com/docs/models)')
def main(apibase, llmtemp, openai, model):
    # Use OpenAI API if specified, otherwise emulate with supplied URL info
    if openai:
        oapi = openai_chat_api(model=(model or 'gpt-3.5-turbo'))
    else:
        oapi = openai_chat_api(model=model, base_url=apibase)

    # Separate models or params—e.g. temp—for each LLM request is left as an exercise 😊
    requests_info = [
        (oapi, TEXTS['test_prompt_joke'].format(topic=t), llmtemp)
        for t in ('wild animals', 'vehicles', 'space aliens')]

    asyncio.run(async_main(requests_info))


if __name__ == '__main__':
    # CLI entry point. Also protects against re-execution of main() after process fork
    # viz https://docs.python.org/3/library/multiprocessing.html#multiprocessing-safe-main-import
    main()

'''
Advanced demo showing quick chat with an LLM, but with 3 simultaneous requests,
and also a separate, progress indicator dislay while the LLM instances are generating.
Key is taking advantage of Python's asyncio, and also multiprocess, which requires some finesse,
to work even when the LLM framework in use doesn't suport asyncio.
Luckily `ogbujipt.async_helper` comes in handy.

```sh
python demo/alpaca_multitask_fix_xml.py --host=http://my-llm-host --port=8000
```

Also allows you to use the actual OpenAI ChatGPT service, by specifying --openai
'''
import sys
import asyncio

import openai

from ogbujipt.async_helper import schedule_callable, openai_api_surrogate
from ogbujipt import config
from ogbujipt.prompting.basic import format
from ogbujipt.prompting.model_style import ALPACA_DELIMITERS

model_params = {}


async def indicate_progress(pause=0.5):
    '''
    Simple progress indicator for the console. Just prints dots.
    '''
    while True:
        print('.', end='', flush=True)
        await asyncio.sleep(pause)


openai_globals = ['api_base', 'api_key', 'api_key_path']


def cache_openai_api_params():
    params = {}
    for k in openai_globals:
        if hasattr(openai, k):
            params[k] = getattr(openai, k)
    return params


def openai_api_surrogate(prompt, api_func=openai.Completion.create, **kwargs):
    # Reset API details, relevant when emulating OpenAI
    trimmed_kwargs = {}
    for k in kwargs:
        if k in openai_globals:
            setattr(openai, k, kwargs[k])
        else:
            trimmed_kwargs[k] = kwargs[k]
    # Send other, provided args to the generation function
    return api_func(prompt=prompt, **trimmed_kwargs)


class llm_request:
    tasks = {}

    def __init__(self, topic):
        self.topic = topic

    def wrap(self):
        prompt = format(f'Tell me a funny joke about {self.topic}', delimiters=ALPACA_DELIMITERS)

        self.task = asyncio.create_task(
            schedule_callable(openai_api_surrogate, prompt, model='text-ada-001', **cache_openai_api_params()))
        llm_request.tasks[self.task] = self
        return self.task


async def async_main():
    topics = ['wild animals', 'vehicles', 'space aliens']

    # model_params = dict(
    #     max_tokens=60,  # Limit number of generated tokens
    #     top_p=1,  # AKA nucleus sampling; can increase generated text diversity
    #     frequency_penalty=0,  # Favor more or less frequent tokens
    #     presence_penalty=1,  # Prefer new, previously unused tokens
    #     )
    indicator_task = asyncio.create_task(indicate_progress())
    # Notice the pattern of passing in the callable iself, then the params
    # You can't just do, say llm(prompt) because that will actually
    # call the function & block on the LLM request
    llm_requests = [llm_request(t) for t in topics]
    llm_tasks = [req.wrap() for req in llm_requests]
    # Need to gather to make sure all LLm tasks are completed
    gathered_llm_tasks = asyncio.gather(*llm_tasks)
    done, _ = await asyncio.wait((indicator_task, gathered_llm_tasks), return_when=asyncio.FIRST_COMPLETED)

    # Only completed task will be from the gather() of llm_tasks, and it has results in original order
    results = zip(llm_requests, next(iter(done)).result())
    for req, resp in results:
        print(f'Result re {req.topic}')
        print(resp)


def main():
    openai.model = 'text-ada-001'
    # Just hardcode these params
    model_params['llmtemp'], model_params['model'] = 1, 'text-ada-001'
    openai.api_key_path = sys.argv[1]
    # openai_api = config.openai_live(model=model, debug=True)
    # model_params['api_key_path'] = openai.api_key_path
    asyncio.run(async_main())


if __name__ == '__main__':
    # Re-entry control. Don't want main() executed on re-import
    main()

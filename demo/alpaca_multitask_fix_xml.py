'''
Advanced demo using an LLM to repair data (XML), like
alpaca_simple_fix_xml.py
but demonstrating asyncio by running a separate, progress indicator task
in the background while the LLM is generating. Should work even
if the LLM framework in use doesn't suport asyncio,
thanks to ogbujipt.async_helper

You need access to an OpenAI-like service. Default assumption is that you
have a self-hosted framework such as llama-cpp-python or text-generation-webui
running. Say it's at my-llm-host:8000, you can do:

python demo/alpaca_multitask_fix_xml.py --host=http://my-llm-host --port=8000

Also allows you to use the actual OpenAI service, by specifying --openai
'''

import asyncio

import click

from ogbujipt.async_helper import schedule_llm_call, openai_api_surrogate
from ogbujipt import config
from ogbujipt.prompting.basic import context_build
from ogbujipt.prompting.model_style import ALPACA_INSTRUCT_DELIMITERS

DOTS_SPACING = 0.5  # Number of seconds between each dot printed to console


# Could probably use something like tqdm.asyncio, if we wanted to be fancy
async def indicate_progress(pause=DOTS_SPACING):
    '''
    Simple progress indicator for the console. Just prints dots.
    '''
    while True:
        print('.', end='', flush=True)
        await asyncio.sleep(pause)


async def async_main(openai_params):
    '''
    Schedule one task to do a long-running/blocking LLM request, and another
    to run a progress indicator in the background
    '''
    BAD_XML_CODE = '''\
<earth>
<country><b>Russia</country></b>
<capital>Moscow</capital>
</Earth>'''

    prompt = context_build(
        'Correct the following XML to make it well-formed',
        contexts=BAD_XML_CODE,
        delimiters=ALPACA_INSTRUCT_DELIMITERS)
    print(prompt, '\n')

    # Customize parameters for model behavior
    # More info: https://platform.openai.com/docs/api-reference/completions
    model_params = dict(
        max_tokens=60,  # Limit number of generated tokens
        top_p=1,  # AKA nucleus sampling; can increase generated text diversity
        frequency_penalty=0,  # Favor more or less frequent tokens
        presence_penalty=1,  # Prefer new, previously unused tokens
        )
    model_params.update(openai_params)

    # Pro tip: When creating tasks with asyncio.create_task be mindful to not
    # accidentally lose references to tasks, lest they get garbage collected,
    # which sows chaos. In some cases asyncio.TaskGroup (new in Python 3.11)
    # is a better alternative, but we can't use them in this case because
    # they wait for all tasks to complete whereas we're done once only
    # the LLM generation task is complete
    indicator_task = asyncio.create_task(indicate_progress())
    # Notice the pattern of passing in the callable iself, then the params
    # You can't just do, say llm(prompt) because that will actually
    # call the function & block on the LLM request
    llm_task = asyncio.create_task(
        schedule_llm_call(openai_api_surrogate, prompt, **model_params))
    tasks = [indicator_task, llm_task]
    done, _ = await asyncio.wait(
        tasks, return_when=asyncio.FIRST_COMPLETED
        )

    # Instance of openai.openai_object.OpenAIObject, with lots of useful info
    retval = next(iter(done)).result()
    print('\nResponse from LLM: ', retval.choices[0].text)


# Command line arguments defined in click decorators
@click.command()
@click.option('--host', default='http://127.0.0.1', help='OpenAI API host')
@click.option('--port', default='8000', help='OpenAI API port')
@click.option('--llmtemp', default='0.1', type=float, help='LLM temperature')
@click.option('--openai', is_flag=True, default=False, type=bool,
              help='Use live OpenAI API. If you use this option, you must have ' +
              '"OPENAI_API_KEY" defined in your environmnt')
@click.option('--model', default='', type=str, help='OpenAI model to use (see https://platform.openai.com/docs/models)')
def main(host, port, llmtemp, openai, model):
    # Use OpenAI API if specified, otherwise emulate with supplied host, etc.
    if openai:
        assert not (host or port), 'Don\'t use --host or --port with --openai'
        model = model or 'text-davinci-003'
        openai_api = config.openai_live(
            model=model, debug=True)
    else:
        # For now the model param is most useful in conjunction with --openai
        model = model or config.HOST_DEFAULT
        openai_api = config.openai_emulation(
            host=host, port=port, model=model, debug=True)

    # Preserve the provided temperature setting
    openai_api.params.temperature = llmtemp
    asyncio.run(async_main(openai_api.params))


if __name__ == '__main__':
    # CLI entry point
    # Also protects against multiple launching of the overall program
    # when a child process imports this
    # viz https://docs.python.org/3/library/multiprocessing.html#multiprocessing-safe-main-import
    main()

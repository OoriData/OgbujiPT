'''
Quick demo—Send an Alpaca-compatible LLM bad XML & ask it to correct

Needs access to an OpenAI-like service. Default assumption is that you
have a self-hosted framework such as llama-cpp-python or text-generation-webui
running. Say it's at my-llm-host:8000, you can do:

python demo/alpaca_simple_fix_xml.py --host=http://my-llm-host --port=8000

You can also use OpenAI by using the --openai param
'''

import click

from ogbujipt import oapi_first_choice_text
from ogbujipt.config import openai_live, openai_emulation
from ogbujipt.prompting.basic import format
from ogbujipt.prompting.model_style import ALPACA_INSTRUCT_INPUT_DELIMITERS


# Command line arguments defined in click decorators
@click.command()
@click.option('--host', default='http://127.0.0.1', help='OpenAI API host')
@click.option('--port', default='8000', help='OpenAI API port')
@click.option('--llmtemp', default='0.1', type=float, help='LLM temperature')
@click.option('--openai', is_flag=True, default=False, type=bool,
              help='Use live OpenAI API. If you use this option, you must have '
              '"OPENAI_API_KEY" defined in your environmnt')
@click.option('--model', default='', type=str, 
              help='OpenAI model to use (see https://platform.openai.com/docs/models)')
def main(host, port, llmtemp, openai, model):
    # Use OpenAI API if specified, otherwise emulate with supplied host, etc.
    if openai:
        assert not (host or port), 'Don\'t use --host or --port with --openai'
        openai_api = openai_live(debug=True)
        model = model or 'text-davinci-003'
    else:
        openai_api = openai_emulation(host=host, port=port)
        model = model or 'LOCAL'

    BAD_XML_CODE = '''\
<earth>
<country><b>Russia</country></b>
<capital>Moscow</capital>
</Earth>'''

    prompt = format(
        'Correct the given XML to make it well-formed',
        contexts= BAD_XML_CODE,
        preamble='You are a helpful assistant, '
        'who answers questions briefly, in 1st grade language',
        delimiters=ALPACA_INSTRUCT_INPUT_DELIMITERS)
    print(prompt, '\n')

    response = openai_api.Completion.create(
        model=model,  # Model (Required)
        prompt=prompt,  # Prompt (Required)
        temperature=llmtemp,  # Temp (Default 1)
        max_tokens=100,  # Max Token length of generated text (Default 16)
        top_p=1,  # AKA nucleus sampling; can increase diversity of the
                  # generated text (Default 1)
        frequency_penalty=0,    # influences the model to favor more or less
                                # frequent tokens (Default 0)
        presence_penalty=1  # influences the model to use new tokens it has
                            # not yet used (Default 0)
        )

    # Response is a json-like object; extract the text
    print('\nFull response data from LLM:\n', response)

    # Response is a json-like object; 
    # just get back the text of the response
    response_text = oapi_first_choice_text(response)
    print('\nResponse text from LLM:\n\n', response_text)


# CLI entry point
if __name__ == '__main__':
    main()

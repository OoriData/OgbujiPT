# SPDX-FileCopyrightText: 2023-present Oori Data <info@oori.dev>
# SPDX-License-Identifier: Apache-2.0
# ogbujipt/demo/alpaca_simple_fix_xml.py
'''
Quick demo—Send an Alpaca-compatible LLM bad XML & ask it to correct

Needs access to an OpenAI-like service. Default assumption is that you
have a self-hosted framework such as llama-cpp-python or text-generation-webui
running. Say it's at my-llm-host:8000, you can do:

python demo/simple_fix_xml.py --apibase=http://localhost:8000

You can alternatively use OpenAI by using the --openai param

Uses a simple completion model, so it probably shouldn't be used with the actual OpenAI service,
now that they've deprecated simple completion in favor of chat completion.
'''

import click

from ogbujipt.llm_wrapper import openai_api
from ogbujipt.prompting.basic import format
from ogbujipt.prompting.model_style import ALPACA_INSTRUCT_INPUT_DELIMITERS


# Command line arguments defined in click decorators
@click.command()
@click.option('--apibase', default='http://127.0.0.1:8000', help='OpenAI API base URL')
@click.option('--llmtemp', default='0.1', type=float, help='LLM temperature')
@click.option('--openai', is_flag=True, default=False, type=bool,
              help='Use live OpenAI API. If you use this option, you must have '
              '"OPENAI_API_KEY" defined in your environmnt')
@click.option('--model', default='', type=str, 
              help='OpenAI model to use (see https://platform.openai.com/docs/models)')
def main(apibase, llmtemp, openai, model):
    # Use OpenAI API if specified, otherwise emulate with supplied host, etc.
    if openai:
        assert not apibase, 'Don\'t use --apibase with --openai'
        oapi = openai_api(model=(model or 'gpt-3.5-turbo'))
    else:
        oapi = openai_api(model=model, base_url=apibase)

    BAD_XML_CODE = '''\
<earth>
<country><b>Russia</country></b>
<capital>Moscow</capital>
</Earth>'''

    # Recommend you use Word Loom for storing, looking up, managing and formatting prompts
    # Left raw for this simple example
    prompt = format(
        'Correct the given XML to make it well-formed',
        contexts= BAD_XML_CODE,
        preamble='You are a helpful assistant, '
        'who answers questions briefly, in 1st grade language',
        delimiters=ALPACA_INSTRUCT_INPUT_DELIMITERS)
    print(prompt, '\n')

    response = oapi.call(
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

    # Response is a json-like object; just get back the text of the response
    print('\nResponse text from LLM:\n\n', response.first_choice_text)


# CLI entry point
if __name__ == '__main__':
    main()

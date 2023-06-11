import click
from langchain import OpenAI

from ogbujipt.config import openai_emulation
from ogbujipt.model_style.alpaca import prep_instru_inputs, ALPACA_PROMPT_TMPL

@click.command()
@click.option('--host', default='http://127.0.0.1', help='OpenAI API host')
@click.option('--port', default='8000', help='OpenAI API port')
def main(host, port):
    openai_emulation(host=host, port=port)
    # Set up the API connector
    llm = OpenAI(temperature=0.1)

    BAD_XML_CODE = '''\
    <earth>    
    <country><b>Russia</country></b>
    <capital>Moscow</capital>
    </Earth>'''

    instru_inputs = prep_instru_inputs(
        'Correct the following XML to make it well-formed',
        inputs=BAD_XML_CODE
        )

    prompt = ALPACA_PROMPT_TMPL.format(instru_inputs=instru_inputs)
    # print(prompt)

    response = llm(prompt)
    print(response)


if __name__ == '__main__':
    main()

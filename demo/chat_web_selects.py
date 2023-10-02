# SPDX-FileCopyrightText: 2023-present Oori Data <info@oori.dev>
# SPDX-License-Identifier: Apache-2.0
# ogbujipt/demo/chat_web_selects.py
'''
Advanced, "Chat my docs" demo, using docs from the web

Download one or more web pages and query an LLM using them as context.
Works especially well with airoboros self-hosted LLM.

Vector store: Qdrant - https://qdrant.tech/
    Alternatives: pgvector, Chroma, Faiss, Weaviate, etc.
Text to vector (embedding) model: 
    Alternatives: https://www.sbert.net/docs/pretrained_models.html / OpenAI ada002

Needs access to an OpenAI-like service. Default assumption is self-hosted
via e.g. llama-cpp-python or text-generation-webui

Assume for the following it's at host my-llm-host, port 8000

pip install prerequisites, in addition to OgbujiPT cloned dir:

click sentence_transformers qdrant-client httpx html2text amara3.xml

```sh
python demo/chat_web_selects.py --apibase http://my-llm-host:8000 "www.newworldencyclopedia.org/entry/Igbo_People"
```

An example question might be "Who are the neighbors of the Igbo people?"
'''
# en.wikipedia.org/wiki/Igbo_people|ahiajoku.igbonet.com/2000/|en.wikivoyage.org/wiki/Igbo_phrasebook"
import asyncio
import os

import click
import httpx
import html2text

from ogbujipt import config, oapi_chat_first_choice_message
from ogbujipt.llm_wrapper import openai_chat_api, prompt_to_chat
from ogbujipt.prompting import format, ALPACA_INSTRUCT_DELIMITERS
from ogbujipt.text_helper import text_splitter
from ogbujipt.embedding_helper import qdrant_collection


# Avoid re-entrace complaints from huggingface/tokenizers
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# default https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
DOC_EMBEDDINGS_LLM = 'all-MiniLM-L12-v2'

COLLECTION_NAME = 'chat-web-selects'
USER_PROMPT = 'What do you want to know from the site(s)?\n'

# Hard-code for demo
EMBED_CHUNK_SIZE = 200
EMBED_CHUNK_OVERLAP = 20
DOTS_SPACING = 0.2  # Number of seconds between each dot printed to console


async def indicate_progress(pause=DOTS_SPACING):
    while True:
        print('.', end='', flush=True)
        await asyncio.sleep(pause)


async def read_site(url, collection):
    # Crude check; good enough for demo
    if not url.startswith('http'): url = 'https://' + url  # noqa E701
    print('Downloading & processing', url)
    async with httpx.AsyncClient(verify=False) as client:
        resp = await client.get(url)
        html = resp.content.decode(resp.encoding or 'utf-8')

    text = html2text.html2text(html)

    # Split text into chunks
    chunks = text_splitter(text, chunk_size=EMBED_CHUNK_SIZE,
                           chunk_overlap=EMBED_CHUNK_OVERLAP, separator='\n')

    # print('\n\n'.join([ch[:100] for ch in chunks]))
    # Crude—for demo. Set URL metadata for all chunks to doc URL
    metas = [{'url': url}]*len(chunks)
    # Add the text to the collection
    collection.update(texts=chunks, metas=metas)
    print(f'{collection.count()} chunks added to collection')


async def async_main(oapi, sites):
    # Automatic download of embedding model from HuggingFace
    # Seem to be reentrancy issues with HuggingFace; defer import
    from sentence_transformers import SentenceTransformer
    embedding_model = SentenceTransformer(DOC_EMBEDDINGS_LLM)
    # Sites fuel in-memory Qdrant vector DB instance
    collection = qdrant_collection(COLLECTION_NAME, embedding_model)

    # Download & process sites in parallel, loading their content & metadata into a knowledgebase
    url_task_group = asyncio.gather(*[
        asyncio.create_task(read_site(site, collection)) for site in sites.split('|')])
    indicator_task = asyncio.create_task(indicate_progress())
    tasks = [indicator_task, url_task_group]
    done, _ = await asyncio.wait(
        tasks, return_when=asyncio.FIRST_COMPLETED)

    # Main chat loop
    done = False
    while not done:
        print()
        user_question = input(USER_PROMPT)
        if user_question.strip() == 'done':
            break

        docs = collection.search(user_question, limit=4)

        print(docs)
        if docs:
            # Collects "chunked_doc" into "gathered_chunks"
            gathered_chunks = '\n\n'.join(
                doc.payload['_text'] for doc in docs if doc.payload)

            # Build prompt the doc chunks as context
            # FIXME: Move this to Word Loom
            prompt = format(
                f'Given the context, {user_question}\n\n'
                f'Context: """\n{gathered_chunks}\n"""\n',
                preamble='### SYSTEM:\nYou are a helpful assistant, who answers '
                'questions directly and as briefly as possible. '
                'If you cannot answer with the given context, just say so.\n',
                delimiters=ALPACA_INSTRUCT_DELIMITERS)

            print(prompt)

            # The rest is much like in demo/alpaca_multitask_fix_xml.py
            model_params = dict(
                max_tokens=1024,  # Limit number of generated tokens
                top_p=1,  # AKA nucleus sampling; can increase generated text diversity
                frequency_penalty=0,  # Favor more or less frequent tokens
                presence_penalty=1,  # Prefer new, previously unused tokens
                temperature=0.1
                )

            indicator_task = asyncio.create_task(indicate_progress())
            llm_task = oapi.wrap_for_multiproc(prompt_to_chat(prompt), **model_params)
            tasks = [indicator_task, llm_task]
            done, _ = await asyncio.wait(
                tasks, return_when=asyncio.FIRST_COMPLETED)

            # Instance of openai.openai_object.OpenAIObject, with lots of useful info
            retval = next(iter(done)).result()
            print(type(retval))
            # Response is a json-like object; extract the text
            print('\nFull response data from LLM:\n', retval)

            # response is a json-like object; 
            # just get back the text of the response
            response_text = oapi_chat_first_choice_message(retval)
            print('\nResponse text from LLM:\n\n', response_text)


# Command line arguments defined in click decorators
@click.command()
@click.option('--apibase', default='http://127.0.0.1:8000', help='OpenAI API base URL')
@click.option('--openai-key',
              help='OpenAI API key. Leave blank to specify self-hosted model via --host & --port')
@click.option('--model', default='', type=str, 
              help='OpenAI model to use (see https://platform.openai.com/docs/models).'
              'Use only with --openai-key')
@click.argument('sites')
def main(apibase, openai_key, model, sites):
    # Use OpenAI API if specified, otherwise emulate with supplied URL info
    if openai_key:
        model = model or 'gpt-3.5-turbo'
        oapi = openai_chat_api(api_key=openai_key, model=model)
    else:
        model = model or config.HOST_DEFAULT
        oapi = openai_chat_api(model=model, api_base=apibase)

    asyncio.run(async_main(oapi, sites))


if __name__ == '__main__':
    main()

# SPDX-FileCopyrightText: 2023-present Oori Data <info@oori.dev>
# SPDX-License-Identifier: Apache-2.0
# ogbujipt/demo/chat_web_selects.py
'''
Advanced, "Chat my docs" demo, using docs from the web

Download one or more web pages and query an LLM using them as context.
Works especially well with airoboros self-hosted LLM.

Vector store: Qdrant - https://qdrant.tech/
    Alternatives: pgvector & Chroma (built-in support from OgbujiPT), Faiss, Weaviate, etc.
Text to vector (embedding) model: 
    Alternatives: https://www.sbert.net/docs/pretrained_models.html / OpenAI ada002

Needs access to an OpenAI-like service. Default assumption is self-hosted
via e.g. llama-cpp-python or text-generation-webui

Assume for the following it's at host localhost, port 8000.
MAKE SURE YOU USE A WORKING SERVER BECAUSE THIS IS A DEMO & THE ERROR HANDLING IS SIMPLISTIC

pip install prerequisites, in addition to OgbujiPT cloned dir:

click sentence_transformers qdrant-client httpx html2text amara3.xml

```sh
python demo/chat_web_selects.py --apibase http://localhost:8000 "www.newworldencyclopedia.org/entry/Igbo_People"
```

An example question might be "Who are the neighbors of the Igbo people?"

You can tweak it with the following command line options:
--verbose - print more information while processing (for debugging)
--limit (max number of chunks to retrieve for use as context)
--chunk-size (characters per chunk, while prepping to create embeddings)
--chunk-overlap (character overlap between chunks, while prepping to create embeddings)
--question (The user question; if None (the default), prompt the user interactively)
'''
# en.wikipedia.org/wiki/Igbo_people|ahiajoku.igbonet.com/2000/|en.wikivoyage.org/wiki/Igbo_phrasebook"
import asyncio
import os

import click
import httpx
import html2text

from ogbujipt.llm_wrapper import openai_chat_api, prompt_to_chat
from ogbujipt.text_helper import text_split_fuzzy
from ogbujipt.embedding.qdrant import collection


# Avoid re-entrace complaints from huggingface/tokenizers
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# default https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
DOC_EMBEDDINGS_LLM = 'all-MiniLM-L12-v2'

COLLECTION_NAME = 'chat-web-selects'
USER_PROMPT = 'What do you want to know from the site(s)?\n'

DOTS_SPACING = 0.2  # Number of seconds between each dot printed to console


async def indicate_progress(pause=DOTS_SPACING):
    while True:
        print('.', end='', flush=True)
        await asyncio.sleep(pause)


async def read_site(url, coll, chunk_size, chunk_overlap):
    # Crude check; good enough for demo
    if not url.startswith('http'): url = 'https://' + url  # noqa E701
    print('Downloading & processing', url)
    async with httpx.AsyncClient(verify=False) as client:
        resp = await client.get(url)
        html = resp.content.decode(resp.encoding or 'utf-8')

    text = html2text.html2text(html)

    # Split text into chunks
    chunks = text_split_fuzzy(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap, separator='\n')

    # print('\n\n'.join([ch[:100] for ch in chunks]))
    # Crude—for demo. Set URL metadata for all chunks to doc URL
    metas = [{'url': url}]*len(chunks)
    # Add the text to the collection
    coll.update(texts=chunks, metas=metas)
    print(f'{coll.count()} chunks added to collection')


async def async_main(oapi, sites, verbose, limit, chunk_size, chunk_overlap, question):
    # Automatic download of embedding model from HuggingFace
    # Seem to be reentrancy issues with HuggingFace; defer import
    from sentence_transformers import SentenceTransformer
    embedding_model = SentenceTransformer(DOC_EMBEDDINGS_LLM)
    # Sites fuel in-memory Qdrant vector DB instance
    coll = collection(COLLECTION_NAME, embedding_model)

    # Download & process sites in parallel, loading their content & metadata into a knowledgebase
    url_task_group = asyncio.gather(*[
        asyncio.create_task(read_site(site, coll, chunk_size, chunk_overlap)) for site in sites.split('|')])
    indicator_task = asyncio.create_task(indicate_progress())
    tasks = [indicator_task, url_task_group]
    done, _ = await asyncio.wait(
        tasks, return_when=asyncio.FIRST_COMPLETED)

    # Main chat loop
    done = False
    while not done:
        print()
        if question:
            user_question = question
        else:
            user_question = input(USER_PROMPT)
        if user_question.strip() == 'done':
            break

        docs = coll.search(user_question, limit=limit)
        if verbose:
            print(docs)
        if docs:
            # Collects "chunked_doc" into "gathered_chunks"
            gathered_chunks = '\n\n'.join(
                doc.payload['_text'] for doc in docs if doc.payload)

            # Build system message with the doc chunks as provided context
            # In practice we'd use word loom to load the propts, as demoed in multiprocess.py
            sys_prompt = '''\
You are a helpful assistant, who answers questions directly and as briefly as possible.
Consider the following context and answer the user\'s question.
If you cannot answer with the given context, just say so.\n\n'''
            sys_prompt += gathered_chunks + '\n\n'
            messages = prompt_to_chat(user_question, system=sys_prompt)
            if verbose:
                print('-'*80, '\n', messages, '\n', '-'*80)

            # The rest is much like in demo/alpaca_multitask_fix_xml.py
            model_params = dict(
                max_tokens=1024,  # Limit number of generated tokens
                top_p=1,  # AKA nucleus sampling; can increase generated text diversity
                frequency_penalty=0,  # Favor more or less frequent tokens
                presence_penalty=1,  # Prefer new, previously unused tokens
                temperature=0.1)

            indicator_task = asyncio.create_task(indicate_progress())
            llm_task = asyncio.Task(oapi(messages, **model_params))
            tasks = [indicator_task, llm_task]
            done, _ = await asyncio.wait(
                tasks, return_when=asyncio.FIRST_COMPLETED)

            # proper cleanup of indicator task, which will still be pending/running
            indicator_task.cancel()

            # Instance of openai.openai_object.OpenAIObject, with lots of useful info
            retval = next(iter(done)).result()
            if verbose:
                print(type(retval))
            # Response is a json-like object; extract the text
            if verbose:
                print('\nFull response data from LLM:\n', retval)

            # just get back the text of the response
            print('\nResponse text from LLM:\n\n', retval.first_choice_text)


# Command line arguments defined in click decorators
@click.command()
@click.option('--verbose/--no-verbose', default=False)
@click.option('--chunk-size', type=int, default=200,
              help='Number of characters to include per chunk')
@click.option('--chunk-overlap', type=int, default=20,
              help='Number of characters to overlap at the edges of chunks')
@click.option('--limit', default=4, type=int,
              help='Maximum number of chunks matched against the posed question to use as context for the LLM')
@click.option('--openai-key',
              help='OpenAI API key. Leave blank to specify self-hosted model via --host & --port')
@click.option('--apibase', default='http://127.0.0.1:8000', help='OpenAI API base URL')
@click.option('--model', default='', type=str, 
              help='OpenAI model to use (see https://platform.openai.com/docs/models).'
              'Use only with --openai-key')
@click.option('--question', default=None, help='The question to ask (or prompt for one)')
@click.argument('sites')
def main(verbose, chunk_size, chunk_overlap, limit, openai_key, apibase, model, question, sites):
    # Use OpenAI API if specified, otherwise emulate with supplied URL info
    if openai_key:
        oapi = openai_chat_api(api_key=openai_key, model=(model or 'gpt-3.5-turbo'))
    else:
        oapi = openai_chat_api(model=model, base_url=apibase)

    asyncio.run(async_main(oapi, sites, verbose, limit, chunk_size, chunk_overlap, question))


if __name__ == '__main__':
    main()

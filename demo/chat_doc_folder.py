# SPDX-FileCopyrightText: 2023-present Oori Data <info@oori.dev>
# SPDX-License-Identifier: Apache-2.0
# ogbujipt/demo/chat_doc_folder.py
'''
"Chat my docs" demo, using docs in a folder. Skill level: intermediate

Indexes a folder full of Word, PDF & Markdown documents, then query an LLM using these as context.

Vector store: Chroma - https://docs.trychroma.com/getting-started
    Alternatives: pgvector & Qdrant (built-in support from OgbujiPT), Faiss, Weaviate, etc.
Text to vector (embedding) model: 
    Alternatives: https://www.sbert.net/docs/pretrained_models.html / OpenAI ada002

Needs access to an OpenAI-like service. This can be private/self-hosted, though.
OgbujiPT's sister project Toolio would work - https://github.com/OoriData/Toolio
via e.g. llama-cpp-python, text-generation-webui, Ollama

Prerequisites, in addition to OgbujiPT (warning: chromadb installs a lot of stuff):

```sh
pip install click sentence_transformers chromadb docx2python PyPDF2 PyCryptodome
```

Assume for the following the server is running on localhost, port 8000.

```sh
python demo/chat_doc_folder.py --apibase http://localhost:8000 demo/sample_docs 
```

Sample query: Tell me about the Calabar Kingdom

You can always check the retrieval using `--verbose`

You can specify your document directory, and/or tweak it with the following command line options:
--verbose - print more information while processing (for debugging)
--limit (max number of chunks to retrieve for use as context)
--chunk-size (characters per chunk, while prepping to create embeddings)
--chunk-overlap (character overlap between chunks, while prepping to create embeddings)
--question (The user question; if None (the default), prompt the user interactively)
'''
import os
import asyncio
from pathlib import Path

import click
import chromadb
from docx2python import docx2python
from PyPDF2 import PdfReader

from ogbujipt.llm_wrapper import openai_chat_api, prompt_to_chat
from ogbujipt.text_helper import text_split_fuzzy

# Avoid re-entrace complaints from huggingface/tokenizers
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

COLLECTION_NAME = 'chat-doc-folder'
USER_PROMPT = 'What do you want to know from the documents?\n'


# Note: simple demo mode, so no duplicate management, cleanup, etc of the chroma DB
# You can always add self.coll.delete_collection(name='chat_doc_folder'), but take case!
class vector_store:
    '''Encapsulates Chroma the vector store and its parameters (e.g. for doc chunking)'''
    def __init__(self, chunk_size, chunk_overlap):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chroma_client = chromadb.Client()
        self.coll = self.chroma_client.get_or_create_collection(name='chat_doc_folder')
        self.id_counter = 0

    def text_split(self, text):
        return text_split_fuzzy(text, chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap, separator='\n')

    def new_docid(self):
        self.id_counter += 1
        nid = f'id-{self.id_counter}'
        return nid

    def update(self, chunks, metas):
        ids = [self.new_docid() for c in chunks]
        self.coll.add(documents=chunks, ids=ids, metadatas=metas)

    def search(self, q, limit=None):
        results = self.coll.query(query_texts=[q], n_results=limit)
        print(results['documents'][0][0][:100])
        return results['documents'][0]


def read_word_doc(fpath, store):
    '''Convert a single word doc to text, split into chunks & add these to vector store'''
    print('Processing as Word doc:', fpath)  # e.g. 'path/to/file.docx'
    with docx2python(fpath) as docx_content:
        doctext = docx_content.text
    chunks = list(store.text_split(doctext))
    metas = [{'source': str(fpath)}]*len(chunks)
    store.update(chunks, metas=metas)


def read_pdf_doc(fpath, store):
    '''Convert a single PDF to text, split into chunks & add these to vector store'''
    print('Processing as PDF:', fpath)  # e.g. 'path/to/file.pdf'
    pdf_reader = PdfReader(fpath)
    doctext = ''.join((page.extract_text() for page in pdf_reader.pages))
    chunks = list(store.text_split(doctext))
    metas = [{'source': str(fpath)}]*len(chunks)
    store.update(chunks, metas=metas)


def read_text_or_markdown_doc(fpath, store):
    '''Split a single text or markdown file into chunks & add these to vector store'''
    print('Processing as text:', fpath)  # e.g. 'path/to/file.txt'
    with open(fpath) as docx_content:
        doctext = docx_content.read()
    chunks = list(store.text_split(doctext))
    metas = [{'source': str(fpath)}]*len(chunks)
    store.update(chunks, metas=metas)


async def async_main(oapi, docs, verbose, limit, chunk_size, chunk_overlap, question):
    store = vector_store(chunk_size, chunk_overlap)

    for fname in docs.iterdir():
        # print(fname, fname.suffix)
        if fname.suffix in ['.doc', '.docx']:
            read_word_doc(fname, store)
        elif fname.suffix == '.pdf':
            read_pdf_doc(fname, store)
        elif fname.suffix in ['.txt', '.md', '.mdx']:
            read_text_or_markdown_doc(fname, store)

    # Main chat loop
    done = False
    while not done:
        print('\n')
        if question:
            user_question = question
        else:
            user_question = input(USER_PROMPT)
        if user_question.strip() == 'done':
            break

        docs = store.search(user_question, limit=limit)
        if verbose:
            print(docs)
        if docs:
            gathered_chunks = '\n\n'.join(docs)
            # Build system message with the approx nearest neighbor chunks as provided context
            # In practice we'd use word loom to load the propts, as demoed in multiprocess.py
            sys_prompt = '''\
You are a helpful assistant, who answers questions directly and as briefly as possible.
Consider the following context and answer the user\'s question.
If you cannot answer with the given context, just say so.\n\n'''
            sys_prompt += gathered_chunks + '\n\n'
            messages = prompt_to_chat(user_question, system=sys_prompt)
            if verbose:
                print('-'*80, '\n', messages, '\n', '-'*80)

            model_params = dict(
                max_tokens=1024,  # Limit number of generated tokens
                top_p=1,  # AKA nucleus sampling; can increase generated text diversity
                frequency_penalty=0,  # Favor more or less frequent tokens
                presence_penalty=1,  # Prefer new, previously unused tokens
                temperature=0.1)

            retval = await oapi(messages, **model_params)
            if verbose:
                print(type(retval))
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
@click.argument('docs', type=click.Path(file_okay=False, dir_okay=True, writable=False, path_type=Path))
def main(verbose, chunk_size, chunk_overlap, limit, openai_key, apibase, model, question, docs):
    # Use OpenAI API if specified, otherwise emulate with supplied URL info
    if openai_key:
        oapi = openai_chat_api(api_key=openai_key, model=(model or 'gpt-3.5-turbo'))
    else:
        oapi = openai_chat_api(model=model, base_url=apibase)

    asyncio.run(async_main(oapi, docs, verbose, limit, chunk_size, chunk_overlap, question))


if __name__ == '__main__':
    main()

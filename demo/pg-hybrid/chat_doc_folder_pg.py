#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2023-present Oori Data <info@oori.dev>
# SPDX-License-Identifier: Apache-2.0
# demo/pg-hybrid/chat_doc_folder.py
'''
"Chat my docs" demo using PG hybrid search. Skill level: intermediate

Indexes a folder full of Word, PDF & Markdown documents, then query an LLM using these as context.

Vector store: PostgreSQL with pgvector - https://github.com/pgvector/pgvector
    Uses OgbujiPT's hybrid search combining dense vector search with sparse BM25 retrieval
Text to vector (embedding) model: 
    Alternatives: https://www.sbert.net/docs/pretrained_models.html / OpenAI ada002
PDF to text [PyPDF2](https://pypdf2.readthedocs.io/en/3.x/)
    Alternative: [Docling](https://github.com/DS4SD/docling)

Needs access to an OpenAI-like service. This can be private/self-hosted, though.
OgbujiPT's sister project Toolio would work - https://github.com/OoriData/Toolio
via e.g. llama-cpp-python, text-generation-webui, Ollama

Prerequisites, in addition to OgbujiPT (or you can just use the `mega` package):

```sh
uv pip install fire sentence-transformers docx2python PyPDF2 PyCryptodome  # or uv pip install -U ".[mega]"
```

PostgreSQL with pgvector must be running. See README.md in this directory for setup.

Assume for the following the LLM server is running on localhost, port 8000.

```sh
python chat_doc_folder_pg.py --docs=../sample-docs --apibase=http://localhost:8000
```

Sample query: "Tell me about the Calabar Kingdom"

You can always check the retrieval using --verbose

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

import fire
from docx2python import docx2python
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer

from ogbujipt.llm.wrapper import openai_chat_api, prompt_to_chat
from ogbujipt.text.splitter import text_split_fuzzy
from ogbujipt.store.postgres import DataDB
from ogbujipt.retrieval import BM25Search, HybridSearch, SimpleDenseSearch

# Avoid re-entrace complaints from huggingface/tokenizers
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

USER_PROMPT = 'What do you want to know from the documents?\n'

# Database connection parameters (can be overridden via environment variables)
PG_DB_NAME = os.environ.get('PG_DB_NAME', 'hybrid_demo')
PG_DB_HOST = os.environ.get('PG_DB_HOST', 'localhost')
PG_DB_PORT = int(os.environ.get('PG_DB_PORT', '5432'))
PG_DB_USER = os.environ.get('PG_DB_USER', 'demo_user')
PG_DB_PASSWORD = os.environ.get('PG_DB_PASSWORD', 'demo_pass_2025')

# Default embedding model
DEFAULT_EMBEDDING_MODEL = 'all-MiniLM-L6-v2'


class VectorStore:
    '''Encapsulates PostgreSQL DataDB and hybrid search with chunking parameters'''
    def __init__(self, chunk_size, chunk_overlap, embedding_model, table_name='chat_doc_folder'):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embedding_model = embedding_model
        self.table_name = table_name
        self.kb_db = None
        self.hybrid_search = None

    async def initialize(self):
        '''Initialize database connection and hybrid search'''
        # Connect to PostgreSQL
        self.kb_db = await DataDB.from_conn_params(
            embedding_model=self.embedding_model,
            table_name=self.table_name,
            db_name=PG_DB_NAME,
            host=PG_DB_HOST,
            port=PG_DB_PORT,
            user=PG_DB_USER,
            password=PG_DB_PASSWORD,
            itypes=['vector'],  # Create HNSW index for fast vector search
            ifuncs=['cosine']
        )

        # Drop existing table if present (for clean demo)
        if await self.kb_db.table_exists():
            await self.kb_db.drop_table()

        # Create fresh table
        await self.kb_db.create_table()

        # Initialize hybrid search
        self.hybrid_search = HybridSearch(
            strategies=[
                SimpleDenseSearch(),  # Dense vector search
                BM25Search(k1=1.5, b=0.75, epsilon=0.25)  # Sparse BM25 search
            ],
            k=60  # RRF constant
        )

    def text_split(self, text):
        return text_split_fuzzy(text, chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap, separator='\n')

    async def update(self, chunks, metas):
        '''Insert chunks into the database'''
        content_list = [(chunk, meta) for chunk, meta in zip(chunks, metas)]
        await self.kb_db.insert_many(content_list)

    async def search(self, q, limit=None):
        '''Search using hybrid search and return content strings'''
        results = []
        async for result in self.hybrid_search.execute(
            query=q,
            backends=[self.kb_db],
            limit=limit or 4
        ):
            results.append(result.content)
        return results


async def read_word_doc(fpath, store):
    '''Convert a single word doc to text, split into chunks & add these to vector store'''
    print('Processing as Word doc:', fpath)  # e.g. 'path/to/file.docx'
    with docx2python(fpath) as docx_content:
        doctext = docx_content.text
    chunks = list(store.text_split(doctext))
    metas = [{'source': str(fpath)}]*len(chunks)
    await store.update(chunks, metas=metas)


async def read_pdf_doc(fpath, store):
    '''Convert a single PDF to text, split into chunks & add these to vector store'''
    print('Processing as PDF:', fpath)  # e.g. 'path/to/file.pdf'
    pdf_reader = PdfReader(fpath)
    doctext = ''.join((page.extract_text() for page in pdf_reader.pages))
    chunks = list(store.text_split(doctext))
    metas = [{'source': str(fpath)}]*len(chunks)
    await store.update(chunks, metas=metas)


async def read_text_or_markdown_doc(fpath, store):
    '''Split a single text or markdown file into chunks & add these to vector store'''
    print('Processing as text:', fpath)  # e.g. 'path/to/file.txt'
    with open(fpath) as docx_content:
        doctext = docx_content.read()
    chunks = list(store.text_split(doctext))
    metas = [{'source': str(fpath)}]*len(chunks)
    await store.update(chunks, metas=metas)


async def async_main(oapi, docs, verbose, limit, chunk_size, chunk_overlap, question, embedding_model):
    store = VectorStore(chunk_size, chunk_overlap, embedding_model)
    await store.initialize()

    # Process all documents
    for fname in docs.iterdir():
        # print(fname, fname.suffix)
        if fname.suffix in ['.doc', '.docx']:
            await read_word_doc(fname, store)
        elif fname.suffix == '.pdf':
            await read_pdf_doc(fname, store)
        elif fname.suffix in ['.txt', '.md', '.mdx']:
            await read_text_or_markdown_doc(fname, store)

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

        docs = await store.search(user_question, limit=limit)
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


def main(
    docs,
    verbose=False,
    chunk_size=200,
    chunk_overlap=20,
    limit=4,
    openai_key=None,
    apibase='http://127.0.0.1:8000',
    model='',
    question=None,
    embedding_model=DEFAULT_EMBEDDING_MODEL
):
    '''
    Chat with documents using PG hybrid search.

    Args:
        docs: Path to directory containing documents (Word, PDF, Markdown, Text)
        verbose: Print more information while processing (for debugging)
        chunk_size: Number of characters to include per chunk
        chunk_overlap: Number of characters to overlap at the edges of chunks
        limit: Maximum number of chunks matched against the posed question to use as context for the LLM
        openai_key: OpenAI API key. Leave blank to specify self-hosted model via --apibase
        apibase: OpenAI API base URL (default: http://127.0.0.1:8000)
        model: OpenAI model to use (see https://platform.openai.com/docs/models). Use only with --openai-key
        question: The question to ask (or prompt for one if None)
        embedding_model: Sentence transformer model for embeddings (default: all-MiniLM-L6-v2)
    '''
    docs_path = Path(docs)
    if not docs_path.exists() or not docs_path.is_dir():
        raise ValueError(f'Document directory does not exist: {docs}')

    # Load embedding model
    print(f'\n📦 Loading embedding model: {embedding_model}…')
    embedding_model_instance = SentenceTransformer(embedding_model)
    print('   ✓ Model loaded!')

    # Use OpenAI API if specified, otherwise emulate with supplied URL info
    if openai_key:
        oapi = openai_chat_api(api_key=openai_key, model=(model or 'gpt-3.5-turbo'))
    else:
        oapi = openai_chat_api(model=model, base_url=apibase)

    asyncio.run(async_main(oapi, docs_path, verbose, limit, chunk_size, chunk_overlap, question, embedding_model_instance))


if __name__ == '__main__':
    fire.Fire(main)


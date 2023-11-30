For all these demos you need access to an OpenAI-like service. Default assumption is that you have a self-hosted framework such as llama-cpp-python or text-generation-webui running

# Simplest

## simple_fix_xml.py

Quick demo, sending a Llama or Alpaca-compatible LLM some bad XML & asking it to make corrections.

# Intermediate

## multiprocess.py

Intermediate demo asking an LLM multiple simultaneous riddles on various topics,
running a separate, progress indicator task in the background, using asyncio.
Works even if the LLM framework suport asyncio, thanks to ogbujipt.async_helper 

## function_calling.py

[OpenAI-style function calling](https://openai.com/blog/function-calling-and-other-api-updates) allows the LLM to specify a structured response to a user's query in the form of details of a function to be called to complete the response, or take an action. Though developed by OpenAI, function calling can be made available through other LLM tools, though progress on this
is just emerging.

* [In llama-cpp-python](https://llama-cpp-python.readthedocs.io/en/latest/server/#function-calling)

You might also be interested in the reAct approach, which Oori has covered in their blog:

* [reAct (reasoning Action)](https://www.oori.dev/blog/2023/10/react/)

See also: [Low-level experiments for agent/tool interaction with locally-hosted LLMs #42](https://github.com/OoriData/OgbujiPT/discussions/42)

# Advanced

## qa_discord.py

<img width="555" alt="image" src="https://github.com/uogbuji/OgbujiPT/assets/279982/82121324-a930-4b2c-ab26-d8a3c6a50f54">

Demo of a Discord chatbot with an LLM back end

Demonstrates:
* Async processing via ogbujipt.async_helper
* Discord API integration
* Client-side serialization of requests to the server. An important
consideration until more sever-side LLM hosting frameworks reliablly
support multiprocessing

## chat_web_selects.py

Command-line "chat my web site" demo, but supporting self-hosted LLM.

Definitely a good idea for you to understand demos/alpaca_multitask_fix_xml.py
before swapping this in.

Vector store: Qdrant - https://qdrant.tech/

Supports multiple web URLs, specified on cmdline

## chat_pdf_streamlit_ui.py

<img width="970" alt="image" src="https://github.com/uogbuji/OgbujiPT/assets/279982/57b479a9-2dbc-4d65-ac19-e954df2a21d0">

Retrieval Augmented Generation (RAG) AKA "Chat my PDF" demo, supporting self-hosted LLM. Definitely a good idea for you to understand
alpaca_multitask_fix_xml.py & chat_web_selects.py
before swapping this in.

RAG technique links: [[overview]](https://www.promptingguide.ai/techniques/rag) | [[paper]](https://arxiv.org/abs/2005.11401)

UI: Streamlit - streamlit.io
    Alternatives: web UI (e.g. Flask), native UI (e.g. Tkinter), cmdline UI (e.g. Blessings)
Vector store: [Qdrant](https://qdrant.tech/)
    Alternatives: pgvector, Chroma, Faiss, Weaviate, etc.
PDF to text: PyPDF2
    Alternatives: pdfplumber, pdfreader
    Note: [Calibre](https://github.com/kovidgoyal/calibre) can be used for e-book cleaning 
Text to vector (embedding) model: [all-MiniLM-L12-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L12-v2)
    Alternatives: [See the sentence-transformers pretrained models list](https://www.sbert.net/docs/pretrained_models.html), / [OpenAI TE ADA 2](https://openai.com/blog/new-and-improved-embedding-model)

Single-PDF support, for now, to keep the demo code simple, 
though you can easily extend it to e.g. work with multiple docs
dropped in a directory

Note: manual used for above demo downloaded from Hasbro via [Board Game Capital](https://www.boardgamecapital.com/monopoly-rules.htm).

## PGvector_demo.py
Demo of the PGvector vector store functionality of OgbujiPT, which takes an initial sample collection of strings and performs a few example actions with them:

1. set up a table in the PGvector store
2. vectorizes and inserts the strings in the PGvector store
3. performs a perfect search for one of the sample strings
4. performs a fuzzy search for a word that is in one of the sample strings

At oori, the demo is run utilizing the [official PGvector docker container](https://hub.docker.com/r/ankane/pgvector) and the following docker compose:
```docker-compose
version: '3.1'

services:

  db:
    image: ankane/pgvector
    # restart: always
    environment:
      POSTGRES_USER: oori
      POSTGRES_PASSWORD: example
      POSTGRES_DB: PGv
    ports:
      - 5432:5432
    volumes:
      - ./pg_hba.conf:/var/lib/postgresql/pg_hba.conf

  adminer:
    image: adminer
    restart: always
    ports:
      - 8080:8080
```

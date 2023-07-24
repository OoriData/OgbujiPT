For all these demos you need access to an OpenAI-like service. Default assumption is that you have a self-hosted framework such as llama-cpp-python or text-generation-webui running

# Simplest

## simple_fix_xml.py

Quick demo, sending a Llama or Alpaca-compatible LLM some bad XML & asking it to make corrections.

# Intermediate

## multiprocess.py

Intermediate demo asking an LLM multiple simultaneous riddles on various topics,
running a separate, progress indicator task in the background, using asyncio.
Works even if the LLM framework suport asyncio, thanks to ogbujipt.async_helper 

# Advanced

## alpaca_simple_qa_discord.py

<img width="555" alt="image" src="https://github.com/uogbuji/OgbujiPT/assets/279982/82121324-a930-4b2c-ab26-d8a3c6a50f54">

Demo of a Discord chatbot with an LLM back end

Demonstrates:
* Async processing via ogbujipt.async_helper
* Discord API integration
* Client-side serialization of requests to the server. An important
consideration until more sever-side LLM hosting frameworks reliablly
support multiprocessing

## chat_web_selects.py

Simple, command-line "chat my web site" demo, but supporting self-hosted LLM.

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

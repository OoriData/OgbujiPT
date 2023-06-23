'''
Advanced, "Chat my PDF" demo, but using self-hosted LLM
definitely a good idea for you to understand demo/alpaca_multitask_fix_xml.py
before swapping this in.

UI: Streamlit - streamlit.io
Vector store: Qdrant - https://qdrant.tech/

Single-PDF support, for now, to keep the demo code simple, 
though you can easily extend it to e.g. work with multiple docs
dropped in a directory

Based on https://github.com/wafflecomposite/langchain-ask-pdf-local
but taking advantage of OgbujiPT

Uses ogbujipt.async_helper to demo multitasking ability you could use in a more real-world app

Prompts via langchain, rather than one of the OgbujiPT models styles

You need access to an OpenAI-like service. Default assumption is that you
have a self-hosted framework such as llama-cpp-python or text-generation-webui
running. Assume for the following it's at my-llm-host:8000

Prerequisites. From OgbujiPT cloned dir:.

```
pip install -upgrade .
pip install streamlit watchdog PyPDF2 PyCryptodome sentence_transformers qdrant-client tiktoken
```

Notice the -- to separate our program's cmdline args from streamlit's
streamlit run demo/chat_pdf_streamlit_ui.py -- --host=http://my-llm-host --port=8000
'''

import asyncio

# Could choose to use controls in streamlit rather than cmdline
import click

import streamlit as st
from PyPDF2 import PdfReader

import langchain
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from langchain.vectorstores import Qdrant
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain import OpenAI  # Using the API, though on a self-hosted LLM

from ogbujipt.async_helper import schedule_llm_call
from ogbujipt.config import openai_emulation

# Workaround for AttributeError: module 'langchain' has no attribute 'verbose'
# see: https://github.com/hwchase17/langchain/issues/4164
langchain.verbose = False

QA_CHAIN_TYPE = 'stuff'
PDF_USER_QUESTION_PROMPT = 'Ask a question about your PDF:'
# Not sure where we use this via OpenAI API. Not a kword to OpenAI initializer
STOP_WORDS = ['### Human:']
# FIXME: We probably want to get this value from the remote API. Figure out how
N_CTX = 2048

# FIXME: Parameterize these, perhaps in streamlit controls, also how many k docs to return
# Chunk size is the number of characters counted in the chunks
EMBED_CHUNK_SIZE = 500

# Model Chunk Overlap to connect ends of chunks together using 
# the last 100 chars and first 100 chars of two chunks
EMBED_CHUNK_OVERLAP = 100

# https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
DOC_EMBEDDINGS_LLM = 'all-MiniLM-L6-v2'

# FIXME: stop using an absolute path, grab from the internet or smth
# define throbber absolutely
throbber = '/Users/osi/dev/OgbujiPT/demo/assets/ooriThrobber.gif'


async def prep_pdf(pdf):
    '''
    Convert "pdf" into chunks according to chunk size and overlap
    Take "chunks" and vectorize it for SLLM lookup
    return "knowledge_base" as the vectorized sets of chunks
    '''
    pdf_reader = PdfReader(pdf)

    # Collect text from pdf
    text = ''
    for page in pdf_reader.pages:
        text += page.extract_text()

    # Split the text into chunks
    text_splitter = CharacterTextSplitter(
        separator='\n',
        chunk_size=EMBED_CHUNK_SIZE,
        chunk_overlap=EMBED_CHUNK_OVERLAP,
        length_function=len
    )
    chunks = text_splitter.split_text(text)

    # LLM will be downloaded from HuggingFace automatically
    embeddings = SentenceTransformerEmbeddings(model_name=DOC_EMBEDDINGS_LLM)

    # Create in-memory Qdrant instance for the embeddings
    knowledge_base = Qdrant.from_texts(
        chunks,
        embeddings,
        location=':memory:',
        collection_name='doc_chunks',
    )

    return knowledge_base


async def handle_user_q(kb, chain):
    '''
    Get a "user_question" from Streamlit
    Get the top K chunks relevant to the user's question
    Return the chunks and user question
    '''
    user_question = st.text_input(PDF_USER_QUESTION_PROMPT)

    docs = None
    if user_question:
        # Return the "k" most relevant objects to the "user_question" as "docs"
        docs = kb.similarity_search(user_question, k=4)

        # Calculating prompt (takes time and can optionally be removed)
        prompt_len = chain.prompt_length(docs=docs, question=user_question)
        print(f'Prompt len: {prompt_len}')
        # Used to catch and prevent long wait times and a potential crash 
        # in a situation where model is fed too many chars
        if prompt_len > N_CTX:
            st.write(
                "Prompt length is more than n_ctx. This will likely fail."
                    "Increase model's context, reduce chunk size or question length,"
                    "or change k to retrieve fewer docs."
                    "Debug info on docs for the prompt:"
            )
            st.text_area(repr(docs))

    return docs, user_question


# Schedule one task to do a long-running/blocking LLM request, and another to chat the PDF
async def async_main(llm):
    # Doc strings turn into streamlit headers
    '''
    Oori — Ask your PDF 📄💬
    '''
    chain = load_qa_chain(llm, chain_type=QA_CHAIN_TYPE)

    # XXX: Do we need to tweak this for Nous-Hermes? It does seem to be working OK
    # Patching qa_chain prompt template to better suit the stable-vicuna model
    # see https://huggingface.co/TheBloke/stable-vicuna-13B-GGML#prompt-template
    if "Helpful Answer:" in chain.llm_chain.prompt.template:
        chain.llm_chain.prompt.template = (
            f"### Human:{chain.llm_chain.prompt.template}".replace(
                "Helpful Answer:", "\n### Assistant:"
            )
        )

    # create file upload box on Streamlit, then set "pdf" as the pdf that the user uploads
    pdf = st.file_uploader("Upload a PDF", type=["pdf"])

    if pdf:
        # Show throbber, and vectorize the PDF, and setup for similarity search
        with st.empty():
            st.image(throbber)
            kb = await prep_pdf(pdf)
            
            docs, user_q = await handle_user_q(kb, chain)

        if docs:
            # Set up LLM prompt task
            llm_task = asyncio.create_task(
                schedule_llm_call(
                    chain.run, 
                    input_documents=docs, 
                    question=user_q
                    )
                )
            
            # Show throbber, and send LLM prompt
            with st.empty():
                st.image(throbber)
                tasks = [llm_task]
                done, _ = await asyncio.wait(
                    tasks, return_when=asyncio.FIRST_COMPLETED
                    )
                response = next(iter(done)).result()
                
                # Write reponse to console and Streamlit
                print('\nResponse from LLM: ', response)
                st.write(response)


# See e.g. demo/alpaca_multitask_fix_xml.py for more explanation
@click.command()
@click.option('--host', default='http://127.0.0.1', help='OpenAI API host')
@click.option('--port', default='8000', help='OpenAI API port')
@click.option('--temp', default='0.1', type=float, help='LLM temperature')
def main(host, port, temp):
    # Callback just to stream output to stdout, can be removed
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

    # Emulate OpenAI API with "host" and "port" for LLM call
    openai_emulation(host=host, port=port)
    llm = OpenAI(
        temperature=temp,
        callback_manager=callback_manager,
        verbose=True,
        )

    # Page setup
    st.set_page_config(
        page_title="Ask your PDF",
        page_icon="📄💬",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    
    asyncio.run(async_main(llm))


if __name__ == "__main__":
    # TODO: Look into isolating hugginface's one time per process setup routines
    main()

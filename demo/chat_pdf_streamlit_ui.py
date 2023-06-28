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
# LLM max context size
# FIXME: Probably want to get this value from the remote API. Figure out how
N_CTX = 2048

# FIXME: Parameterize these, perhaps in streamlit controls, also how many k docs to return
# Chunk size is the number of characters counted in the chunks
EMBED_CHUNK_SIZE = 500

# Model Chunk Overlap to connect ends of chunks together using 
# the last 100 chars and first 100 chars of two chunks
EMBED_CHUNK_OVERLAP = 100

# https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
DOC_EMBEDDINGS_LLM = 'all-MiniLM-L6-v2'

# Creating the throbber.gif as a data URL
# TODO: use a less archaic format than GIF; perhaps some sort of animatable vector image
throbber = 'data:image/gif;base64,R0lGODlhgACAAKEDAGwZDPPVSf///wAAACH/C05FVFNDQVBFMi4wAwEAAAAh+QQFFAADACwAAAAAgACAAAACm5yPqcvtD6OctNqLs968+w+G4kiW5omm6sq27gvH8kzX9o3n+s73/g8MCofEovGITCqXzKbzCY1Kp9Sq9YrNarfcrvcLDovH5LL5jE6r1+y2+w2Py+f0uv2Oz+v3/L7/DxgoOEhYaHiImKi4yNjo+AgZKTlJWWl5iZmpucnZ6fkJGio6SlpqeoqaqrrK2ur6ChsrO0tba3uLK1YAACH5BAUUAAMALAAAAACAAIAAAAL+nI+py+0Po5y02jsAwLz7P2kaSJZmJYrnyrJp2sYy9r7zjTd1nff5vvMJW0Dg8EgqFpFMmnLZjEaeT6mVUc0or1xD1rvtSr8HsnhoLofPwjTCzY7B3+u4bJ7A2z/6fH1f0qcgCBjyZwhV6EGIdahIwajj+AgR6WD5iHk5STnIyaEZF1r5STk6VVp4KrHa1MqayvaKaNQ5Cxl7dYuSG7Vr8bsSDNyLVnx3/JOMnKi7PDPs1NwVzTstVg17fZb90B34vBh+9O23DVgOdq46ThvUaV4r125VnX4zfI/zq6+8TvwPXrx34gIKHMijQz9y4RYi0eSQiaWIrnpR9BXr4pj+UhrrcerobBtILmZGUltjEluYlCcTsfQ47WVFkfQOLpgjsw3Hmja1GPQpr6e0oN542jqWk8iypCdGMU3S7mlBokMJCr1pVCpAqlOtXgXqFepPRfaMSjSLdawotI3UqnTbNeFRuHzYMuOKz64LvajoGvNrginInCZfskwpc2TSjk8vao2oVR3eu2HR8U07eU+6yJgrN7287zLnTYBJlebGdnTfzFs9X52lGtdpya6/0pZ7IXbuZ7qr4tbG2jZCG+5+Czcd/Dbx47KD9wb373ngSdI/X6surA72vVC2K6WSnHlr8CPEcye/wfx38Oopz26PPDx8he/nr4ZhPzT+/PoJ0/NX9l+AFhQAACH5BAUUAAMALAAAAACAAIAAAAL+nI+py+0Po5y02jsAwLz7P2kaSJZmJYrnyrJp2sYy9r7zjTd1nff5vvMJW0Dg8EgqFpFMmnLZjEaeT6mVUc0or1xD1rvtSr8HsnhoLofPwjTCzY7B3+u4bJ7A2z/6fH1f0qcgCBjyZwhV6IEXEOBAqIh1iNDY+DgZCcFYaamDmSmZuMDJeSkK6nmaQEpqaoSa+srAyuoaBDv4OUBLa8uDq6ZKydsbe4u7SVwcKguarLy8AHn1DB2dKxxXXbur7GtTuN16AP0Nsydeuupt/MuWXjnLznzcBd8pT2yuYq9b/vCPnrsx/uYBNIitGZN7jiYElKYLB0MLDxPWa1NQ34X+in6yzZjIgSMdj0Qy8vogMpjCOyavhUTYcaWLltxIpARDEgRIEze15Oyw80TPaYhkkoPJE6nKi3xojpMxNCKFoFCV4jSKwqm6HFF/PqCKoytWTVrjHRHLdEpZfGet+hwLkWRPrm4hgUWCdmA7cPlOcsnLV2BgBXPbahR8Lu7YwnjrTrr71/EpyF0AJ6YsxjI/zJUly+JsRfOIkYvdShG9waLedYcze057FbYBxkJQf13b8IptsnJN99ittnfrxsNjykbMr69LH8Cn4ibuF/noC6BXNKf+/Pfr1diFR59xHWj2qsVJH+9eunyJ8DrHC90+2ET1Cuzlu0cJPzFL78v1N+ZPfsN8DtQnx330/TedDwKy9p1q8fWw4GwIpraQgQNOaMV8BKJhIYP9xcZdE5htWCF/NXl4ooP6iQEWiQSZ+FQ36oH44BkgucgFQzj2A6M1uSl2nja4ffhWkHbouBWQIWaC5I8qAghMkUvKOKOUNSKz1j4JRmnelA0aN2WU22hJIZfSlUmYWWeayVtpZLIZHFx7rQjnnFDGaWSdNNL5pp5F5UmUn30E6qeVfC4VZqF2bonolYq2yRShjzaqn6STUrqZVJdCSoWjm/7ZKaOfOhGqqKOS2ump9lGh6gmstuqqV7BaIOesqJpqa3u45toer74qUAAAIfkECRQAAwAsAAAAAIAAgAAAAv6cj6nL7Q+jnLTaOwDAvPs/aRpIlmYliufKsmnaxjL2vvONN3Wd9/m+8wlbQODwSCoWkUyactmMRp5PqZVRzSivXEPWu+1KvweyeGguh8/CNMLNjsHf67hsnsDbP/p8fV/SpyAIGPJnCFXogRcQ4ECoiHWI0Nj4OBkJwVhpqYOZKZm4wMl5KQrqeZpASmpqhJr6ysDK6hoEO/g5QEtry4OrpkrJ2xt7i7tJXBwqC5qsvLwAefUMHZ0rHFddu6vsa1O43XoA/Q2zJ1666m38y5ZeOcvOfNwF3ylPbK5ir1v+8I+euzH+5gE0iK0Zk3uOJgSUpgsHQwsPE9ZrU1Dfhf6KfrLNmNhAgAAHHOl4JJKRlwORIkki7KjwTsprCViydKlRILiPM7kxsGkTp8p2O1GeLHkAKFChNE3GDNRz3E+lSxsgBXOSA8ipVKvmG6rzHNSjLxF07crUJ8SsFLYuOHs2rdS1Ty24VQAXrlx1Yflpjcr3bV69VssGqzsFcLyQhPPuXdx3hF3F+ATHTUr4a9PDFzVRbsgVbc3MowxjRWxxoIKrAxxbFq1ZbeqiRMWWzvma6krSq01ryXp39OXdw2+DpduZs+p1uPHyZly8d3OYyYObfU4ctvHNpy9axxw9guvYc2eL/W5gfAX10o+b54e+NXbx8w2w/hKfPf3ww/6mO/X+WXa6TaBff+5Rt9xvqLFWYG5KPVbZe5IhlyA5vjV4HX8W+qccbRJuUBiH6dUnn4b2+SZIfvNh2I2ICiYXGYjkBeZceCzeF9GHEILmoFclatcedy9W+ICKl92IYo61+bWdbMINNuCMkFHoIQoBQgdlUCEe+B+RbV0ZWpY77jMhH2D2aGKLXHZoGwhGIuniNIgseCGca3bn5SJn1hhlk+UhWKUJb2opZYSAtmkUnS4+uKWQcnbw5phLlinRnqNJGuMR8WFKJaI9bOonjYcyqamlnOpIEFkuGuiokj+YytydQwa6EKwnxukqRqquiSNbte5KU6+oWeGWsDCKAaeSsXn2A6w1f3ZJqzaf4errGQy1Wu0704oKrafObMsjqsCwSWqssj5qxz1kyjjuqJSaO6W47cobZLjusjsvvbGum2+RBfHbr7/Z3htwYgPjGW3B3ZYLsMIGD0vuuw43PCm+E3t2EroTw6HxxWl0fPGstoEcssjw5VpyxSSnjDAVFrPshMsMwxyzyzQLavPNOKOsswQ89+zzsUDr6e3Qbs5sdBIvJ510AQAh+QQJFAADACwAAAAAgACAAAAC/pyPqcvtD6OctNqLs968+w+G4kiW5omm6sq27gvH8kzX9o3n+s73/g8MCofEovGITCqXzKbzGQoEoDWplCqzWrEvrZbL8nrBKbGYbDKb0SO1mg1yu+EduZyusdvxl/3AzycB+HcXCDFogGiYoJhYuMj4uNAYSIlgCYd5KcmnGTlXyRnh+UQ6KcqFKSDgYJqkusra4GoEGyvLQDtke4t7ClqKetDb2yp8xEu8agyslKy8PHss9AwdnTv9U219/ftWO83Nzfy9Gy4+Lt2sfW5tIE6+BrQdiwCvXs5D3/2ejr2OY58ve/68ycsh0MG9f/mqtIMGYaHBMzYSRpCoQBcJ/osSMH5q6ILjBI+bAK4QSYHkAY0eUI4smDFbmofKLKh0ZDIMTWIpYX48mGVnsYs+S4KM4VLBTUI5kQq9pbDoSpldntZbsJSlCpdZqQZtOkBiV7A3LI49qs8quoE/KZoDu7YmPqBv0faLy6/tGCQChzKkCw7u2nhunanNaxTwK7WE9wYTzHNu4cd2w/qd6BgLr8Zf0MDivCWT15hkoWjUurh04sl4TKFm4ul1E0yynVCqTRkwbtOSdlPx41uzHtWQpg7PXHzU8dDJKSyf0tzC8egY9FDPMPo6ZujasSPv3oc5+Dzcx5s/jz69+vXs27t/Dz++/Pn069u/jz+//v38BPvfKAAAIfkECRQAAwAsAAAAAIAAgAAAAv6cj6nL7Q+jnLTai7PevPsPhuJIluaJpurKtu4Lx/JM1/aN5/rO9/4PDAqHxKLxiEwql8ym8wmNSqfUqvWKzWq33K73Cw6Lx+Sy+YxOq9fstkEgcF/gcDmFTrdH8Hi9g8/ntwAIKIhASGg4gIgoyMio9/goJynZZrk4qYb5pnnGeQA6JhrqKUZ6aPqFmsCa5dqqqgWrQCtlWytbhTuoO8WbCznr+0dMBZwp3MWLHGXbfGsM/es7fSxrvWuabYXJfVVpXBaeaEfe53ceZ0iu2CnuBk9Z7h6LXh+8jt+73+//DzCgwIEECxo8iDChwoUMGzp8CDGixIkUK1q8iDGjxg2NHDt6/AgypMiRcgoAADs='


async def prep_pdf(pdf):
    # Convert "pdf" into chunks according to chunk size and overlap
    # Take "chunks" and vectorize it for SLLM lookup
    # return "knowledge_base" as the vectorized sets of chunks

    pdf_reader = PdfReader(pdf)

    # Collect text from pdf
    text = ''.join((page.extract_text() for page in pdf_reader.pages))

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
        collection_name='doc_chunks'
    )

    return knowledge_base


async def handle_user_q(kb, chain):
    # Get a "user_question" from Streamlit
    # Get the top K chunks relevant to the user's question
    # Return the chunks and user question

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

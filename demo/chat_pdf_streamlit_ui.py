'''
Advanced. Retrieval Augmented Generation (RAG) AKA "Chat my PDF" demo

Use a PDF document as a knowledge base to provide context for natural language Q&A

UI: Streamlit - streamlit.io
Vector store: Qdrant
PDF to text: PyPDF2
Text to vector (embedding) model: HuggingFace all-MiniLM-L12-v2

Single-PDF support, for now, to keep the demo code simple. Can easily extend to
e.g. work with multiple docs dropped in a directory

Prerequisites. From OgbujiPT cloned dir:.

```sh
pip install --upgrade https://github.com/uogbuji/OgbujiPT.git
pip install streamlit watchdog PyPDF2 PyCryptodome sentence_transformers qdrant-client tiktoken
```

You'll probably need a .env file. See demo/.env for an example to copy. Run the demo:

```sh
streamlit run demo/chat_pdf_streamlit_ui.py
```
'''
import os

from dotenv import load_dotenv

import streamlit as st
from PyPDF2 import PdfReader

from ogbujipt.config import openai_emulation, openai_live, HOST_DEFAULT
from ogbujipt.prompting import format, CHATGPT_DELIMITERS
from ogbujipt import oapi_first_choice_text
from ogbujipt.text_helper import text_splitter
from ogbujipt.embedding_helper import qdrant_collection

from sentence_transformers import SentenceTransformer

# Avoid re-entrace complaints from huggingface/tokenizers
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# Load the main parameters from .env file
load_dotenv()
# User can set a variety of likely values to trigger use of OpenAI full-service
OPENAI = os.getenv('OPENAI', 'False') in \
    ['True', 'true', 'TRUE', 'Yes', 'yes', 'YES']
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
LLM = os.getenv('LLM', 'LLM')  # TODO: get this from non-openai openai api hosts
LLM_HOST = os.getenv('LLM_HOST', 'my-llm-host')
LLM_PORT = os.getenv('LLM_PORT', '8000')
LLM_TEMP = float(os.getenv('LLM_TEMP', '1'))  # LLM temperature (randomness)
N_CTX = int(os.getenv('N_CTX', '2048'))  # LLM max context size
K = int(os.getenv('K', '3'))  # how many chunks to return for query context
EMBED_CHUNK_SIZE = int(os.getenv('EMBED_CHUNK_SIZE', '500'))  # Character count used in slicing up the document
EMBED_CHUNK_OVERLAP = int(os.getenv('EMBED_CHUNK_OVERLAP', '100'))  # Character count overlap between chunks
# LLM for vector DB embeddings; will be d/led: https://huggingface.co/sentence-transformers/all-MiniLM-L12-v2
DOC_EMBEDDINGS_LLM = os.getenv('DOC_EMBEDDINGS_LLM', 'all-MiniLM-L12-v2')

CONSOLE_WIDTH = 80

PDF_USER_QUERY_PROMPT = 'Ask a question about your PDF:'

favicon = 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAYAAABzenr0AAAAx3pUWHRSYXcgcHJvZmlsZSB0eXBlIGV4aWYAAHjabVBbDsMwCPvPKXaEBMiD46RrK+0GO/6cQKu2mqU4gJFDCNv3s4fXACUJkmsrWkoEREWpI2jR0CenKJMnyCXkt3o4BUKJcbOlrXj/UU+ngV0dUb4YtbcLy11Qcf/2MPKHeEw0pljdSN2IyYTkBt2+FYu2ev3CssU7mp0waH+TjlpeTHvmUrG9NeMdJto4cQQziw3A43DgjqCCEysawbMi4MzHJFjIvz0dCD94G1pUNtnUqQAAAYNpQ0NQSUNDIHByb2ZpbGUAAHicfZE9SMNAHMVfU6VFKg4tKOKQoXayi4o41ioUoUKoFVp1MLl+QpOGJMXFUXAtOPixWHVwcdbVwVUQBD9AXF2cFF2kxP8lhRYxHhz34929x907QGjVmGr2JQBVs4xMKinm8qti4BUBhDGMGIIyM/U5SUrDc3zdw8fXuzjP8j735xgsFE0G+ETiBNMNi3iDeGbT0jnvE0dYRS4QnxNPGHRB4keuKy6/cS47LPDMiJHNzBNHiMVyDys9zCqGSjxNHC2oGuULOZcLnLc4q7UG69yTvzBU1FaWuU5zDCksYgkSRChooIoaLMRp1UgxkaH9pId/1PFL5FLIVQUjxwLqUCE7fvA/+N2tWZqadJNCSaD/xbY/xoHALtBu2vb3sW23TwD/M3Cldf31FjD7SXqzq0WPgKFt4OK6qyl7wOUOMPKky4bsSH6aQqkEvJ/RN+WB8C0wsOb21tnH6QOQpa7SN8DBIRArU/a6x7uDvb39e6bT3w+I2XKvBcxDUgAADXppVFh0WE1MOmNvbS5hZG9iZS54bXAAAAAAADw/eHBhY2tldCBiZWdpbj0i77u/IiBpZD0iVzVNME1wQ2VoaUh6cmVTek5UY3prYzlkIj8+Cjx4OnhtcG1ldGEgeG1sbnM6eD0iYWRvYmU6bnM6bWV0YS8iIHg6eG1wdGs9IlhNUCBDb3JlIDQuNC4wLUV4aXYyIj4KIDxyZGY6UkRGIHhtbG5zOnJkZj0iaHR0cDovL3d3dy53My5vcmcvMTk5OS8wMi8yMi1yZGYtc3ludGF4LW5zIyI+CiAgPHJkZjpEZXNjcmlwdGlvbiByZGY6YWJvdXQ9IiIKICAgIHhtbG5zOnhtcE1NPSJodHRwOi8vbnMuYWRvYmUuY29tL3hhcC8xLjAvbW0vIgogICAgeG1sbnM6c3RFdnQ9Imh0dHA6Ly9ucy5hZG9iZS5jb20veGFwLzEuMC9zVHlwZS9SZXNvdXJjZUV2ZW50IyIKICAgIHhtbG5zOkdJTVA9Imh0dHA6Ly93d3cuZ2ltcC5vcmcveG1wLyIKICAgIHhtbG5zOmRjPSJodHRwOi8vcHVybC5vcmcvZGMvZWxlbWVudHMvMS4xLyIKICAgIHhtbG5zOnRpZmY9Imh0dHA6Ly9ucy5hZG9iZS5jb20vdGlmZi8xLjAvIgogICAgeG1sbnM6eG1wPSJodHRwOi8vbnMuYWRvYmUuY29tL3hhcC8xLjAvIgogICB4bXBNTTpEb2N1bWVudElEPSJnaW1wOmRvY2lkOmdpbXA6MmVhNWM3OTUtMGJmNi00ODc1LWE5YzgtZWY1M2FjYWUwZDBkIgogICB4bXBNTTpJbnN0YW5jZUlEPSJ4bXAuaWlkOmIxOTFhYWIxLWYxMzAtNDEwZC05M2U2LTgxNTI2MWNlMTA1NSIKICAgeG1wTU06T3JpZ2luYWxEb2N1bWVudElEPSJ4bXAuZGlkOjY0YjU3MjEwLTg4ZmUtNDRjZS1iODJlLTE2MTU3ODJjZWNmZSIKICAgR0lNUDpBUEk9IjIuMCIKICAgR0lNUDpQbGF0Zm9ybT0iTWFjIE9TIgogICBHSU1QOlRpbWVTdGFtcD0iMTY4OTgxNTA5NTk2Nzk5NyIKICAgR0lNUDpWZXJzaW9uPSIyLjEwLjM0IgogICBkYzpGb3JtYXQ9ImltYWdlL3BuZyIKICAgdGlmZjpPcmllbnRhdGlvbj0iMSIKICAgeG1wOkNyZWF0b3JUb29sPSJHSU1QIDIuMTAiCiAgIHhtcDpNZXRhZGF0YURhdGU9IjIwMjM6MDc6MTlUMTk6MDQ6NTQtMDY6MDAiCiAgIHhtcDpNb2RpZnlEYXRlPSIyMDIzOjA3OjE5VDE5OjA0OjU0LTA2OjAwIj4KICAgPHhtcE1NOkhpc3Rvcnk+CiAgICA8cmRmOlNlcT4KICAgICA8cmRmOmxpCiAgICAgIHN0RXZ0OmFjdGlvbj0ic2F2ZWQiCiAgICAgIHN0RXZ0OmNoYW5nZWQ9Ii8iCiAgICAgIHN0RXZ0Omluc3RhbmNlSUQ9InhtcC5paWQ6ZTY2MzFlOTctN2UyMC00ZWU2LTk0N2EtMDNhZDBkMGFhNmJiIgogICAgICBzdEV2dDpzb2Z0d2FyZUFnZW50PSJHaW1wIDIuMTAgKE1hYyBPUykiCiAgICAgIHN0RXZ0OndoZW49IjIwMjMtMDctMTlUMTk6MDQ6NTUtMDY6MDAiLz4KICAgIDwvcmRmOlNlcT4KICAgPC94bXBNTTpIaXN0b3J5PgogIDwvcmRmOkRlc2NyaXB0aW9uPgogPC9yZGY6UkRGPgo8L3g6eG1wbWV0YT4KICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgCiAgICAgICAgICAgICAgICAgICAgICAgICAgIAo8P3hwYWNrZXQgZW5kPSJ3Ij8+2ofEMAAAAAZiS0dEADUAOQA+ellPPQAAASxJREFUWMPNlzFuwkAQAGeD4gL5HuCKB9C7s9JCkYYmPzCNn0NDPgJt5M4SJQ8gjR+A5cIUTnOREFKiaDcJe+X59mZ0tnf3wDCqLC2rLC0te0wscGALPOchaZtuOPybwBW8By7ASishRvgiTu+BKbDetN3rn53ALXzTdnXTDe95SGrgRXMSYoGfj8sZQJjvTlWWFpqTECP8LS550kqIET6Ly05aCTHCPzcvtRJigYf5bg1wPi63WgmxwMdxnAKISK+VECN8H8MWWgkxwosYWmslxAivY3ihlRAj/DoVqyTEAheRHuCrZz+ReODO4/6vwMVH6OI3dJGIXKRiF8XIRTl20ZC4aMlcNKXfScR5FdzFxUQ0BeRGAi1cfTdsuuGQh6QFVsCjFm4ev3E9/wD9cjGz2siT7AAAAABJRU5ErkJggg=='  # noqa E501

st.set_page_config(  # Set up Streamlit page
    page_title='Ask your PDF',
    page_icon=favicon,
    layout='centered',
    initial_sidebar_state='expanded',
    )


@st.cache_data
def load_throbber():
    # oori_throbber.gif as a data URL
    # TODO: use a less archaic format than GIF; perhaps some sort of animatable vector image
    return 'data:image/gif;base64,R0lGODlhgACAAKEDAGwZDPPVSf///wAAACH/C05FVFNDQVBFMi4wAwEAAAAh+QQFFAADACwAAAAAgACAAAACm5yPqcvtD6OctNqLs968+w+G4kiW5omm6sq27gvH8kzX9o3n+s73/g8MCofEovGITCqXzKbzCY1Kp9Sq9YrNarfcrvcLDovH5LL5jE6r1+y2+w2Py+f0uv2Oz+v3/L7/DxgoOEhYaHiImKi4yNjo+AgZKTlJWWl5iZmpucnZ6fkJGio6SlpqeoqaqrrK2ur6ChsrO0tba3uLK1YAACH5BAUUAAMALAAAAACAAIAAAAL+nI+py+0Po5y02jsAwLz7P2kaSJZmJYrnyrJp2sYy9r7zjTd1nff5vvMJW0Dg8EgqFpFMmnLZjEaeT6mVUc0or1xD1rvtSr8HsnhoLofPwjTCzY7B3+u4bJ7A2z/6fH1f0qcgCBjyZwhV6EGIdahIwajj+AgR6WD5iHk5STnIyaEZF1r5STk6VVp4KrHa1MqayvaKaNQ5Cxl7dYuSG7Vr8bsSDNyLVnx3/JOMnKi7PDPs1NwVzTstVg17fZb90B34vBh+9O23DVgOdq46ThvUaV4r125VnX4zfI/zq6+8TvwPXrx34gIKHMijQz9y4RYi0eSQiaWIrnpR9BXr4pj+UhrrcerobBtILmZGUltjEluYlCcTsfQ47WVFkfQOLpgjsw3Hmja1GPQpr6e0oN542jqWk8iypCdGMU3S7mlBokMJCr1pVCpAqlOtXgXqFepPRfaMSjSLdawotI3UqnTbNeFRuHzYMuOKz64LvajoGvNrginInCZfskwpc2TSjk8vao2oVR3eu2HR8U07eU+6yJgrN7287zLnTYBJlebGdnTfzFs9X52lGtdpya6/0pZ7IXbuZ7qr4tbG2jZCG+5+Czcd/Dbx47KD9wb373ngSdI/X6surA72vVC2K6WSnHlr8CPEcye/wfx38Oopz26PPDx8he/nr4ZhPzT+/PoJ0/NX9l+AFhQAACH5BAUUAAMALAAAAACAAIAAAAL+nI+py+0Po5y02jsAwLz7P2kaSJZmJYrnyrJp2sYy9r7zjTd1nff5vvMJW0Dg8EgqFpFMmnLZjEaeT6mVUc0or1xD1rvtSr8HsnhoLofPwjTCzY7B3+u4bJ7A2z/6fH1f0qcgCBjyZwhV6IEXEOBAqIh1iNDY+DgZCcFYaamDmSmZuMDJeSkK6nmaQEpqaoSa+srAyuoaBDv4OUBLa8uDq6ZKydsbe4u7SVwcKguarLy8AHn1DB2dKxxXXbur7GtTuN16AP0Nsydeuupt/MuWXjnLznzcBd8pT2yuYq9b/vCPnrsx/uYBNIitGZN7jiYElKYLB0MLDxPWa1NQ34X+in6yzZjIgSMdj0Qy8vogMpjCOyavhUTYcaWLltxIpARDEgRIEze15Oyw80TPaYhkkoPJE6nKi3xojpMxNCKFoFCV4jSKwqm6HFF/PqCKoytWTVrjHRHLdEpZfGet+hwLkWRPrm4hgUWCdmA7cPlOcsnLV2BgBXPbahR8Lu7YwnjrTrr71/EpyF0AJ6YsxjI/zJUly+JsRfOIkYvdShG9waLedYcze057FbYBxkJQf13b8IptsnJN99ittnfrxsNjykbMr69LH8Cn4ibuF/noC6BXNKf+/Pfr1diFR59xHWj2qsVJH+9eunyJ8DrHC90+2ET1Cuzlu0cJPzFL78v1N+ZPfsN8DtQnx330/TedDwKy9p1q8fWw4GwIpraQgQNOaMV8BKJhIYP9xcZdE5htWCF/NXl4ooP6iQEWiQSZ+FQ36oH44BkgucgFQzj2A6M1uSl2nja4ffhWkHbouBWQIWaC5I8qAghMkUvKOKOUNSKz1j4JRmnelA0aN2WU22hJIZfSlUmYWWeayVtpZLIZHFx7rQjnnFDGaWSdNNL5pp5F5UmUn30E6qeVfC4VZqF2bonolYq2yRShjzaqn6STUrqZVJdCSoWjm/7ZKaOfOhGqqKOS2ump9lGh6gmstuqqV7BaIOesqJpqa3u45toer74qUAAAIfkECRQAAwAsAAAAAIAAgAAAAv6cj6nL7Q+jnLTaOwDAvPs/aRpIlmYliufKsmnaxjL2vvONN3Wd9/m+8wlbQODwSCoWkUyactmMRp5PqZVRzSivXEPWu+1KvweyeGguh8/CNMLNjsHf67hsnsDbP/p8fV/SpyAIGPJnCFXogRcQ4ECoiHWI0Nj4OBkJwVhpqYOZKZm4wMl5KQrqeZpASmpqhJr6ysDK6hoEO/g5QEtry4OrpkrJ2xt7i7tJXBwqC5qsvLwAefUMHZ0rHFddu6vsa1O43XoA/Q2zJ1666m38y5ZeOcvOfNwF3ylPbK5ir1v+8I+euzH+5gE0iK0Zk3uOJgSUpgsHQwsPE9ZrU1Dfhf6KfrLNmNhAgAAHHOl4JJKRlwORIkki7KjwTsprCViydKlRILiPM7kxsGkTp8p2O1GeLHkAKFChNE3GDNRz3E+lSxsgBXOSA8ipVKvmG6rzHNSjLxF07crUJ8SsFLYuOHs2rdS1Ty24VQAXrlx1Yflpjcr3bV69VssGqzsFcLyQhPPuXdx3hF3F+ATHTUr4a9PDFzVRbsgVbc3MowxjRWxxoIKrAxxbFq1ZbeqiRMWWzvma6krSq01ryXp39OXdw2+DpduZs+p1uPHyZly8d3OYyYObfU4ctvHNpy9axxw9guvYc2eL/W5gfAX10o+b54e+NXbx8w2w/hKfPf3ww/6mO/X+WXa6TaBff+5Rt9xvqLFWYG5KPVbZe5IhlyA5vjV4HX8W+qccbRJuUBiH6dUnn4b2+SZIfvNh2I2ICiYXGYjkBeZceCzeF9GHEILmoFclatcedy9W+ICKl92IYo61+bWdbMINNuCMkFHoIQoBQgdlUCEe+B+RbV0ZWpY77jMhH2D2aGKLXHZoGwhGIuniNIgseCGca3bn5SJn1hhlk+UhWKUJb2opZYSAtmkUnS4+uKWQcnbw5phLlinRnqNJGuMR8WFKJaI9bOonjYcyqamlnOpIEFkuGuiokj+YytydQwa6EKwnxukqRqquiSNbte5KU6+oWeGWsDCKAaeSsXn2A6w1f3ZJqzaf4errGQy1Wu0704oKrafObMsjqsCwSWqssj5qxz1kyjjuqJSaO6W47cobZLjusjsvvbGum2+RBfHbr7/Z3htwYgPjGW3B3ZYLsMIGD0vuuw43PCm+E3t2EroTw6HxxWl0fPGstoEcssjw5VpyxSSnjDAVFrPshMsMwxyzyzQLavPNOKOsswQ89+zzsUDr6e3Qbs5sdBIvJ510AQAh+QQJFAADACwAAAAAgACAAAAC/pyPqcvtD6OctNqLs968+w+G4kiW5omm6sq27gvH8kzX9o3n+s73/g8MCofEovGITCqXzKbzGQoEoDWplCqzWrEvrZbL8nrBKbGYbDKb0SO1mg1yu+EduZyusdvxl/3AzycB+HcXCDFogGiYoJhYuMj4uNAYSIlgCYd5KcmnGTlXyRnh+UQ6KcqFKSDgYJqkusra4GoEGyvLQDtke4t7ClqKetDb2yp8xEu8agyslKy8PHss9AwdnTv9U219/ftWO83Nzfy9Gy4+Lt2sfW5tIE6+BrQdiwCvXs5D3/2ejr2OY58ve/68ycsh0MG9f/mqtIMGYaHBMzYSRpCoQBcJ/osSMH5q6ILjBI+bAK4QSYHkAY0eUI4smDFbmofKLKh0ZDIMTWIpYX48mGVnsYs+S4KM4VLBTUI5kQq9pbDoSpldntZbsJSlCpdZqQZtOkBiV7A3LI49qs8quoE/KZoDu7YmPqBv0faLy6/tGCQChzKkCw7u2nhunanNaxTwK7WE9wYTzHNu4cd2w/qd6BgLr8Zf0MDivCWT15hkoWjUurh04sl4TKFm4ul1E0yynVCqTRkwbtOSdlPx41uzHtWQpg7PXHzU8dDJKSyf0tzC8egY9FDPMPo6ZujasSPv3oc5+Dzcx5s/jz69+vXs27t/Dz++/Pn069u/jz+//v38BPvfKAAAIfkECRQAAwAsAAAAAIAAgAAAAv6cj6nL7Q+jnLTai7PevPsPhuJIluaJpurKtu4Lx/JM1/aN5/rO9/4PDAqHxKLxiEwql8ym8wmNSqfUqvWKzWq33K73Cw6Lx+Sy+YxOq9fstkEgcF/gcDmFTrdH8Hi9g8/ntwAIKIhASGg4gIgoyMio9/goJynZZrk4qYb5pnnGeQA6JhrqKUZ6aPqFmsCa5dqqqgWrQCtlWytbhTuoO8WbCznr+0dMBZwp3MWLHGXbfGsM/es7fSxrvWuabYXJfVVpXBaeaEfe53ceZ0iu2CnuBk9Z7h6LXh+8jt+73+//DzCgwIEECxo8iDChwoUMGzp8CDGixIkUK1q8iDGjxg2NHDt6/AgypMiRcgoAADs='  # noqa E501


@st.cache_resource
def load_embedding_model(embedding_model_name):
    # LLM will be downloaded from HuggingFace automatically
    return SentenceTransformer(embedding_model_name)


def prep_pdf():
    '''
    Oori — Ask your PDF 📄💬
    '''
    'Convert pdf content into chunks according to chunk size & overlap'
    placeholder = st.empty()

    # Load PDF & collect its text & split it into chunks
    pdf = st.session_state['pdf']
    if not pdf:
        return

    with placeholder.container():
        # Get the embedding model
        if not st.session_state['embedding_model']:
            st.session_state['embedding_model'] = load_embedding_model(embedding_model_name=DOC_EMBEDDINGS_LLM)
        emb_model = st.session_state['embedding_model']

        # Vectorizes chunks for sLLM lookup
        # XXX: Look up rules around uploaded object names
        kb = qdrant_collection(pdf.name, emb_model)  # in-memory vector DB instance

        # Show throbber, embed the PDF, and get ready for similarity search
        embedding_placeholder = st.container()
        embedding_placeholder.write('Embedding PDF...')

        # Load throbber from cache
        throbber = load_throbber()
        embedding_placeholder.image(throbber)

        # Prepare a vector knowledgebase based on the pdf contents
        # Use st.session_state to avoid unnecessary reprocessing/reloading
        pdf_reader = PdfReader(pdf)
        text = ''.join((page.extract_text() for page in pdf_reader.pages))
        chunks = text_splitter(
            text, 
            chunk_size=EMBED_CHUNK_SIZE,
            chunk_overlap=EMBED_CHUNK_OVERLAP,
            separator='\n')

        # Update vector DB collection, insert the text chunks & update app state
        kb.update(texts=chunks)
        st.session_state['kb'] = kb  # Update state
    placeholder.empty()

    # Get the user query
    st.text_input(
        label=PDF_USER_QUERY_PROMPT,
        key='user_query_str',
        on_change=query_llm,
        args=(st.session_state['openai_api'], st.session_state['model']))



def query_llm(openai_api, model):
    '''
    Oori — Ask your PDF 📄💬
    '''
    kb = st.session_state['kb']
    user_query = st.session_state['user_query_str']

    # Placeholder for throbber & LLM response
    response_placeholder = st.empty()
    # Load throbber from cache
    throbber = load_throbber()
    response_placeholder.image(throbber)

    docs = kb.search(user_query, limit=K)

    # Concatenate text chunks for insertion into prompt
    gathered_chunks = '\n\n'.join(doc.payload['_text'] for doc in docs if doc.payload)

    # Build prompt the doc chunks as context
    prompt = format(
        f'Given the context, {user_query}\n\n'
        f'Context: """\n{gathered_chunks}\n"""\n',
        preamble='### SYSTEM:\nYou are a helpful assistant, who answers '
        'questions directly and as briefly as possible. '
        'If you cannot answer with the given context, just say so.',
        delimiters=CHATGPT_DELIMITERS)

    print('  PROMPT FOR LLM:  '.center(CONSOLE_WIDTH, '='), '\n', prompt)

    # Remember max Token length default is 16)
    response = openai_api.Completion.create(model=model, prompt=prompt, temperature=LLM_TEMP, max_tokens=1024)

    print('\nFull response data from LLM:\n', response)

    # Response is a json-like object; extract the text
    response_text = oapi_first_choice_text(response)
    print('\nResponse text from LLM:\n', response_text)

    response_placeholder.write(response_text)


def main():
    '''
    Oori — Ask your PDF 📄💬
    '''
    # Streamlit treats function docstrings as magic strings for user display. Use comments instead
    # Set up LLM host connection & launch the main loop
    # Use OpenAI API if specified, otherwise emulate with supplied host, etc. for self-hosted LLM
    if OPENAI:
        assert not (LLM_HOST or LLM_PORT), 'Don\'t use --host or --port with --openai'
        model = LLM
        openai_api = openai_live(
            model=LLM, debug=True)
    else:
        # For now the model param is most useful when OPENAI is True
        model = LLM or HOST_DEFAULT
        openai_api = openai_emulation(
            host=LLM_HOST, port=LLM_PORT, model=LLM, debug=True)

    st.session_state['embedding_model'] = None
    st.session_state['openai_api'] = openai_api
    st.session_state['model'] = model

    # Create file upload box on Streamlit, set from the user's upload
    # Use st.session_state to avoid unnessisary reprocessing/reloading
    st.file_uploader('Upload a PDF', type=['pdf'], accept_multiple_files=False,
                     on_change=prep_pdf, key='pdf')


main()  # Code to execute

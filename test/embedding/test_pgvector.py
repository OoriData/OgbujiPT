'''
pytest test

or

pytest test/embedding/test_pgvector.py

Uses fixtures from ../conftest.py

'''

# Remove, or narow down, once we no longer need to skip
# ruff: noqa

import pytest
from unittest.mock import MagicMock, patch

import os
from ogbujipt.embedding.pgvector import docDB
from sentence_transformers       import SentenceTransformer
import numpy as np

# FIXME: This stanza to go away once mocking is complete - Kai
HOST = os.environ.get('PGHOST', '0.0.0.0')
DB_NAME = os.environ.get('PGDATABASE', 'mock_db')
USER = os.environ.get('PGUSER', 'mock_user')
PASSWORD = os.environ.get('PGPASSWORD', 'mock_password')
PORT = os.environ.get('PGPORT', 5432)

pacer_copypasta = [  # Demo document
    'The FitnessGram™ Pacer Test is a multistage aerobic capacity test that progressively gets more difficult as it continues.', 
    'The 20 meter pacer test will begin in 30 seconds. Line up at the start.', 
    'The running speed starts slowly, but gets faster each minute after you hear this signal.', 
    '[beep] A single lap should be completed each time you hear this sound.', 
    '[ding] Remember to run in a straight line, and run as long as possible.', 
    'The second time you fail to complete a lap before the sound, your test is over.', 
    'The test will begin on the word start. On your mark, get ready, start.'
]

@patch('sentence_transformers.SentenceTransformer', spec=SentenceTransformer)
@pytest.mark.asyncio
async def test_PGv_embed_pacer(mock_SentenceTransformer):
    dummy_model = mock_SentenceTransformer()
    dummy_model.encode.return_value = np.array([1, 2, 3])
    print(f'EMODEL: {dummy_model}')
    TABLE_NAME = 'embedding_test'
    vDB = await docDB.from_conn_params(
        embedding_model=dummy_model,
        table_name=TABLE_NAME,
        db_name=DB_NAME,
        host=HOST,
        port=int(PORT),
        user=USER,
        password=PASSWORD)
    
    assert vDB is not None, ConnectionError("Postgres docker instance not available for testing PG code")
    # assert vDB is not None, pytest.skip("Postgres instance/docker not available for testing PG code", allow_module_level=True)
    
    # Create tables
    await vDB.drop_table()
    await vDB.create_doc_table()

    # Insert data
    for index, text in enumerate(pacer_copypasta):   # For each line in the copypasta
        await vDB.insert_doc(                        # Insert the line into the table
            content=text,                            # The text to be embedded
            permission='public',                     # Permission metadata for access control
            title=f'Pacer Copypasta line {index}',   # Title metadata
            page_numbers=[1, 2, 3],                  # Page number metadata
            tags=['fitness', 'pacer', 'copypasta'],  # Tag metadata
        )

    # search table with perfect match
    search_string = '[beep] A single lap should be completed each time you hear this sound.'
    sim_search = await vDB.search_doc_table(query_string=search_string, limit=3)
    assert sim_search is not None, Exception("No results returned from perfect search")

    await vDB.drop_table()

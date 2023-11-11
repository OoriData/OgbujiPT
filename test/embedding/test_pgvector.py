'''
pytest test

or

pytest test/embedding/test_pgvector.py

Uses fixtures from ../conftest.py

'''

import pytest
# from unittest.mock import MagicMock, patch
from unittest.mock import MagicMock

import os
from ogbujipt.embedding.pgvector import DocDB
import numpy as np

# XXX: This stanza to go away once mocking is complete - Kai
HOST = os.environ.get('PG_HOST', '0.0.0.0')
DB_NAME = os.environ.get('PG_DATABASE', 'mock_db')
USER = os.environ.get('PG_USER', 'mock_user')
PASSWORD = os.environ.get('PG_PASSWORD', 'mock_password')
PORT = os.environ.get('PG_PORT', 5432)

pacer_copypasta = [  # Demo document
    'The FitnessGram™ Pacer Test is a multistage aerobic capacity test that progressively gets more difficult as it \
        continues.', 
    'The 20 meter pacer test will begin in 30 seconds. Line up at the start.', 
    'The running speed starts slowly, but gets faster each minute after you hear this signal.', 
    '[beep] A single lap should be completed each time you hear this sound.', 
    '[ding] Remember to run in a straight line, and run as long as possible.', 
    'The second time you fail to complete a lap before the sound, your test is over.', 
    'The test will begin on the word start. On your mark, get ready, start.'
]

# XXX: This is to get around the fact that we can't mock the SentenceTransformer class without importing it - Kai
class SentenceTransformer(object):
    def __init__(self, model_name_or_path):
        self.encode = MagicMock()

@pytest.mark.asyncio
async def test_PGv_embed_pacer():
    dummy_model = SentenceTransformer('mock_transformer')
    dummy_model.encode.return_value = np.array([1, 2, 3])
    print(f'EMODEL: {dummy_model}')
    TABLE_NAME = 'embedding_test'
    try:
        vDB = await DocDB.from_conn_params(
            embedding_model=dummy_model,
            table_name=TABLE_NAME,
            db_name=DB_NAME,
            host=HOST,
            port=int(PORT),
            user=USER,
            password=PASSWORD)
    except ConnectionRefusedError:
        pytest.skip("No Postgres instance made available for test. Skipping.", allow_module_level=True)
    
    assert vDB is not None, ConnectionError("Postgres docker instance not available for testing PG code")
    
    # Create tables
    await vDB.drop_table()
    assert await vDB.table_exists() is False, Exception("Table exists before creation")
    await vDB.create_table()
    assert await vDB.table_exists() is True, Exception("Table does not exist after creation")

    # Insert data
    for index, text in enumerate(pacer_copypasta):   # For each line in the copypasta
        await vDB.insert(                            # Insert the line into the table
            content=text,                            # The text to be embedded
            title=f'Pacer Copypasta line {index}',   # Title metadata
            page_numbers=[1, 2, 3],                  # Page number metadata
            tags=['fitness', 'pacer', 'copypasta'],  # Tag metadata
        )

    assert await vDB.count_items() == len(pacer_copypasta), Exception("Not all documents inserted")

    # search table with perfect match
    search_string = '[beep] A single lap should be completed each time you hear this sound.'
    sim_search = await vDB.search(query_string=search_string, limit=3)
    assert sim_search is not None, Exception("No results returned from perfect search")

    await vDB.drop_table()

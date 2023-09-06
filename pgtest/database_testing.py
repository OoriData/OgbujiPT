import asyncio

from sentence_transformers import SentenceTransformer

from ogbujipt.embedding_helper import pgvector_connection

# Loading the embedding model
e_model = SentenceTransformer('all-mpnet-base-v2')

# Demo data
pacer_copypasta = [
    'The FitnessGram™ Pacer Test is a multistage aerobic capacity test that progressively gets more difficult as it continues.', 
    'The 20 meter pacer test will begin in 30 seconds. Line up at the start.', 
    'The running speed starts slowly, but gets faster each minute after you hear this signal.', 
    '[beep] A single lap should be completed each time you hear this sound.', 
    '[ding] Remember to run in a straight line, and run as long as possible.', 
    'The second time you fail to complete a lap before the sound, your test is over.', 
    'The test will begin on the word start. On your mark, get ready, start.'
]


async def main():
    # Connecting to the database
    vDB = await pgvector_connection.create(
        e_model, 
        'oori', 
        'example', 
        'PGv', 
        'sofola', 
        int('5432')
        )

    # Ensuring that the vector extension is installed
    await vDB.conn.execute('''CREATE EXTENSION IF NOT EXISTS vector;''')

    # Dropping the table if it exists
    await vDB.conn.execute('''DROP TABLE IF EXISTS embeddings;''')

    # Creating a new table
    await vDB.create_doc_table(table_name='embeddings')

    # Inserting data into the table
    for index, text in enumerate(pacer_copypasta):
        await vDB.insert_doc_table(
            table_name='embeddings',
            content=text,
            permission='public',
            title=f'Pacer Copypasta line {index}',
            page_numbers=[1, 2, 3],
            tags=['fitness', 'pacer', 'copypasta'],
            )

    # Setting K for the search
    k = 3

    # Searching the table with a perfect match
    print()
    search_string = '[beep] A single lap should be completed each time you hear this sound.'
    print('Semantic Searching data...')
    print('using search string:', search_string)
    sim_search = await vDB.search_doc_table(table_name='embeddings', query_string=search_string, k=k)
    print('RETURNED Title:', sim_search[0]['title'])
    print('RETURNED Content:', sim_search[0]['content'])
    print('RETURNED Cosine Similarity:', f'{sim_search[0]["cosine_similarity"]:.2f}')
    print('RAW RETURN:', sim_search)

    # Searching the table with a partial match
    print()
    search_string = 'Straight'
    print('Semantic Searching data...')
    print('using search string:', search_string)
    sim_search = await vDB.search_doc_table(table_name='embeddings', query_string=search_string, k=k)
    print('RETURNED Title:', sim_search[0]['title'])
    print('RETURNED Content:', sim_search[0]['content'])
    print('RETURNED Cosine Similarity:', f'{sim_search[0]["cosine_similarity"]:.2f}')
    print('RAW RETURN:', sim_search)


if __name__ == '__main__':
    asyncio.run(main())  # Running the main function asynchronously
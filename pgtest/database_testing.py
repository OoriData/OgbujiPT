from ogbujipt.embedding_helper import pgvector_connection
import asyncio
from sentence_transformers import SentenceTransformer

e_model = SentenceTransformer('all-mpnet-base-v2')

pacer_copypasta = [
    'The FitnessGram™ Pacer Test is a multistage aerobic capacity test that progressively gets more difficult as it continues.', 
    'The 20 meter pacer test will begin in 30 seconds. Line up at the start.', 
    'The running speed starts slowly, but gets faster each minute after you hear this signal.', 
    '[beep] A single lap should be completed each time you hear this sound.', 
    '[ding] Remember to run in a straight line, and run as long as possible.', 
    'The second time you fail to complete a lap before the sound, your test is over.', 
    'The test will begin on the word start. On your mark, get ready, start.'
]

e_pacer_copypasta = [e_model.encode(t) for t in pacer_copypasta]


async def main():
    print('Connecting to database...')
    vDB = await pgvector_connection.create(
        e_model, 
        'oori', 
        'example', 
        'PGv', 
        'sofola', 
        int('5432')
        )
    print('Connected to database')

    print('Ensuring that the vector extension is installed...')
    await vDB.execute('''CREATE EXTENSION IF NOT EXISTS vector;''')
    print('Ensured that the vector extension is installed')

    print('Dropping old table...')
    await vDB.execute('''DROP TABLE IF EXISTS embeddings;''')
    print('Dropped old table')

    print('Creating new table...')
    await vDB.execute(f'''\
        CREATE TABLE embeddings (
            id bigserial primary key, 
            embedding vector({len(e_pacer_copypasta[0])}), -- embedding vector field size
            content text NOT NULL, -- text content of the chunk
            permission text, -- permission of the chunk
            tokens integer, -- number of tokens in the chunk
            title text, -- title of file
            page_numbers integer[], -- page number of the document that the chunk is found in
            tags text[] -- tags associated with the chunk
        );''')
    print('Created new table')

    print('Inserting data...')
    for index, (embedding, text) in enumerate(zip(e_pacer_copypasta, pacer_copypasta)):
        await vDB.execute(f'''\
            INSERT INTO embeddings (
                embedding,
                content,
                title
            ) VALUES (
                '{list(embedding)}',
                '{text}',
                'Pacer Copypasta line {index}'
            );''')
    print('Inserted data')

    print('Querying data...')
    qanon = await vDB.fetch('''\
        SELECT 
            title, 
            content, 
            embedding 
        FROM 
            embeddings 
        WHERE 
            title = 'Pacer Copypasta line 4'
        ;''')
    print('Fetched Title:', qanon[0]['title'])
    print('Fetched Content:', qanon[0]['content'])

    # search_embedding = e_model.encode('[beep] A single lap should be completed each time you hear this sound.')
    search_embedding = e_model.encode('Straight')
    k = 3

    print('Semantic Searching data...')
    ss = await vDB.fetch(f'''\
        SELECT 
            1 - (embedding <=> '{list(search_embedding)}') AS cosine_similarity,
            title,
            content
        FROM 
            embeddings
        ORDER BY
            cosine_similarity DESC
        LIMIT {k}
        ;''')
    print('Returned Title:', ss[0]['title'])
    print('Returned Content:', ss[0]['content'])
    print('Returned Cosine Similarity:', f'{ss[0]["cosine_similarity"]:.2f}')


if __name__ == '__main__':
    asyncio.run(main())
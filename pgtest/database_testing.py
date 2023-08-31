from ogbujipt.embedding_helper import pgvector_connection
import asyncio
import os
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import numpy

load_dotenv()
print('USER:', os.getenv('DB_USER'))

e_model = SentenceTransformer("all-mpnet-base-v2")

lorem_ipsum = 'Lorem ipsum dolor sit amet, consectetur adipiscing elit. Fusce vestibulum nisl eget mauris malesuada, quis facilisis arcu vehicula. Sed consequat, quam ut auctor volutpat, augue ex tincidunt massa, in varius nulla ex vel ipsum. Nullam vitae eros nec ante sagittis luctus. Nullam scelerisque dolor eu orci iaculis, at convallis nulla luctus. Praesent eget ex id arcu facilisis varius vel id neque. Donec non orci eget elit aliquam tempus. Sed at tortor at tortor congue dictum. Nulla varius erat at libero lacinia, id dignissim risus auctor. Ut eu odio vehicula, tincidunt justo ac, viverra erat. Sed nec sem sit amet erat malesuada finibus. Nulla sit amet diam nec dolor tristique dignissim. Sed vehicula, justo nec posuere eleifend, libero ligula interdum neque, at lacinia arcu quam non est. Integer aliquet, erat id dictum euismod, felis libero blandit lorem, nec ullamcorper quam justo at elit.'

e_lorem_ipsum = e_model.encode(lorem_ipsum, show_progress_bar=True)
print(len(e_lorem_ipsum))

async def main():
    print('Connecting to database...')
    vDB = await pgvector_connection.create(
        e_model, 
        os.getenv('DB_USER'), 
        os.getenv('DB_PASSWORD'), 
        os.getenv('DB_NAME'), 
        os.getenv('DB_HOST'), 
        int(os.getenv('DB_PORT'))
        )
    print('Connected to database')

    print('Ensuring that the vector extension is installed...')
    await vDB.raw_sql(f'''CREATE EXTENSION IF NOT EXISTS vector;''')
    print('Ensured that the vector extension is installed')

    print('Dropping old table...')
    await vDB.raw_sql(f'''DROP TABLE IF EXISTS embeddings;''')
    print('Dropped old table')

    print('Creating new table...')
    await vDB.raw_sql(f'''\
        CREATE TABLE IF NOT EXISTS embeddings (
            id bigserial primary key, 
            embedding vector({len(e_lorem_ipsum)}), -- embedding vector field size
            content text NOT NULL, -- text content of the chunk
            permission text, -- permission of the chunk
            tokens integer, -- number of tokens in the chunk
            title text, -- title of file
            page_numbers integer[], -- page number of the document that the chunk is found in
            tags text[] -- tags associated with the chunk
            );''')
    print('Created new table')

    print('Inserting data...')
    await vDB.raw_sql(f'''INSERT INTO embeddings (embedding, content, title) VALUES ('{list(e_lorem_ipsum)}', '{lorem_ipsum}', 'Lorem Ipsum example text'); ''')
    print('Inserted data')

    print('Querying data...')
    qanon = await vDB.conn.fetch('''SELECT title, content FROM embeddings WHERE title = 'Lorem Ipsum example text'; ''')
    print(qanon)


if __name__ == '__main__':
    asyncio.run(main())
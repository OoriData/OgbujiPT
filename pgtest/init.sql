CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS CREATE TABLE embeddings (
    id bigserial primary key, 
    embedding vector(768) -- TODO: make me configurable!
    content text, -- text content of the chunk
    permission text, -- permission of the chunk
    tokens integer, -- number of tokens in the chunk
    title text, -- title of file
    page_numbers integer[], -- page number of the document that the chunk is found in
    tags text[], -- tags associated with the chunk
    );
# SPDX-FileCopyrightText: 2023-present Oori Data <info@oori.dev>
# SPDX-License-Identifier: Apache-2.0
# ogbujipt.embedding.pgvector_dupe
'''
Helpers to handle duplicates in PGVector DB tables

# Example usage
from ogbujipt.store.postgres.pgvector_dupe import create_jsonb_array_update_function, update_jsonb_array
async def example_usage():
    pool = await asyncpg.create_pool(user='postgres', password='password', database='dbname', host='localhost')
    
    try:
        # First, ensure the PostgreSQL function exists
        await create_jsonb_array_update_function(pool)
        
        # Example update
        rows_updated = await update_jsonb_array(pool=pool, table='my_table', json_column='json_column',
            array_key='my_array_key', new_item={'name': 'John', 'age': 30}, where_clause='id = $3',
            where_params=(123,)
        )
        
        print(f"Updated {rows_updated} rows")
    
    finally:
        await pool.close()

# Run the example (in an async context)
# await example_usage()

'''
from typing import Any
import asyncpg
import json


async def create_jsonb_array_update_function(pool: asyncpg.Pool) -> None:
    '''
    Creates the PostgreSQL function for JSONB array updates if it doesn't exist.
    
    Args:
        pool: AsyncPG connection pool
    '''
    create_function_sql = '''
    CREATE OR REPLACE FUNCTION add_to_jsonb_array(
        data jsonb,
        key text,
        new_item jsonb
    ) RETURNS jsonb AS $$
    BEGIN
        IF jsonb_typeof(data->key) = 'array' THEN
            RETURN jsonb_set(
                data,
                ARRAY[key],
                (data->key) || new_item
            );
        ELSE
            RETURN jsonb_set(
                data,
                ARRAY[key],
                jsonb_build_array(new_item),
                true
            );
        END IF;
    END;
    $$ LANGUAGE plpgsql;
    '''
    async with pool.acquire() as conn:
        await conn.execute(create_function_sql)


async def update_jsonb_array(
    pool: asyncpg.Pool,
    table: str,
    json_column: str,
    array_key: str,
    new_item: Any,
    where_clause: str,
    where_params: tuple | None = None
) -> int:
    '''
    Updates a JSONB array field in the specified table.
    
    Args:
        pool: AsyncPG connection pool
        table: Name of the table to update
        json_column: Name of the JSONB column
        array_key: Key in the JSONB object whose array we want to update
        new_item: New item to add to the array (will be JSON serialized)
        where_clause: SQL WHERE clause (e.g., "id = $3"). Make sure you use query params, for safety and correctness
            Query params should use $3, then $4, etc.
        where_params: Parameters for the WHERE clause
    
    Returns:
        Number of rows updated
    
    Example:
        await update_jsonb_array(pool, 'my_table', 'json_column', 'my_array_key',
            {'key': 'value'}, 'id = $3', (123,)
        )
    '''
    # Convert the new item to a JSON string and escape it properly
    json_item = json.dumps(new_item)
    
    update_sql = f'''
    UPDATE {table}
    SET {json_column} = add_to_jsonb_array(
        {json_column},
        $1,
        $2::jsonb
    )
    WHERE {where_clause}
    '''
    
    # Combine all parameters
    params = (array_key, json_item) + (where_params or ())
    
    async with pool.acquire() as conn:
        result = await conn.execute(
            update_sql,
            *params
        )
        return int(result.split()[-1])  # Extract number of rows updated


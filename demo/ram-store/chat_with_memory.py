#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025-present Oori Data <info@oori.dev>
# SPDX-License-Identifier: Apache-2.0
# demo/memory-store/chat_with_memory.py
'''
Chat with Memory Demo - Conversational AI with Vector Search

Demonstrates using in-memory message storage for chat applications with
semantic search over conversation history. No database required!

Prerequisites:
    Install: uv pip install -U . && uv pip install sentence-transformers

Usage:
    python chat_with_memory.py
'''
# See pyproject.toml for instruction to ignore `E501` (line too long)

import asyncio
from datetime import datetime, timezone
from uuid import uuid4
from sentence_transformers import SentenceTransformer

from ogbujipt.store import RAMMessageDB


async def simulate_conversation(message_db, conversation_id):
    '''Simulate a conversation with message storage'''

    print('\n💬 Simulating Conversation\n')
    print('=' * 80)

    # Sample conversation
    messages = [
        ('user', 'Hi! I want to learn about machine learning.'),
        ('assistant', 'Great! Machine learning is a field of AI that enables computers to learn from data. What aspect interests you most?'),
        ('user', 'How do neural networks work?'),
        ('assistant', 'Neural networks are inspired by the human brain. They consist of layers of interconnected nodes that process information.'),
        ('user', 'Can you recommend some Python libraries?'),
        ('assistant', 'Sure! For machine learning in Python, I recommend scikit-learn for classical ML, and PyTorch or TensorFlow for deep learning.'),
        ('user', 'What about preprocessing data?'),
        ('assistant', 'For data preprocessing, pandas is excellent for data manipulation, and numpy for numerical operations. scikit-learn also has preprocessing tools.'),
    ]

    # Insert messages
    for role, content in messages:
        timestamp = datetime.now(tz=timezone.utc)
        await message_db.insert(
            history_key=conversation_id,
            role=role,
            content=content,
            timestamp=timestamp,
            metadata={'session': 'demo'}
        )
        print(f'{role.upper()}: {content}')

    count = await message_db.count_items()
    print(f'\n✓ Stored {count} messages')
    print('=' * 80)


async def demonstrate_message_retrieval(message_db, conversation_id):
    '''Show different ways to retrieve messages'''

    print('\n📜 Message Retrieval Demo\n')
    print('=' * 80)

    # Get all messages
    print('\n1. All messages (chronological):')
    print('-' * 80)
    messages = await message_db.get_messages(history_key=conversation_id)
    for msg in messages:
        print(f'  [{msg.role}] {msg.content[:60]}...')

    # Get limited messages (most recent)
    print('\n2. Last 3 messages:')
    print('-' * 80)
    recent = await message_db.get_messages(history_key=conversation_id, limit=3)
    for msg in recent:
        print(f'  [{msg.role}] {msg.content[:60]}...')

    print('\n' + '=' * 80)


async def demonstrate_semantic_search(message_db, conversation_id):
    '''Demonstrate semantic search over conversation history'''

    print('\n🔍 Semantic Search Over Chat History\n')
    print('=' * 80)

    # Search queries
    queries = [
        'What did we discuss about Python?',
        'Tell me about neural networks',
        'Data preparation and cleaning'
    ]

    for query in queries:
        print(f'\nSearching: "{query}"')
        print('-' * 80)

        # Search conversation history
        results = await message_db.search(
            history_key=conversation_id,
            text=query,
            limit=2
        )

        for i, result in enumerate(results, 1):
            print(f'\n{i}. Similarity: {result.cosine_similarity:.3f} [{result.role}]')
            print(f'   {result.content}')

    print('\n' + '=' * 80)


async def demonstrate_windowed_chat(embedding_model):
    '''Demonstrate conversation windowing (keep last N messages)'''

    print('\n🪟 Windowed Chat Demo (Keep Last 4 Messages)\n')
    print('=' * 80)

    # Create windowed message DB (keeps only last 4 messages)
    windowed_db = RAMMessageDB(
        embedding_model=embedding_model,
        collection_name='windowed_chat',
        window=4  # Keep only last 4 messages
    )

    await windowed_db.setup()

    conversation_id = uuid4()

    # Insert 8 messages
    messages = [
        f'Message {i}: This is message number {i} in the conversation.'
        for i in range(1, 9)
    ]

    print('Inserting 8 messages (window size = 4)...\n')
    for i, content in enumerate(messages, 1):
        await windowed_db.insert(
            history_key=conversation_id,
            role='user',
            content=content,
            timestamp=datetime.now(tz=timezone.utc),
            metadata={}
        )
        count = await windowed_db.count_items()
        print(f'  After message {i}: {count} messages stored (window applied)')

    # Verify only last 4 remain
    print('\nFinal messages in database:')
    print('-' * 80)
    final_messages = await windowed_db.get_messages(history_key=conversation_id)
    for msg in final_messages:
        print(f'  {msg.content}')

    await windowed_db.cleanup()
    print('\n' + '=' * 80)


async def main():
    '''Main demo function'''
    print('\n' + '=' * 80)
    print('  Chat with Memory Demo')
    print('  In-Memory Message Storage & Semantic Search')
    print('=' * 80)

    # Load embedding model
    print('\n🤖 Loading embedding model…')
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    print('✓ Model loaded')

    # Create message database
    print('\n📚 Setting up message database…')
    message_db = RAMMessageDB(
        embedding_model=embedding_model,
        collection_name='chat_demo'
    )
    await message_db.setup()
    print('✓ Database ready')

    try:
        # Generate conversation ID
        conversation_id = uuid4()
        print(f'✓ Conversation ID: {conversation_id}')

        # Run demonstrations
        await simulate_conversation(message_db, conversation_id)
        await demonstrate_message_retrieval(message_db, conversation_id)
        await demonstrate_semantic_search(message_db, conversation_id)
        await demonstrate_windowed_chat(embedding_model)

        print('\n✨ Demo complete!')
        print('\nKey Features Demonstrated:')
        print('  • Message storage with embeddings')
        print('  • Chronological message retrieval')
        print('  • Semantic search over chat history')
        print('  • Conversation windowing (memory management)')
        print('\nUse Cases:')
        print('  • Chatbots with context awareness')
        print('  • Customer support ticket search')
        print('  • Personal note-taking with search')
        print('  • Interactive tutoring systems')

    finally:
        # Cleanup
        await message_db.cleanup()
        print('\n✓ Cleanup complete')


if __name__ == '__main__':
    asyncio.run(main())

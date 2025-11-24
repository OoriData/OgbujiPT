#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025-present Oori Data <info@oori.dev>
# SPDX-License-Identifier: Apache-2.0
# demo/pg-hybrid/chat_with_hybrid_kb.py
'''
Conversational AI with Hybrid Knowledge Base Search

Demonstrates a complete conversational AI system that combines:
    1. Chat history tracking (MessageDB)
    2. Knowledge base with hybrid search (DataDB + BM25 + RRF)
    3. Context-aware responses using both conversation and KB

This pattern is useful for:
    - RAG chatbots with conversation memory
    - Customer support systems with documentation
    - Personal assistants with knowledge bases

Prerequisites:
    1. PostgreSQL with pgvector running (see README.md)
    2. Install: uv pip install -U .
    3. Optional: OpenAI API key for LLM responses (or use local LLM)

Usage:
    python chat_with_hybrid_kb.py

Note: This demo focuses on retrieval patterns. For production, integrate
with your LLM of choice (OpenAI, Anthropic, local models, etc.)
'''

import asyncio
import os
import uuid
from datetime import datetime
# from typing import AsyncIterator
from sentence_transformers import SentenceTransformer

from ogbujipt.store.postgres import DataDB, MessageDB
from ogbujipt.retrieval import BM25Search, HybridSearch, SimpleDenseSearch
# from ogbujipt.memory.base import SearchResult


# Database connection parameters
PG_DB_NAME = os.environ.get('PG_DB_NAME', 'hybrid_demo')
PG_DB_HOST = os.environ.get('PG_DB_HOST', 'localhost')
PG_DB_PORT = int(os.environ.get('PG_DB_PORT', '5432'))
PG_DB_USER = os.environ.get('PG_DB_USER', 'demo_user')
PG_DB_PASSWORD = os.environ.get('PG_DB_PASSWORD', 'demo_pass_2025')

# Demo embedding model
DEMO_EMBEDDING_MODEL = 'all-MiniLM-L6-v2'

# Knowledge base: Sample tech documentation
TECH_KB = [
    {
        'content': 'PostgreSQL is a powerful, open-source object-relational database system with over 35 years of active development.',
        'metadata': {'category': 'database', 'topic': 'postgresql', 'difficulty': 'beginner'}
    },
    {
        'content': 'pgvector is a PostgreSQL extension for vector similarity search. It supports exact and approximate nearest neighbor search using HNSW and IVFFlat indexes.',
        'metadata': {'category': 'database', 'topic': 'pgvector', 'difficulty': 'intermediate'}
    },
    {
        'content': 'BM25 (Best Matching 25) is a ranking function used by search engines to estimate the relevance of documents to a given search query.',
        'metadata': {'category': 'search', 'topic': 'bm25', 'difficulty': 'intermediate'}
    },
    {
        'content': 'Reciprocal Rank Fusion (RRF) is a method for combining multiple ranked lists. It weights results by their rank rather than raw scores, making it robust across different scoring systems.',
        'metadata': {'category': 'search', 'topic': 'rrf', 'difficulty': 'advanced'}
    },
    {
        'content': 'Dense vector embeddings capture semantic meaning by representing text as continuous vectors in high-dimensional space. They excel at finding conceptually similar content.',
        'metadata': {'category': 'search', 'topic': 'embeddings', 'difficulty': 'intermediate'}
    },
    {
        'content': 'Sparse vectors store only non-zero elements, making them efficient for high-dimensional data like term frequency vectors in BM25.',
        'metadata': {'category': 'search', 'topic': 'sparse-vectors', 'difficulty': 'advanced'}
    },
    {
        'content': 'HNSW (Hierarchical Navigable Small World) is a graph-based algorithm for approximate nearest neighbor search. It provides excellent recall-speed tradeoffs.',
        'metadata': {'category': 'algorithms', 'topic': 'hnsw', 'difficulty': 'advanced'}
    },
    {
        'content': 'Semantic search uses machine learning models to understand the meaning and context of queries, going beyond keyword matching.',
        'metadata': {'category': 'search', 'topic': 'semantic-search', 'difficulty': 'beginner'}
    },
    {
        'content': 'Hybrid search combines multiple retrieval methods (e.g., dense + sparse) to leverage their complementary strengths and achieve better results.',
        'metadata': {'category': 'search', 'topic': 'hybrid-search', 'difficulty': 'intermediate'}
    },
    {
        'content': 'Python asyncio provides infrastructure for writing single-threaded concurrent code using coroutines, enabling efficient I/O-bound operations.',
        'metadata': {'category': 'programming', 'topic': 'asyncio', 'difficulty': 'intermediate'}
    },
]


class ConversationalRAG:
    '''
    Conversational RAG system with hybrid search.

    Combines:
    - MessageDB for chat history
    - DataDB with hybrid search for knowledge base
    - Context-aware retrieval
    '''

    def __init__(self, kb_db, msg_db, hybrid_search, history_key):
        self.kb_db = kb_db
        self.msg_db = msg_db
        self.hybrid_search = hybrid_search
        self.history_key = history_key

    async def store_message(self, role: str, content: str):
        '''Store a message in conversation history'''
        await self.msg_db.insert(
            history_key=self.history_key,
            role=role,
            content=content,
            metadata={'timestamp': datetime.now().isoformat()}
        )

    async def get_recent_history(self, limit: int = 5):
        '''Get recent conversation history'''
        messages = []
        for msg in await self.msg_db.get_messages(history_key=self.history_key, limit=limit):
            messages.append(msg)
        # Reverse to get chronological order
        return list(reversed(messages))

    async def search_kb(self, query: str, limit: int = 3):
        '''Search knowledge base using hybrid search'''
        results = []
        async for result in self.hybrid_search.execute(
            query=query,
            backends=[self.kb_db],
            limit=limit
        ):
            results.append(result)
        return results

    async def generate_response(self, user_query: str):
        '''
        Generate a response using both conversation history and KB search.

        In production, this would call your LLM with the context.
        For this demo, we just show what context would be provided.
        '''
        # Store user message
        await self.store_message('user', user_query)

        # Get recent conversation history
        history = await self.get_recent_history(limit=5)

        # Search knowledge base
        kb_results = await self.search_kb(user_query, limit=3)

        # Format context (what you'd send to LLM)
        context = {
            'conversation_history': [
                {'role': msg['role'], 'content': msg['content']}
                for msg in history[:-1]  # Exclude current message
            ],
            'knowledge_base_results': [
                {
                    'content': r.content,
                    'score': r.score,
                    'category': r.metadata.get('category', 'unknown'),
                    'source': r.source
                }
                for r in kb_results
            ],
            'current_query': user_query
        }

        return context

    def format_context_display(self, context):
        '''Format context for display (for demo purposes)'''
        output = []

        # Show conversation history
        if context['conversation_history']:
            output.append('📜 Recent conversation:')
            for msg in context['conversation_history'][-3:]:  # Last 3 messages
                role_icon = '👤' if msg['role'] == 'user' else '🤖'
                output.append(f'   {role_icon} {msg["role"]}: {msg["content"][:60]}…')

        # Show KB results
        output.append('\n📚 Knowledge base context:')
        for i, kb in enumerate(context['knowledge_base_results'], 1):
            output.append(f'   {i}. [{kb["score"]:.3f}] {kb["content"][:80]}…')
            output.append(f'      Category: {kb["category"]} | Source: {kb["source"]}')

        return '\n'.join(output)


async def setup_databases(embedding_model):
    '''Initialize both MessageDB and DataDB'''
    print('\n🧱 Setting up databases…')

    # Knowledge base DB
    kb_db = await DataDB.from_conn_params(
        embedding_model=embedding_model,
        table_name='tech_kb',
        db_name=PG_DB_NAME,
        host=PG_DB_HOST,
        port=PG_DB_PORT,
        user=PG_DB_USER,
        password=PG_DB_PASSWORD,
        itypes=['vector'],
        ifuncs=['cosine']
    )

    # Message/chat history DB
    msg_db = await MessageDB.from_conn_params(
        embedding_model=embedding_model,
        table_name='chat_history',
        db_name=PG_DB_NAME,
        host=PG_DB_HOST,
        port=PG_DB_PORT,
        user=PG_DB_USER,
        password=PG_DB_PASSWORD
    )

    # Drop and recreate tables for clean demo
    for db, name in [(kb_db, 'tech_kb'), (msg_db, 'chat_history')]:
        if await db.table_exists():
            await db.drop_table()
        await db.create_table()
        print(f'   ✓ Created table: {name}')

    # Populate knowledge base
    await kb_db.insert_many([
        (doc['content'], doc['metadata'])
        for doc in TECH_KB
    ])
    print(f'   ✓ Inserted {len(TECH_KB)} KB documents')

    return kb_db, msg_db


async def demo_conversation(rag_system):
    '''Run a sample conversation demonstrating the system'''
    print('\n' + '='*80)
    print('💬 DEMO CONVERSATION')
    print('='*80)

    # Simulated conversation
    conversation = [
        ('What is pgvector?', 'User asks about a specific technology'),
        ('How does it compare to regular search?', 'Follow-up question - needs conversation context'),
        ('Tell me about hybrid search approaches', 'Broader question'),
        ('What algorithms are used for fast search?', 'Technical deep dive'),
    ]

    for i, (query, explanation) in enumerate(conversation, 1):
        print(f'\n--- Turn {i}: {explanation} ---')
        print(f'👤 User: {query}')

        # Generate response with context
        context = await rag_system.generate_response(query)

        # Display what context would be sent to LLM
        print(f'\n🔍 Context gathered for LLM:')
        print(rag_system.format_context_display(context))

        # Simulate assistant response (in production, call LLM here)
        assistant_response = f'[Assistant would use the above context to generate a response about: {query}]'
        print(f'\n🤖 Assistant: {assistant_response}')

        # Store assistant response
        await rag_system.store_message('assistant', assistant_response)

        # Pause for readability
        await asyncio.sleep(0.5)


async def interactive_mode(rag_system):
    '''
    Interactive chat mode for user experimentation
    
    In production, this would call your LLM with the context.
    For this demo, we just show what context would be provided.

    Feel free to adapt this pattern to your own chat bot code.
    '''
    print('\n' + '='*80)
    print('💬 INTERACTIVE MODE')
    print('='*80)
    print('\nType your questions (or "quit" to exit)\n')

    while True:
        try:
            user_input = input('👤 You: ').strip()

            if not user_input:
                continue

            if user_input.lower() in ['quit', 'exit', 'q']:
                print('\n👋 Goodbye!')
                break

            # Generate context
            context = await rag_system.generate_response(user_input)

            # Display context
            print(f'\n🔍 Context:')
            print(rag_system.format_context_display(context))

            # In production, you'd call your LLM here with the context
            print(f'\n💡 In production: Send context to LLM for response generation')

            # Store placeholder response
            await rag_system.store_message(
                'assistant',
                '[Response generated by LLM would be stored here]'
            )

        except (EOFError, KeyboardInterrupt):
            # Handle Ctrl+D (EOFError) & Ctrl+C (KeyboardInterrupt)
            print('\n\n👋 Goodbye!')
            break
        except Exception as e:
            print(f'\n❌ Error: {e}')


async def main():
    '''Main demo flow'''
    print('='*80)
    print('Conversational AI with Hybrid Knowledge Base')
    print('='*80)
    print('\nThis demo shows how to combine:')
    print('  • Chat history tracking (MessageDB)')
    print('  • Knowledge base with hybrid search (DataDB + BM25)')
    print('  • Context-aware retrieval for RAG systems')

    # Load embedding model
    print('\n📦 Loading embedding model…')
    embedding_model = SentenceTransformer(DEMO_EMBEDDING_MODEL)
    print('   ✓ Model loaded!')

    # Setup databases
    kb_db, msg_db = await setup_databases(embedding_model)

    # Initialize hybrid search
    print('\n🔧 Initializing hybrid search…')
    hybrid_search = HybridSearch(
        strategies=[
            SimpleDenseSearch(),
            BM25Search(k1=1.5, b=0.75)
        ],
        k=60
    )
    print('   ✓ Hybrid search ready!')

    # Create RAG system with unique conversation ID
    history_key = uuid.uuid4()
    print(f'\n🆔 Conversation ID: {history_key}')

    rag_system = ConversationalRAG(
        kb_db=kb_db,
        msg_db=msg_db,
        hybrid_search=hybrid_search,
        history_key=history_key
    )

    # Run demo conversation
    await demo_conversation(rag_system)

    # Offer interactive mode
    print('\n' + '='*80)
    try:
        response = input('\nTry interactive mode? (y/n): ').strip().lower()
        if response == 'y':
            await interactive_mode(rag_system)
    except (EOFError, KeyboardInterrupt):
        print('\n\n👋 Goodbye!')

    # Cleanup
    print('\n🧹 Cleaning up…')
    try:
        await kb_db.drop_table()
        await msg_db.drop_table()
        print('   ✓ Dropped demo tables')
    except (asyncio.CancelledError, KeyboardInterrupt):
        # Handle case where cleanup was interrupted
        print('   ⚠ Cleanup interrupted')
    except Exception as e:
        # Handle any other errors during cleanup gracefully
        print(f'   ⚠ Cleanup error: {e}')

    # Summary
    print('\n' + '='*80)
    print('✅ Demo Complete!')
    print('='*80)
    print('\nKey Patterns Demonstrated:')
    print('  • Conversation history tracking with MessageDB')
    print('  • Hybrid search on knowledge base (dense + sparse)')
    print('  • Context assembly for LLM (history + KB results)')
    print('  • Follow-up question handling with conversation context')
    print('\nProduction Integration:')
    print('  • Replace placeholder responses with actual LLM calls')
    print('  • Add prompt engineering for better responses')
    print('  • Implement proper error handling and retries')
    print('  • Add metadata filtering for domain-specific search')
    print('  • Consider adding reranking for better relevance')
    print('\n📚 See README.md for more information')


if __name__ == '__main__':
    asyncio.run(main())

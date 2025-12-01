# SPDX-FileCopyrightText: 2025-present Oori Data <info@oori.dev>
# SPDX-License-Identifier: Apache-2.0
# ogbujipt.store.kgraph
'''
Knowledge graph storage backends using Onya.

Provides in-memory graph storage and retrieval for GraphRAG applications.
Can load .onya files from disk for static knowledge bases.
'''

from ogbujipt.store.kgraph.onya_store import OnyaKB

__all__ = ['OnyaKB']

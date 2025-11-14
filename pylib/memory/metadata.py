# SPDX-FileCopyrightText: 2025-present Oori Data <info@oori.dev>
# SPDX-License-Identifier: Apache-2.0
# ogbujipt.memory.metadata
'''
Metadata handling for knowledge bank items.

Includes schema definitions, filters, and RBAC helpers.
Philosophy: Keep it simple. Metadata is just dicts. Provide helpers, not frameworks.
'''

from typing import Any, Optional, Callable
from datetime import datetime
from dataclasses import dataclass, field, asdict


@dataclass
class ItemMetadata:
    '''
    Standard metadata structure for KB items.

    All fields are optional. Backends may add their own fields.
    This is a convenience class, not a requirement - plain dicts work fine.
    '''
    # Content metadata
    source: Optional[str] = None  # Where did this content come from?  (URL, file path, etc.)
    content_type: Optional[str] = None  # MIME type or semantic type ('chat_message', 'document', etc.)
    language: Optional[str] = None  # ISO 639-1 code (e.g., 'en', 'es')

    # Temporal metadata
    created_at: Optional[datetime] = None  # When was this item created?
    updated_at: Optional[datetime] = None  # When was it last updated?
    expires_at: Optional[datetime] = None  # When should it be pruned?

    # Provenance
    author: Optional[str] = None  # Who created this?
    version: Optional[str] = None  # Version or revision number

    # Organization
    tags: list[str] = field(default_factory=list)  # Free-form tags
    category: Optional[str] = None  # Single categorical label

    # RBAC (longer term - initial approach is separate KBs per role)
    access_level: Optional[str] = None  # 'public', 'private', 'team', etc.
    allowed_roles: list[str] = field(default_factory=list)  # Which roles can access this?

    # Custom fields
    custom: dict[str, Any] = field(default_factory=dict)  # Backend or app-specific metadata

    def to_dict(self) -> dict[str, Any]:
        '''Convert to plain dict, excluding None values and empty collections'''
        result = {}
        for key, value in asdict(self).items():
            if value is None:
                continue
            if isinstance(value, (list, dict)) and not value:
                continue
            # Convert datetime to ISO format
            if isinstance(value, datetime):
                result[key] = value.isoformat()
            else:
                result[key] = value
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'ItemMetadata':
        '''Create from plain dict, handling datetime strings'''
        # Parse datetime strings
        for key in ('created_at', 'updated_at', 'expires_at'):
            if key in data and isinstance(data[key], str):
                data[key] = datetime.fromisoformat(data[key])

        # Extract known fields, rest goes to custom
        known_fields = {f.name for f in cls.__dataclass_fields__.values()}
        custom = {k: v for k, v in data.items() if k not in known_fields}
        filtered = {k: v for k, v in data.items() if k in known_fields}

        if custom:
            filtered['custom'] = custom

        return cls(**filtered)


# Metadata filter builders
# These create functions that can be passed to backend search methods

def tag_filter(tags: list[str], match_all: bool = False) -> Callable[[dict], bool]:
    '''
    Create a filter function that matches items by tags.

    Args:
        tags: List of tags to match
        match_all: If True, item must have ALL tags. If False, any tag matches.

    Returns:
        Filter function suitable for in-memory filtering

    Example:
        >>> filter_func = tag_filter(['python', 'async'])
        >>> results = [r for r in all_results if filter_func(r.metadata)]
    '''
    def filter_func(metadata: dict) -> bool:
        item_tags = metadata.get('tags', [])
        if not item_tags:
            return False
        if match_all:
            return all(tag in item_tags for tag in tags)
        else:
            return any(tag in item_tags for tag in tags)
    return filter_func


def date_range_filter(
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
    field: str = 'created_at'
) -> Callable[[dict], bool]:
    '''
    Create a filter function that matches items by date range.

    Args:
        start: Start of date range (inclusive)
        end: End of date range (inclusive)
        field: Which datetime field to check ('created_at', 'updated_at', etc.)

    Returns:
        Filter function

    Example:
        >>> from datetime import datetime, timedelta
        >>> last_week = datetime.now() - timedelta(days=7)
        >>> filter_func = date_range_filter(start=last_week)
    '''
    def filter_func(metadata: dict) -> bool:
        field_value = metadata.get(field)
        if not field_value:
            return False
        if isinstance(field_value, str):
            field_value = datetime.fromisoformat(field_value)
        if start and field_value < start:
            return False
        if end and field_value > end:
            return False
        return True
    return filter_func


def role_filter(user_roles: list[str]) -> Callable[[dict], bool]:
    '''
    Create a filter function for RBAC filtering.

    Args:
        user_roles: List of roles the current user has

    Returns:
        Filter function that allows access if user has any allowed role

    Example:
        >>> filter_func = role_filter(['user', 'analyst'])
        >>> accessible = [r for r in results if filter_func(r.metadata)]
    '''
    def filter_func(metadata: dict) -> bool:
        allowed_roles = metadata.get('allowed_roles', [])
        if not allowed_roles:  # No restrictions
            return True
        return any(role in allowed_roles for role in user_roles)
    return filter_func


def combine_filters(*filters: Callable[[dict], bool], mode: str = 'and') -> Callable[[dict], bool]:
    '''
    Combine multiple filter functions with AND or OR logic.

    Args:
        *filters: Variable number of filter functions
        mode: 'and' (all must match) or 'or' (any must match)

    Returns:
        Combined filter function

    Example:
        >>> tag_f = tag_filter(['python'])
        >>> date_f = date_range_filter(start=last_week)
        >>> combined = combine_filters(tag_f, date_f, mode='and')
    '''
    def combined_filter(metadata: dict) -> bool:
        if mode == 'and':
            return all(f(metadata) for f in filters)
        elif mode == 'or':
            return any(f(metadata) for f in filters)
        else:
            raise ValueError(f'Invalid mode: {mode}. Use "and" or "or".')
    return combined_filter


__all__ = [
    'ItemMetadata',
    'tag_filter',
    'date_range_filter',
    'role_filter',
    'combine_filters',
]

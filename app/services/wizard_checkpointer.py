"""LangGraph checkpointer factories for the wizard graph.

Usage:
  - Tests:      get_memory_saver()    → in-memory, no persistence
  - Dev / prod: get_sqlite_saver()    → SQLite file, survives restarts
"""

from langgraph.checkpoint.memory import MemorySaver

_DEV_CHECKPOINTER_PATH = ".langgraph_checkpoints.sqlite"


def get_memory_saver() -> MemorySaver:
    """Return a fresh in-memory checkpointer (no file I/O, ideal for tests)."""
    return MemorySaver()


async def get_sqlite_saver():
    """Return an AsyncSqliteSaver backed by a local SQLite file.

    For dev / prod use where graph threads must survive process restarts.
    """
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

    return AsyncSqliteSaver.from_conn_string(_DEV_CHECKPOINTER_PATH)

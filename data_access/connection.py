"""
PostgreSQL Database Manager for AI Vigilance (Microservice Edition).

Drop-in replacement for SqliteManager — identical public API, but uses
psycopg2 against a PostgreSQL server so that multiple containers can
share the same database concurrently.

Key differences from SQLite version:
  - Connection pooling (psycopg2.pool.ThreadedConnectionPool)
  - %s parameter bindings instead of ?
  - SERIAL instead of AUTOINCREMENT
  - BYTEA instead of BLOB
  - No PRAGMA statements
  - ON CONFLICT uses Postgres syntax
"""

import os
import json
import logging
import time
import sqlite3
import numpy as np
from datetime import datetime, timedelta
import pytz

try:
    import psycopg2
    import psycopg2.extras
    import psycopg2.pool
except ImportError:
    psycopg2 = None

IST = pytz.timezone('Asia/Kolkata')
logger = logging.getLogger(__name__)


class SqliteCursorWrapper:
    def __init__(self, cursor):
        self._cursor = cursor

    def _translate(self, query, params=None):
        if params is None:
            params = ()
        if isinstance(params, (list, tuple)):
            params = tuple(self._unwrap_param(p) for p in params)
        return query.replace('%s', '?'), params

    def _unwrap_param(self, param):
        if isinstance(param, (bytes, bytearray)):
            return bytes(param)
        if isinstance(param, sqlite3.Binary):
            return param
        if hasattr(param, 'tobytes'):
            return param.tobytes()
        return param

    def execute(self, query, params=None):
        query, params = self._translate(query, params)
        return self._cursor.execute(query, params)

    def executemany(self, query, params_seq):
        query = query.replace('%s', '?')
        seq = [tuple(self._unwrap_param(p) for p in params) for params in params_seq]
        return self._cursor.executemany(query, seq)

    def fetchone(self):
        return self._cursor.fetchone()

    def fetchall(self):
        return self._cursor.fetchall()

    def __iter__(self):
        return iter(self._cursor)

    def __getattr__(self, item):
        return getattr(self._cursor, item)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._cursor.close()


class SqliteConnectionWrapper:
    def __init__(self, conn):
        self._conn = conn
        self._default_isolation_level = conn.isolation_level
        self._autocommit = False
        self._conn.row_factory = sqlite3.Row

    @property
    def autocommit(self):
        return self._autocommit

    @autocommit.setter
    def autocommit(self, value):
        self._autocommit = bool(value)
        self._conn.isolation_level = None if self._autocommit else self._default_isolation_level

    def cursor(self, cursor_factory=None):
        return SqliteCursorWrapper(self._conn.cursor())

    def commit(self):
        return self._conn.commit()

    def rollback(self):
        return self._conn.rollback()

    def close(self):
        return self._conn.close()

    def __getattr__(self, item):
        return getattr(self._conn, item)

# Default connection string — overridden by DATABASE_URL env var.
# Uses "postgres" hostname which matches the Docker Compose service name.
# For local (non-Docker) runs, set DATABASE_URL in your environment.
DEFAULT_DSN = "postgresql://aiv_user:aiv_password@postgres:5432/aiv_db"



class BaseConnection:
    def _get_connection(self): raise NotImplementedError
    def _put_connection(self, conn): raise NotImplementedError
    def _binary(self, value): raise NotImplementedError
    def _init_db(self): raise NotImplementedError

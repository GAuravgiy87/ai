import re

with open('database/sqlite_manager.py', 'r', encoding='utf-8') as f:
    code = f.read()

# Setup imports
code = code.replace('import sqlite3', 'import psycopg2\nimport psycopg2.extras\nfrom psycopg2.pool import SimpleConnectionPool')
code = code.replace('SqliteManager', 'PostgresManager')
code = code.replace('DatabaseManager = SqliteManager', '')

# Remove pragmas
code = re.sub(r'cursor\.execute\(\'PRAGMA.*?\'\)\n', '', code)

# Fix Schema
code = code.replace('AUTOINCREMENT', '')
code = code.replace('INTEGER PRIMARY KEY', 'SERIAL PRIMARY KEY')
code = code.replace('BLOB', 'BYTEA')
code = code.replace('DATETIME', 'TIMESTAMP')

# Fix INSERT OR REPLACE -> INSERT ... ON CONFLICT
code = code.replace('INSERT OR REPLACE INTO cameras', 'INSERT INTO cameras')

# The only INSERT OR REPLACE was on cameras. We need to append ON CONFLICT to the query
code = code.replace(
'''                    INSERT INTO cameras (camera_id, source, updated_at)
                    VALUES (?, ?, ?)''', 
'''                    INSERT INTO cameras (camera_id, source, updated_at)
                    VALUES (?, ?, ?)
                    ON CONFLICT(camera_id) DO UPDATE SET source = EXCLUDED.source, updated_at = EXCLUDED.updated_at'''
)

code = code.replace('sqlite3 Database Manager', 'PostgreSQL Database Manager')

# Let's completely replace the _get_connection method with a dynamic proxy
proxy = '''    def _get_connection(self):
        class CurProxy:
            def __init__(self, conn):
                self.conn = conn
                self.cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
            def execute(self, query, params=None):
                query = query.replace('?', '%s')
                self.cur.execute(query, params)
                return self.cur
            def fetchone(self):
                return self.cur.fetchone()
            def fetchall(self):
                return self.cur.fetchall()
            @property
            def lastrowid(self):
                self.cur.execute('SELECT LASTVAL()')
                return self.cur.fetchone()[0]

        class ConnProxy:
            def __init__(self, db_url):
                self.conn = psycopg2.connect(db_url)
                self.conn.autocommit = False
            def execute(self, query, params=None):
                cur = CurProxy(self.conn)
                return cur.execute(query, params)
            def cursor(self):
                return CurProxy(self.conn)
            def commit(self):
                self.conn.commit()
            def __enter__(self):
                return self
            def __exit__(self, exc_type, *args):
                if exc_type:
                    self.conn.rollback()
                else:
                    self.conn.commit()
                self.conn.close()
        
        return ConnProxy(self.db_path)'''

code = re.sub(r'    def _get_connection\(self\):.*?return conn', proxy, code, flags=re.DOTALL)

with open('database/postgres_manager.py', 'w', encoding='utf-8') as f:
    f.write(code)
print('Generated cleaner postgres_manager.py!')

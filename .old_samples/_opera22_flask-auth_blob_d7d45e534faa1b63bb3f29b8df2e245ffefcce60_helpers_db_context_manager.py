# Class is from user carusot42 on StackOverflow
# https://stackoverflow.com/questions/38076220/python-mysqldb-connection-in-a-class
import sqlite3

class DatabaseCM:
    def __init__(self, name):
        self._conn = sqlite3.connect(name)
        # Added this to parse all results as a list of dictionaries instead of tuples -opera22
        self._conn.row_factory = self.dict_factory
        # Added this to enforce FK constraints upon connection -opera22
        self._conn.execute('PRAGMA foreign_keys = ON')
        self._cursor = self._conn.cursor()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    @property
    def connection(self):
        return self._conn

    @property
    def cursor(self):
        return self._cursor

    def commit(self):
        self.connection.commit()

    def close(self, commit=True):
        if commit:
            self.commit()
        self.connection.close()

    def execute(self, sql, params=None):
        self.cursor.execute(sql, params or ())

    def fetchall(self):
        return self.cursor.fetchall()
    
    # From sqlite3 docs
    # https://docs.python.org/2/library/sqlite3.html#sqlite3.Connection.row_factory
    def dict_factory(self, cursor, row):
        d = {}
        for idx, col in enumerate(cursor.description):
            d[col[0]] = row[idx]
        return d
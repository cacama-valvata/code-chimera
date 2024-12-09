"""
Use pg8000 for the driver instead of psycopg2 because psycopg2 doesn't compile on lambda linux boxes:
 https://stackoverflow.com/questions/36607952/using-psycopg2-with-lambda-to-update-redshift-python/36608956#36608956,
 and I wasn't able to get the workaround library to work locally: https://github.com/jkehler/awslambda-psycopg2

Depends on SQLAlchemy logic. I wasn't able to abstract that away yet. Ideally this class would be an implmentation of some
abstract class.

"""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, scoped_session

from properties.src.env_properties import EnvProperties, Database
from storage.src.sql_alchemy_dtos import table_classes
from storage.src.sql_alchemy_dtos.base import Base
from utils.src.logging import print_if_debug_enabled


class SqlAlchemyEngine:
    def __init__(self, dialect='postgres', driver='pg8000', username='limiteduser', password=None, host='localhost',
                 port=5432, database='market_data', debug=False, **kwargs):
        # dialect+driver://username:password@host:port/database
        self.connection_string = '{dialect}+{driver}://{username}:{password}@{host}:{port}/{database}'.format(**{
            'dialect': dialect,
            'driver': driver,
            'username': username,
            'password': password,
            'host': host,
            'port': port,
            'database': database
        })
        self.debug = EnvProperties.debug
        print_if_debug_enabled(self.debug, 'connecting to database at {0} on port {1}'.format(host, port))
        self.sql_alchemy_engine = create_engine(self.connection_string, echo=debug, **kwargs)
        # sqlalchemy's API is confusingly named. Call session_maker_instance to create a Session instance. Then Session is called
        # to create thread-local session instances.
        # http://docs.sqlalchemy.org/en/latest/orm/session_api.html#sqlalchemy.orm.session.Session
        session_maker_instance = sessionmaker(bind=self.sql_alchemy_engine, autocommit=False, autoflush=False)
        self.scoped_session_maker = scoped_session(session_maker_instance)
        self.base = Base

    def dispose(self):
        """
        Closes all connections in the connection pool

        https://stackoverflow.com/questions/21738944/how-to-close-a-sqlalchemy-session
        :return:
        """
        self.sql_alchemy_engine.dispose()
        print_if_debug_enabled(self.debug, 'closed all SQLAlchemy connections')

    def drop_tables(self):
        print_if_debug_enabled(self.debug, 'dropping tables')
        self.base.metadata.drop_all(self.sql_alchemy_engine)

    def update_tables(self):
        print_if_debug_enabled(self.debug, 'updating tables')
        return self.base.metadata.create_all(self.sql_alchemy_engine)

    def initialize_tables(self):
        print_if_debug_enabled(self.debug, 'initializing tables')
        self.drop_tables()
        table_classes.exchange_data_tables()
        return self.base.metadata.create_all(self.sql_alchemy_engine)

    @classmethod
    def rds_engine(cls, **kwargs):
        return cls(dialect='postgres', driver='pg8000', username=Database.prod_db_username,
                   password=Database.prod_db_password, host=Database.rds_host, port=5432, database='market_data',
                   **kwargs)

    @classmethod
    def local_engine_maker(cls, **kwargs):
        return cls(dialect='postgres', driver='pg8000', username='limiteduser',
                   password=Database.local_db_password, host='localhost', port=5432,
                   database='market_data', **kwargs)

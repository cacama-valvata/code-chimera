"""
Use pg8000 for the driver instead of psycopg2 because psycopg2 doesn't compile on lambda linux boxes:
 https://stackoverflow.com/questions/36607952/using-psycopg2-with-lambda-to-update-redshift-python/36608956#36608956,
 and I wasn't able to get the workaround library to work locally: https://github.com/jkehler/awslambda-psycopg2

Depends on SQLAlchemy logic. I wasn't able to abstract that away yet. Ideally this class would be an implmentation of some
abstract class.

"""
import os
import warnings

from sqlalchemy import create_engine, event, exc
from sqlalchemy.orm import sessionmaker, scoped_session

from trading_platform.properties.env_properties import EnvProperties, DatabaseProperties
from trading_platform.storage.sql_alchemy_dtos import table_classes
from trading_platform.storage.sql_alchemy_dtos.base import Base
from trading_platform.utils.logging import print_if_debug_enabled


class SqlAlchemyEngine:
    def __init__(self, dialect='postgres', driver='pg8000', username='limiteduser', password=None, host='localhost',
                 port=5432, database='market_data', debug=False, **kwargs):
        """
        Create a SqlAlchemyEngine instance, using sqlalchemy's create_engine method. The following sqlalchemy
        options are hardcoded:
            pool_pre_ping: when checking out a connection from a pool, ping the database using the connection to make
                sure that the connection isn't stale.
                http://docs.sqlalchemy.org/en/latest/core/pooling.html#disconnect-handling-pessimistic
        Args:
            dialect:
            driver:
            username:
            password:
            host:
            port:
            database:
            debug:
            **kwargs:
        """
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
        self.sql_alchemy_engine = create_engine(self.connection_string, echo=self.debug, pool_pre_ping=True, **kwargs)
        # sqlalchemy's API is confusingly named. session_factory creates Session instances.
        # http://docs.sqlalchemy.org/en/latest/orm/session_api.html#sqlalchemy.orm.session.Session
        session_maker: sessionmaker = sessionmaker(bind=self.sql_alchemy_engine, autocommit=False, autoflush=False)
        # scoped_session_maker creates thread-local session instances.
        self.scoped_session_maker: scoped_session = scoped_session(session_maker)
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

    def add_engine_pidguard(self):
        """
        Add multiprocessing guards.

        Forces a connection to be reconnected if it is detected
        as having been shared to a sub-process.

        Taken from SQLAlchemy docs:
        http://docs.sqlalchemy.org/en/rel_1_0/faq/connections.html#how-do-i-use-engines-connections-sessions-with-python-multiprocessing-or-os-fork

        """

        @event.listens_for(self.sql_alchemy_engine, "connect")
        def connect(dbapi_connection, connection_record):
            connection_record.info['pid'] = os.getpid()

        @event.listens_for(self.sql_alchemy_engine, "checkout")
        def checkout(dbapi_connection, connection_record, connection_proxy):
            pid = os.getpid()
            if connection_record.info['pid'] != pid:
                # substitute log.debug() or similar here as desired
                warnings.warn(
                    "Parent process %(orig)s forked (%(newproc)s) with an open "
                    "database connection, "
                    "which is being discarded and recreated." %
                    {"newproc": pid, "orig": connection_record.info['pid']})
                connection_record.connection = connection_proxy.connection = None
                raise exc.DisconnectionError(
                    "Connection record belongs to pid %s, "
                    "attempting to check out in pid %s" %
                    (connection_record.info['pid'], pid)
                )

    @classmethod
    def rds_engine(cls, **kwargs):
        return cls(dialect='postgres', driver='pg8000', username=DatabaseProperties.prod_db_username,
                   password=DatabaseProperties.prod_db_password, host=DatabaseProperties.rds_host,
                   port=5432, database='market_data', debug=EnvProperties.debug,
                   **kwargs)

    @classmethod
    def local_engine_maker(cls, **kwargs):
        return cls(dialect='postgres', driver='pg8000', username='limiteduser',
                   password=DatabaseProperties.local_db_password, host='localhost', port=5432,
                   debug=EnvProperties.debug,
                   database='market_data', **kwargs)
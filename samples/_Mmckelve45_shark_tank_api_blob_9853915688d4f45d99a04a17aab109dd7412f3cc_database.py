from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
# from config import NEON_DATABASE_URL


# ----- SQLite ------ (Read and write from local file - please note, you will have to change some things)
# Load the data with SQLite localDB
# - Please note if you do this you need to change the Arrays in the models file to be a different representation
# SQLite does not support array types
# SQLALCHEMY_DATABASE_URL = 'sqlite:///./sharktankapp.db'
# engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={'check_same_thread': False})


# ----- Postgres ------ (Best Option for easy setup - Recommended)
# Load the data with postgres localDB - create a Server/DB in postgres and give it a password
SQLALCHEMY_DATABASE_URL = 'postgresql://postgres:Testme321!@localhost/SharkTankDB'
engine = create_engine(SQLALCHEMY_DATABASE_URL)

# ----- Neon Tech ----- (https://neon.tech/)
# Neon Tech is an open source, serverless postgres platform.  They have a free tier.
# I am not affiliated with them but its probably one of the only free tier postgres hosting sites out there.
# Here's an example of what the database url looks like:
# DATABASE_URL="postgresql://alex:AbC123dEf@ep-cool-darkness-123456.us-east-2.aws.neon.tech/dbname"
# please note that the docs will say postgres:// (make sure you say postgresql://)
# https://stackoverflow.com/questions/62688256/sqlalchemy-exc-nosuchmoduleerror-cant-load-plugin-sqlalchemy-dialectspostgre
# engine = create_engine(NEON_DATABASE_URL)

# ----- Host on GCP, Azure, or AWS.  They all have Postgres but there's upfront costs associated
# ----- ElephantSql is another option with a free tier (greater limitations).

# ----------------------
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

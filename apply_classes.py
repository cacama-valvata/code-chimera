#!/usr/bin/env python3

import ast
from pathlib import Path
import sqlite3
import sys

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("please provide the name of a classifications file")
        sys.exit(1)

    classifications_path = Path(sys.argv[1])

    if not classifications_path.is_file():
        print("provided classification file does not exist")
        sys.exit(1)
    
    with open(classifications_path, "r") as classes_file:
        classes = ast.literal_eval(classes_file.read())

    # make into dict for executemany
    classes = [{"rowid": rowid, "classification": class_} for rowid, class_ in classes.items()]

    db = sqlite3.connect("samples.db")

    try:
        db.executemany("UPDATE samples SET classification = :classification WHERE rowid = :rowid", classes)
        db.commit()
    except sqlite3.DatabaseError as e:
        print(f"sqlite error: {e}")
        sys.exit(1)
    finally:
        db.close()

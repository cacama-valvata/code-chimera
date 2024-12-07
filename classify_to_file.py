import sqlite3
import sys
import pathlib
import ast

VALID_ORDERS = {"ASC", "DESC"}

if len(sys.argv) < 2:
    order = "asc"
else:
    order = sys.argv[1]
    if order.upper() not in VALID_ORDERS:
        print("valid orders:", VALID_ORDERS)
        sys.exit(1)

if len(sys.argv) == 3:
    outfile = pathlib.Path(sys.argv[2] + "_classifications.txt")
else:
    outfile = pathlib.Path("classifications.txt")

classifications = {}

if outfile.exists():
    with open(outfile, "r") as f:
        classifications = ast.literal_eval(f.read())

db = sqlite3.connect("samples.db")

try:
    # oh nooo sql injection
    for rowid, file in db.execute(f"SELECT rowid, file FROM samples WHERE classification IS NULL ORDER BY rowid {order}"):
        if rowid not in classifications:
            print()
            print("rowid:", rowid)
            print(f"file: samples/{file}")
            classifications[rowid] = input("classification: ")
except KeyboardInterrupt:
    print()
    print("interrupted, saving progress to txt file")
finally:
    db.close()

    with open(outfile, "w") as f:
        f.write(repr(classifications))

import sqlite3
import sys

VALID_ORDERS = {"ASC", "DESC"}

if len(sys.argv) < 2:
    order = "asc"
else:
    order = sys.argv[1]
    if order.upper() not in VALID_ORDERS:
        print("valid orders:", VALID_ORDERS)
        sys.exit(1)
    
db = sqlite3.connect("samples.db")

try:
    # oh nooo sql injection
    for rowid, file, line_no in db.execute(f"SELECT rowid, file, link_line_number FROM samples WHERE answer_link and classification IS NULL ORDER BY rowid {order}"):
        print()
        print("rowid:", rowid)
        print(f"file: samples/{file}")
        print(f"line no: {line_no}")
        classification = input("classification: ")

        # no more sql injection
        db.execute("UPDATE samples SET classification = ? WHERE rowid = ?", (classification, rowid))
except KeyboardInterrupt:
    print()
    print("interrupted, saving progress")
finally:
    db.commit()
    db.close()

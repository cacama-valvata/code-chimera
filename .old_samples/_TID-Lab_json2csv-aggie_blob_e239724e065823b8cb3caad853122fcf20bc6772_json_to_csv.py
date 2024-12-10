import argparse
import csv
import json

# The JSON keys that you want to use
CSV_HEADER_ROW = [u'$oid', u'verified', u'status', u'$date', u'title', u'notes', u'locationName', u'assignedTo',
                  u'__v', u'updatedAt']

parser = argparse.ArgumentParser(description='JSON TO CSV for Aggie by Alex Stelea.')
parser.add_argument('-i', '--input', help='Input file name', required=True)
parser.add_argument('-o', '--output', help='Output file name', required=True)
args = parser.parse_args()

# INPUT and OUTPUT FILES
INPUT_FILE = args.input
OUTPUT_FILE = args.output


# Recursively find a key in a dictionary
# from http://stackoverflow.com/questions/14962485/finding-a-key-recursively-in-a-dictionary
def _finditem(obj, key):
    if key in obj: return obj[key]
    for k, v in obj.items():
        if isinstance(v,dict):
            item = _finditem(v, key)
            if item is not None:
                return item


def parse_json_stream(stream):
    decoder = json.JSONDecoder()
    obj, idx = decoder.raw_decode(stream)
    return obj


read_file = open(INPUT_FILE, 'r')
save_file = csv.writer(open(OUTPUT_FILE, "wb+"))

dict_str = read_file.read()
dict_list = [d.strip() for d in dict_str.splitlines()]

json_dicts = [parse_json_stream(i) for i in dict_list]

# You can use CSV_HEADER_ROW = json_dicts[0].keys here instead if expecting consistent JSON

# Write the header row
save_file.writerow(CSV_HEADER_ROW)

# Print every row in the file into csv format using the csv_header_row headers
for line in json_dicts:
    save_file.writerow([_finditem(line, row) for row in CSV_HEADER_ROW])

print "Converted " + str(len(json_dicts)) + " lines of JSON to " + OUTPUT_FILE

# Close the file for reading
read_file.close()


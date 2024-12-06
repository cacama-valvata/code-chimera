import requests
import pathlib
import sqlite3
import re

samples_dir = pathlib.Path("samples")
answers_dir = pathlib.Path("linked_answers")

answer_link_re = re.compile(r"stackoverflow\.com/a/(\d+)")

API_ENDPOINT = "https://api.stackexchange.com/2.3"

if __name__ == "__main__":
    answers_dir.mkdir(exist_ok=True)

    db = sqlite3.connect("samples.db")
    answer_ids = set()

    # get deduplicated ids from files
    for (filename,) in db.execute("SELECT file FROM samples WHERE answer_link"):
        with open(samples_dir / filename) as sample_file:
            sample = sample_file.read()

        answers_from_file = set(map(lambda match: match.group(1), answer_link_re.finditer(sample)))

        for answer_id in list(answers_from_file):
            if (answers_dir / f"{answer_id}.md").exists():
                answers_from_file.remove(answer_id)

        answer_ids |= answers_from_file

    # split ids into 100-id chunks, to batch into requests
    answer_id_list = list(answer_ids)
    answer_id_queries = []
    for offset in range(0, len(answer_id_list), 100):
        query_string = ";".join(answer_id_list[offset:offset+100])
        answer_id_queries.append(query_string)

    query_args = {
        # this filter only contains answer_id, body, and body_markdown
        "filter": "SI0J-PI2B.mFzmdq13Xemi",
        "site": "stackoverflow",
        "pagesize": 100
    }

    not_present = answer_ids
    for query in answer_id_queries:
        answers_resp = requests.get(API_ENDPOINT + "/answers/" + query, params=query_args)
        resp_json = answers_resp.json()
        present = {str(answer["answer_id"]) for answer in resp_json["items"]}
        not_present -= present

        for answer in resp_json["items"]:
            answer_id = answer["answer_id"]
            with open(answers_dir / (str(answer_id) + ".md"), "w") as markdown:
              markdown.write(answer["body_markdown"])

            with open(answers_dir / (str(answer_id) + ".html"), "w") as html:
              html.write(answer["body"])

    print("answer IDs not returned by stackoverflow:", not_present)

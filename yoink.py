import requests
import pathlib
import sqlite3
import re
import time
from stackapi import StackAPI

samples_dir = pathlib.Path("samples")
answers_dir = pathlib.Path("timely_answers")

answer_link_re = re.compile(r"stackoverflow\.com/a/(\d+)")

API_ENDPOINT = "https://api.stackexchange.com/2.3"
SITE = StackAPI('stackoverflow')

if __name__ == "__main__":
    answers_dir.mkdir(exist_ok=True)

    db = sqlite3.connect("samples.db")

    revisions = {}

    query_args = {
        # this filter only contains body and creation_date
        "filter": "dnzp4nYyxcbLfyMJb",
        "site": "stackoverflow",
        "pagesize": 100
    }

    not_exist = set()

    # get deduplicated ids from files
    for (rowid, filename, commit_timestamp) in db.execute("SELECT rowid, file, commit_timestamp FROM samples WHERE answer_link"):
        with open(samples_dir / filename) as sample_file:
            sample = sample_file.read()

        answers_from_file = set(map(lambda match: match.group(1), answer_link_re.finditer(sample)))

        for answer_id in list(answers_from_file):
            save_path = answers_dir / f"{answer_id}-{rowid}.html"

            # don't request an answer's revisions if we know it doesn't exist or if we've already fetched it
            if answer_id not in not_exist and not save_path.exists():
                # revision_resp = requests.get(API_ENDPOINT + "/posts/" + str(answer_id) + "/revisions", params=query_args)
                # resp_json = revision_resp.json()
                resp_json = SITE.fetch('posts/396109/revisions', filter='dnzp4nYyxcbLfyMJb')

                # whether post exists on SO (info was returned from server)
                exists = bool(resp_json.get("items"))

                if not exists:
                    not_exist.add(answer_id)
                else:
                    # only process if this answer has revisions (i.e., it exists)
                    # NOTE: in this branch exists has at least one item

                    relevant_revision = max((rev for rev in resp_json["items"] if rev["creation_date"] <= commit_timestamp), key=lambda rev: rev["creation_date"], default=None)

                    if relevant_revision is None:
                        print(f"no revisions that make sense timingwise for answer {answer_id} (this shouldn't happen???)")
                    else:
                        # save body to file
                        with open(save_path, "w") as out:
                            out.write(relevant_revision["body"])

    print("answer IDs not returned by stackoverflow:", not_exist)

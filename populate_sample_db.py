import dataclasses
import pathlib
import re
import sqlite3

@dataclasses.dataclass
class Sample:
    filename: str
    question_link: bool
    answer_link: bool
    link_line_number: int
    commit_timestamp: int
    

QUESTION_REGEX = re.compile(r"stackoverflow\.com/q")
ANSWER_REGEX = re.compile(r"stackoverflow\.com/a")

if __name__ == "__main__":
    db = sqlite3.connect("samples.db", autocommit=False)
    # db.execute("CREATE TABLE IF NOT EXISTS samples(file TEXT, commit_timestamp DATETIME, question_link BOOLEAN DEFAULT FALSE, answer_link BOOLEAN DEFAULT FALSE, classification TEXT)")

    with open("github.list.output.new", "r") as f:
        urls = f.read().splitlines()
    with open("github.lines.output.new", "r") as f:
        line_numbers = f.read().splitlines()
    with open("github.dates.output.new", "r") as f:
        commit_dates = f.read().splitlines()
    assert (len(urls) == len(line_numbers))
    assert (len(line_numbers) == len(commit_dates))

    sample_entries = []
    for i in range(len(urls)):
        file_path = urls[i][18:].replace("/", "_")
        line_no = int(line_numbers[i]) # 1-indexed
        commit_date = int(commit_dates[i])

        with open("samples/" + file_path, "r") as sample:
            contents = sample.read().splitlines()

        question_link = bool(QUESTION_REGEX.search(contents[line_no - 1]))
        answer_link = bool(ANSWER_REGEX.search(contents[line_no - 1]))

        sample_obj = Sample(file_path, question_link, answer_link, line_no, commit_date)
        sample_dict = dataclasses.asdict(sample_obj)
        sample_entries.append(sample_dict)

    db.executemany("INSERT INTO samples(file, question_link, answer_link, link_line_number, commit_timestamp) VALUES(:filename, :question_link, :answer_link, :link_line_number, :commit_timestamp)", sample_entries)
    db.commit()
    db.close()

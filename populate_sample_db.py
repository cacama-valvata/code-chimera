import dataclasses
import pathlib
import re
import sqlite3

@dataclasses.dataclass
class Sample:
    filename: str
    question_link: bool
    answer_link: bool
    

QUESTION_REGEX = re.compile(r"stackoverflow\.com/q")
ANSWER_REGEX = re.compile(r"stackoverflow\.com/a")

if __name__ == "__main__":
    db = sqlite3.connect("samples.db", autocommit=False)
    db.execute("CREATE TABLE IF NOT EXISTS samples(file TEXT, commit_timestamp DATETIME, question_link BOOLEAN DEFAULT FALSE, answer_link BOOLEAN DEFAULT FALSE)")

    sample_entries = []
    samples_dir = pathlib.Path("samples")
    for sample_file in samples_dir.iterdir():
        with open(sample_file, "r") as sample:
            contents = sample.read()

        question_link = bool(QUESTION_REGEX.search(contents))
        answer_link = bool(ANSWER_REGEX.search(contents))

        sample_obj = Sample(sample_file.name, question_link, answer_link)
        sample_dict = dataclasses.asdict(sample_obj)
        sample_entries.append(sample_dict)

    db.executemany("INSERT INTO samples(file, question_link, answer_link) VALUES(:filename, :question_link, :answer_link)", sample_entries)
    db.commit()
    db.close()

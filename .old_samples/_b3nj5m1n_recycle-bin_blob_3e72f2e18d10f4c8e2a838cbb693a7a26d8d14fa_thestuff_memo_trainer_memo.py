from abc import ABC, abstractmethod
from datetime import datetime
from difflib import SequenceMatcher
from math import ceil
from prettytable import PrettyTable
from sty import fg, bg, ef, rs
import argparse
import humanize
import os
import random
import shutil
import signal
import sqlite3
import string
import sys

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--mode", required=True, type=str)
parser.add_argument("--stats", action='store_true', required=False)
parser.add_argument("-i", "--length-min", required='--stats' not in sys.argv, type=int)
parser.add_argument("-x", "--length-max", required='--stats' not in sys.argv, type=int)
parser.add_argument("-d", "--database", required=False, type=str, default='./data.db')
args = parser.parse_args()

def sanatize_answer(string):
    table = string.maketrans('', '', ' \n\t\r')
    return string.strip().lower().translate(table)

def get_db_conn():
    return sqlite3.connect(args.database)

def init_db():
    conn = get_db_conn()
    conn.execute('''CREATE TABLE IF NOT EXISTS memos (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT,
    time_memo_started BIG INT,
    time_memo_finished BIG INT,
    time_recall_started BIG INT,
    time_recall_finished BIG INT,
    i INT,
    mode TEXT,
    question TEXT,
    answer TEXT,
    correct BOOLEAN NOT NULL CHECK (correct IN (0, 1))
    );''')
    conn.commit()
    conn.close()

class Mode(ABC):
    @abstractmethod
    def __init__(self, length_min, length_max, seed=random.random()):
        self.length_min = length_min
        self.length_max = length_max
        self.seed = seed
        self.i = 0
        self.rnd = random.Random(seed)
        self.current = self.rnd.random()
        self.time_memo_started = datetime.now()
        self.time_memo_finished = None
        self.time_recall_started = None
        self.time_recall_finished = None
        self.answer = None
    def new(self):
        self.i += 1
        self.current = self.rnd.random()
        self.time_memo_started = datetime.now()
        pass
    @abstractmethod
    def get_prompt(self):
        pass
    @abstractmethod
    def get_answer(self):
        pass
    @abstractmethod
    def check_answer(self, answer):
        self.time_recall_finished = datetime.now()
        self.answer = answer
        pass
    def save(self):
        conn = get_db_conn()
        query = f"""INSERT INTO memos (session_id,time_memo_started,time_memo_finished,time_recall_started,time_recall_finished,i,mode,question,answer,correct) VALUES ( ?, ?, ?, ?, ?, ?, ?, ?, ?, ? )"""
        args = (self.seed, int(self.time_memo_started.timestamp()*1000), int(self.time_memo_finished.timestamp()*1000), int(self.time_recall_started.timestamp()*1000), int(self.time_recall_finished.timestamp()*1000), self.i, type(self).__name__, self.get_prompt(), self.answer, self.check_answer(answer))
        conn.execute(query, args)
        conn.commit()
        conn.close()
    def finish_memo(self):
        self.time_memo_finished = datetime.now()
    def start_recall(self):
        self.time_recall_started = datetime.now()
    def finish_recall(self):
        self.time_recall_finished = datetime.now()
    def stats(self):
        pt = PrettyTable()
        pt.field_names = ["Time Memo", "Time Recall", "Time Total"]
        pt.add_row([
            humanize.precisedelta(self.time_memo_finished - self.time_memo_started, minimum_unit="seconds"),
            humanize.precisedelta(self.time_recall_finished - self.time_recall_started, minimum_unit="seconds"),
            humanize.precisedelta(self.time_recall_finished - self.time_memo_started, minimum_unit="seconds")
        ])
        return pt.get_string()

class ModeFromCharList(Mode):
    def __init__(self, char_list, *args):
        super().__init__(*args)
        self.char_list = char_list
    def get_prompt(self):
        result = []
        g = random.Random(self.current)
        for i in range(g.choice(range(self.length_min, self.length_max + 1))):
            result.append(g.choice(self.char_list))
        return ' '.join(result).upper()
    def get_answer(self):
        return self.get_prompt()
    def check_answer(self, answer):
        super().check_answer(answer)
        answer = sanatize_answer(answer)
        correct = sanatize_answer(self.get_answer())
        if not len(answer) == len(correct):
            return False
        for pair in zip(answer, correct):
            if not pair[0] == pair[1]:
                return False
        return True

class ModeLetters(ModeFromCharList):
    def __init__(self, *args):
        super().__init__(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'], *args)

class ModeNumbers(ModeFromCharList):
    def __init__(self, *args):
        super().__init__(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'], *args)


modes = {
    "letters": ModeLetters,
    "numbers": ModeNumbers,
}

generator = modes[args.mode](args.length_min, args.length_max)

class ErasablePrint():
    def print(self, text):
        self.text = text
        print(text)
    # Start and End are for escape codes
    def pretty_print(self, text, start, end):
        self.text = text
        print(start, end='')
        print(text, end='')
        print(end)
    def erase(self):
        columns, lines = shutil.get_terminal_size()
        length = len(self.text)
        for _ in range(ceil(length/columns) + self.text.count("\n")):
            print("\r\033[F", end='')
            print(" " * columns, end='')
            print("\r", end='')

# https://stackoverflow.com/a/1394994/11110290
def wait_for_any_key():
    try:
        # Win32
        from msvcrt import getch
    except ImportError:
        # UNIX
        def getch():
            import sys, tty, termios
            fd = sys.stdin.fileno()
            old = termios.tcgetattr(fd)
            try:
                tty.setraw(fd)
                result = sys.stdin.read(1)
                return result
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, old)
    getch()


# https://stackoverflow.com/a/47617607/11110290
def inline_diff(a, b):
    matcher = SequenceMatcher(None, a, b)
    def process_tag(tag, i1, i2, j1, j2):
        if tag == 'replace':
            return bg.cyan + '{' + matcher.a[i1:i2] + ' -> ' + matcher.b[j1:j2] + '}' + rs.all
        if tag == 'delete':
            return bg.da_red + '{- ' + matcher.a[i1:i2] + '}' + rs.all
        if tag == 'equal':
            return bg.green + matcher.a[i1:i2] + rs.all
        if tag == 'insert':
            return bg.red + '{+ ' + matcher.b[j1:j2] + '}' + rs.all
        assert False, "Unknown tag %r"%tag
    return ''.join(process_tag(*t) for t in matcher.get_opcodes())

def diff_answer(ground_truth, user_answer):
    result = ""
    ground_truth = sanatize_answer(ground_truth)
    user_answer = sanatize_answer(user_answer)
    diff = inline_diff(ground_truth, user_answer)
    result += f"""
    {diff}
    Correct Answer: {ground_truth}
    Your Answer:    {user_answer}
    """
    return result

prompts = {
    "start_memo": ("Press any key to start memo.", fg(255, 150, 50) + bg(32, 32, 32), rs.all),
    "stop_memo": ("Press any key when you're done memorizing.\n\n", fg(50, 155, 255) + bg(32, 32, 32) + ef.bold, rs.all),
    "start_recall": ("Press any key to start recall.\n", fg(50, 255, 150) + bg(32, 32, 32), rs.all),
    "enter_answer": (fg(150, 50, 255) + bg(32, 32, 32) + "Enter your answer:" + rs.all + "\n\n" + fg(99, 50, 255) + "> " + rs.all + ef.bold),
    "correct": (rs.all + fg(0, 255, 128) + "Correct" + rs.all),
    "incorrect": (rs.all + fg(255, 77, 77) + "Incorrect:\n" + rs.all),
    "seperator": (rs.all + "\n" + ef.underl + fg(22, 22, 22) + " " * shutil.get_terminal_size().columns + rs.all),
}

if args.stats == True:
    print("Showing stats.")
    conn = get_db_conn()
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    mode = type(modes[args.mode](0,0)).__name__
    records = cursor.execute("SELECT * FROM memos WHERE mode = ?", (mode,)).fetchall()
    for row in records:
        sim = SequenceMatcher(None, sanatize_answer(row["question"]), sanatize_answer(row["answer"])).ratio()
        print(sim)
    exit()

init_db()
ep = ErasablePrint()

while True:
    print("\n")
    ep.pretty_print(*prompts["start_memo"])
    wait_for_any_key()
    ep.erase()
    generator.new()
    ep.pretty_print(prompts["stop_memo"][0] + generator.get_prompt(), prompts["stop_memo"][1], prompts["stop_memo"][2])
    wait_for_any_key()
    ep.erase()
    generator.finish_memo()
    ep.pretty_print(*prompts["start_recall"])
    wait_for_any_key()
    ep.erase()
    generator.start_recall()
    answer = input(f"{prompts['enter_answer']}")
    generator.finish_recall()
    print("\n")
    if generator.check_answer(answer):
        print(prompts["correct"])
    else:
        print(f"{prompts['incorrect']} {diff_answer(generator.get_answer(), answer)}")
    generator.save()
    print(rs.all)
    print(generator.stats())
    print(prompts["seperator"])

from abc import ABC, abstractmethod
from datetime import datetime
from math import ceil
from prettytable import PrettyTable
import argparse
import humanize
import os
import random
import shutil
import string

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--mode", required=True, type=str)
parser.add_argument("-i", "--length-min", required=True, type=int)
parser.add_argument("-x", "--length-max", required=True, type=int)
args = parser.parse_args()

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
        pass
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
        super().check_answer(self)
        table = answer.maketrans('', '', ' \n\t\r')
        answer = answer.strip().lower().translate(table)
        correct = self.get_answer().strip().lower().translate(table)
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

print("Press any key when you're done memorizing.\n")

class ErasablePrint():
    def print(self, text):
        self.text = text
        print(text)
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
                return sys.stdin.read(1)
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, old)
    getch()


ep = ErasablePrint()
while True:
    print("\n" * 2)
    ep.print("Press any key to start memo.\n")
    wait_for_any_key()
    ep.erase()
    generator.new()
    ep.print(generator.get_prompt())
    wait_for_any_key()
    ep.erase()
    generator.finish_memo()
    ep.print("Press any key to start recall.\n")
    wait_for_any_key()
    ep.erase()
    generator.start_recall()
    prompt = "Enter your answer"
    space = ''.join([" " for _ in  range(len(generator.get_answer()) - len("Enter your answer:"))])
    answer = input(f"{prompt}:{space}\n> ")
    generator.finish_recall()
    if generator.check_answer(answer):
        print("Correct")
    else:
        print(f"The correct answer would have been: {generator.get_answer()}")
    print(generator.stats())

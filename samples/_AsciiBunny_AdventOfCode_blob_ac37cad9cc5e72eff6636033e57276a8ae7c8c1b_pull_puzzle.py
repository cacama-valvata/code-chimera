from urllib.request import urlopen, Request

import markdownify
from bs4 import BeautifulSoup
from pathlib import Path
from shutil import copy

# region markdownify setup

"""
https://stackoverflow.com/questions/45034227/html-to-markdown-with-html2text
https://beautiful-soup-4.readthedocs.io/en/latest/#multi-valued-attributes
https://beautiful-soup-4.readthedocs.io/en/latest/#contents-and-children
https://github.com/matthewwithanm/python-markdownify
"""


class CustomMarkdownConverter(markdownify.MarkdownConverter):
    def convert_a(self, el, text, convert_as_inline):
        classList = el.get("class")
        if classList and "searched_found" in classList:
            # custom transformation
            # unwrap child nodes of <a class="searched_found">
            text = ""
            for child in el.children:
                text += super().process_tag(child, convert_as_inline)
            return text
        # default transformation
        return super().convert_a(el, text, convert_as_inline)


# Create shorthand method for conversion
def md(soup, **options):
    return CustomMarkdownConverter(**options).convert_soup(soup)


# endregion


BASE_URL = "https://adventofcode.com/"
YEAR = 2023


def get_page(page):
    req = Request(BASE_URL + page, headers={'User-Agent': 'Mozilla/5.0', 'Cookie': 'session=' + session_cookie})
    html = urlopen(req)
    html = html.read().decode("utf-8")
    return html


def extract_instructions(page):
    html = get_page(page)
    soup = BeautifulSoup(html, "html.parser")

    instructions = soup.main.find_all(class_="day-desc")

    total_markdown = ""

    part_one = md(instructions[0])
    title = part_one.splitlines()[0][4:-4]
    total_markdown += "# " + title + "\n\n"
    total_markdown += f"[{BASE_URL + page}](BASE_URL + page)\n\n"
    total_markdown += "## Description\n\n"
    total_markdown += "### Part One\n\n"
    total_markdown += "\n".join(part_one.splitlines()[3:])

    print("Retrieved instructions...")

    if len(instructions) > 1:
        print("    also found part 2...")
        part_two = md(instructions[1])
        total_markdown += "### Part Two\n\n"
        total_markdown += "\n".join(part_two.splitlines()[3:])

    return total_markdown


def setup_day_directory(puzzle, markdown, input_file):
    path = Path(f"./{YEAR}/{puzzle:02}")
    path.mkdir(parents=True, exist_ok=True)

    with open(path / "markdown.md", "w") as f_md:
        f_md.write(markdown)

    with open(path / "input.txt", "w") as f_input:
        f_input.write(input_file)

    new_src = path / f"day{puzzle:02}.py"
    if not new_src.exists():
        copy("./template.py", new_src)


def main():
    puzzle = input("Which puzzle to import?\n")
    puzzle = int(puzzle)
    page = f"{YEAR}/day/{puzzle}"
    print(f"Importing puzzle: {page}")

    markdown = extract_instructions(page)
    input_file = get_page(page + "/input")
    print("Retrieved input...")

    setup_day_directory(puzzle, markdown, input_file)

if __name__ == '__main__':
    with open('session.txt') as f:
        session_cookie = f.read()
    main()

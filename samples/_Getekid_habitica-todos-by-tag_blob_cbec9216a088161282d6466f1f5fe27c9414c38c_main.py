from bottle import route, run, template
import requests
from os import getenv, path
from dotenv import load_dotenv

load_dotenv()

@route('/')
def todos():
    # Set the base variables
    baseUrl = "https://habitica.com/api/v3/"
    headers = {
        "Content-Type": "application/json",
        "x-api-user": getenv("API_USER_ID"),
        "x-api-key": getenv("API_TOKEN"),
    }
    tags = {}
    todos = {}

    # Get the tags
    tagsResp = requests.get(baseUrl + "tags", headers=headers)
    tagsResp = tagsResp.json()

    if tagsResp["success"]:
        # Build the Tags array and set the keys for the Todos array
        for tag in tagsResp["data"]:
            tags[tag["id"]] = tag["name"]
            todos[tag["id"]] = []

    # Get the Todos
    todosResp = requests.get(baseUrl + "tasks/user?type=todos", headers=headers)
    todosResp = todosResp.json()

    if todosResp["success"]:
        # Assign the Todos to the key with the respective Tag ID
        for todo in todosResp["data"]:
            for todoTag in todo["tags"]:
                todos[todoTag].append(todo["text"])

    return template('todo-list', tags=tags, todos=todos)

@route("/styles.css")
def get_styles():
    # If we have the minified CSS file, load that.
    if path.exists("views/todo-list.min.css"):
        with open("views/todo-list.min.css", "r") as css_min_file:
            return css_min_file.read()

    # If minified CSS file doesn't exist try and generate it.
    with open("views/todo-list.css", "r") as css_file:
        css = css_file.read()

    # Minify the CSS using the CSS Minifier Tool and store it.
    # Idea from https://stackoverflow.com/a/29980020.
    css_min_resp = requests.post("https://www.toptal.com/developers/cssminifier/api/raw", data={"input": css})
    if css_min_resp.status_code == 200:
        # Use a try block to catch any errors while writting the file (e.g. due to permissions).
        try:
            with open("views/todo-list.min.css", "w") as css_min_file:
                css_min_file.write(css_min_resp.text)
            css = css_min_resp.text
        finally:
            return css

    return css


run(host='localhost', port=8080)

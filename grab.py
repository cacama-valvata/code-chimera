import requests
import json
import urllib
import subprocess
from multiprocessing.dummy import Pool as ThreadPool

# https://docs.github.com/en/search-github/searching-on-github/searching-code
def grab_github ():
    github_pat = open('github.pat', 'r').read().strip()

    u = "https://api.github.com/search/code"
    #f = "filename:/sshd_config/"
    f = "stackoverflow.com in:file language:python"

    f_q = urllib.parse.quote(f, safe='')
    #print(f_q)
    url = u + "?q=" + f_q + "&per_page=100"

    page_num = 1

    # Loop 1
    g_resp = requests.get(url + "&page=" + str(page_num),
                          headers= {'Authorization': 'Bearer ' + github_pat,
                                    'Accept': 'application/vnd.github.text-match+json'} )
    #print(json.dumps(json.loads(g_resp.text), indent=2))
    #print(g_resp.headers)
    repos = json.loads(g_resp.text)["items"]
    #print(json.loads(g_resp.text)["total_count")
    with open("github.list.output", "a") as f:
        for repo in repos:
            print(str(page_num) + ": " + repo["html_url"])
            f.write(repo["html_url"] + "\n")
    
    page_num += 1

    # Loop until got all of them
    while 'Link' in g_resp.headers and page_num <= 10:
        #print(url + "&page=" + str(page_num))
        g_resp = requests.get(url + "&page=" + str(page_num),
                              headers= {'Authorization': 'Bearer ' + github_pat,
                                        'Accept': 'application/vnd.github.text-match+json'} )
        #print(g_resp.headers)
        text = json.loads(g_resp.text)
        if "items" not in text:
            print("items not in response, pagenum: " + str(page_num))
            break
        repos = text["items"]
        with open("github.list.output", "a") as f:
            for repo in repos:
                print(str(page_num) + ": " + repo["html_url"])
                f.write(repo["html_url"] + "\n")

        page_num += 1

    #print(repos["total_count"])
    #print(json.dumps(repos, indent=4))


def dl_gh():
    def repo_process(repo_url):
        resdir = "results"
        resfn = repo_url[18:].replace("/", "_")
        
        response = requests.get(repo_url.replace("blob", "raw"))
        if response.status_code != 200:
            return

        with open(resdir + "/" + resfn, "w") as f:
            f.write(response.text)

        return

    repos = open("github.list.output", "r").read().splitlines()
    pool = ThreadPool(50)
    pool.map(repo_process, repos)
    print("done :3")


#grab_github()
dl_gh()

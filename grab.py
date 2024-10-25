import requests
import json
import urllib

# https://docs.github.com/en/search-github/searching-on-github/searching-code
def grab_github ():
    github_pat = open('github.pat', 'r').read().strip()

    u = "https://api.github.com/search/code"
    f = "filename:/sshd_config/"

    print(urllib.parse.quote(f, safe=''))

    url = u + "?q=" + f + "&per_page=100"

    g_resp = requests.get(url,
                          headers= {'Authorization': 'Bearer ' + github_pat,
                                    'Accept': 'application/vnd.github+json'} )
    print(g_resp)
    repos = json.loads(g_resp.text)
    print(repos["total_count"])
    #print(json.dumps(repos, indent=4))

grab_github()

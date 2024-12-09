import requests
import json
import urllib
import subprocess
from multiprocessing.dummy import Pool as ThreadPool
from multiprocessing import Process, Queue

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


def redo_links():
    def repo_process(repo_url):
        if repo_url in last:
            #print("duplicate: " + repo_url)
            return

        cmd = ["bash", "process.sh", repo_url]
        ret_cmd = subprocess.run(cmd, capture_output=True)
        if ret_cmd.returncode != 0:
            print(f"1. Errored at {repo_url} with code {ret_cmd.returncode}\nOutput:")
            print(f"{ret_cmd.stderr.decode("utf-8")}")
            teardown_repo(repo_url)
            return
        
        output = ret_cmd.stdout.decode("utf-8").strip().splitlines()
        for i in range(len(output)):
            output[i] = output[i].strip()
        output = [x for x in output if x] # remove empty strings
        if len(output)//3 != len(output)/3:
            print(f"2. Errored at {repo_url}\n{len(output)//3}  {len(output)/3}\nOutput:")
            print(f"{ret_cmd.stdout.decode("utf-8")}")
            teardown_repo(repo_url)
            return
        num_urls = len(output)//3

        to_return = []
        for i in range(num_urls):
            new_url = output[i + 0*num_urls]
            new_date = output[i + 1*num_urls]
            new_lineno = output[i + 2*num_urls]
            to_return.append( (repo_url, new_url, new_date, new_lineno) )

        print(f"Success: {repo_url.split('/')[4]}")
        writing_q.put(to_return)
        teardown_repo(repo_url)
        return

    def teardown_repo(repo_url):
        # ['https:', '', 'github.com', 'kristovatlas', 'multi-sig-check-demo', 'blob', '78231bb8cff94323d0fff827f745db1905146028', 'examples', 'osx-config-check-master', 'app.py']
        repo_dir = repo_url.split('/')[4]

        td_cmd = ["rm", "-rf", "/tmp/workdir/" + repo_dir]
        ret_td = subprocess.run(td_cmd, capture_output=True)
        return

    def record_results (queue):
        while True:
            if not queue.empty():
                result = queue.get()
                if result == "DONE":
                    break
                for old_url, url, date, line_number in result:
                    with open("github.cache", "a") as f:
                        f.write(old_url + "\n")
                    with open("github.list.output.new", "a") as f:
                        f.write(url + "\n")
                    with open("github.dates.output.new", "a") as f:
                        f.write(date + "\n")
                    with open("github.lines.output.new", "a") as f:
                        f.write(line_number + "\n")
        return


    # for resuming
    try:
        last_repos = open('github.cache', 'r').read().splitlines()
        last = set(last_repos)
    except FileNotFoundError:
        last = set()

    # print(len(last))
    # exit()

    writing_q = Queue()
    record_results_p = Process(target=record_results, args=(writing_q,))
    record_results_p.start()

    repos = open("github.list.output", "r").read().splitlines()
    pool = ThreadPool(1)
    output = pool.map(repo_process, repos)

    writing_q.put("DONE")
    record_results_p.join()

    print("done :3")

def dl_gh():
    def repo_process(repo_url):
        resdir = "samples"
        resfn = repo_url[18:].replace("/", "_")
        
        response = requests.get(repo_url.replace("blob", "raw"))
        if response.status_code != 200:
            return

        with open(resdir + "/" + resfn, "w") as f:
            f.write(response.text)

        return

    repos = open("github.list.output", "r").read().splitlines()
    pool = ThreadPool(1)
    pool.map(repo_process, repos)
    print("done :3")


#grab_github()
#dl_gh()
redo_links()

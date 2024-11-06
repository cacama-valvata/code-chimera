import os
from datetime import timezone
from operator import itemgetter

import requests
from dateutil import parser
from github import Github
from packaging.version import parse


def is_final_version(version: str) -> bool:
    parsed_version = parse(version)

    return not parsed_version.is_prerelease and not parsed_version.is_postrelease


REF_REPO = "altair-viz/altair"

# https://github.com/altair-viz/altair/releases/tag/v2.4.1
# https://raw.githubusercontent.com/altair-viz/altair/v2.4.1/doc/requirements.txt
if __name__ == "__main__":
    # https://pygithub.readthedocs.io/en/latest/apis.html
    # https://pygithub.readthedocs.io/en/latest/github_objects/Repository.html#github.Repository.Repository.get_release
    # https://docs.github.com/en/rest/releases/releases?apiVersion=2022-11-28#get-a-release-by-tag-name
    # https://docs.github.com/en/rest/git/tags?apiVersion=2022-11-28#get-a-tag
    # https://pygithub.readthedocs.io/en/latest/github_objects/Repository.html#github.Repository.Repository.get_git_tag
    # https://stackoverflow.com/a/29199717
    g = Github(os.environ["GITHUB_TOKEN"])
    # print(g)

    repo = g.get_repo(REF_REPO)

    # https://stackoverflow.com/a/2364277
    # https://docs.python.org/3/library/functions.html#next
    tag = next(tag for tag in repo.get_tags() if tag.name == "v2.4.1")

    # https://pygithub.readthedocs.io/en/latest/github_objects/Commit.html
    # https://pygithub.readthedocs.io/en/latest/github_objects/GitCommit.html#github.GitCommit.GitCommit
    # https://pygithub.readthedocs.io/en/latest/github_objects/GitAuthor.html#github.GitAuthor.GitAuthor
    # https://docs.github.com/en/rest/git/commits?apiVersion=2022-11-28#get-a-commit
    # https://github.com/PyGithub/PyGithub/issues/1905
    # https://github.com/PyGithub/PyGithub/issues/512
    tag_date = tag.commit.commit.author.date
    tag_date_tz = tag_date.replace(tzinfo=timezone.utc)
    # print(tag_date, repr(tag_date), tag_date.tzinfo)
    # print(tag_date_tz, repr(tag_date_tz), tag_date_tz.tzinfo)
    # print(tag.commit.commit.committer.date)

    # https://requests.readthedocs.io/en/latest/user/advanced/#session-objects
    # https://requests.readthedocs.io/en/latest/api/#request-sessions
    with requests.Session() as s:
        res = s.get(
            "https://raw.githubusercontent.com/altair-viz/altair/v2.4.1/doc/requirements.txt"
        )

        pkgs = res.text.split()
        # pkgs = ["sphinx"]
        # pkgs = ["sphinx_rtd_theme"]

    with requests.Session() as s:
        deps = []
        for pkg in pkgs:
            res = s.get(f"https://pypi.org/pypi/{pkg}/json")
            pkg_data = res.json()
            # print(pkg_data)

            final_versions = [
                version for version in pkg_data["releases"] if is_final_version(version)
            ]
            # print(final_versions)

            pairs = []
            for version in final_versions:
                files = pkg_data["releases"][version]
                # print(files)

                # No files: https://pypi.org/project/sphinx-rtd-theme/0.1.0/
                # https://pypi.org/project/sphinx-rtd-theme/0.1.1/
                if files:
                    dts = [
                        parser.isoparse(file["upload_time_iso_8601"]) for file in files
                    ]
                    min_dt = min(dts)

                    pairs.append((version, min_dt))
                    # print(min_dt, repr(min_dt), min_dt.tzinfo)

            pairs = sorted(pairs, key=itemgetter(1), reverse=True)
            max_version = next(pair[0] for pair in pairs if pair[1] <= tag_date_tz)
            # print(max_version)

            dep = f"{pkg}<={max_version}"
            print(dep)
            deps.append(dep)

    # print(pkgs)
    # print(repr(pkgs))
    # print(deps)

    with open("requirements.txt", "w") as f:
        f.write(f"# Reference date: {tag_date_tz.strftime('%b %d, %Y')}")
        f.write("\n" * 2)
        f.write("\n".join(deps))
        f.write("\n")

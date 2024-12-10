#!/bin/bash
set -euo pipefail

# example $1 = https://github.com/kristovatlas/multi-sig-check-demo/blob/78231bb8cff94323d0fff827f745db1905146028/examples/osx-config-check-master/app.py

clone_url=${1%/blob/*}
clone_dir=$(basename $clone_url)
file_path=${1#*blob/*/}
file_path=$(printf '%b' "${file_path//%/\\x}")
#file_path=$(printf '%s' "${file_path// /\\ }") # take care of spaces in file paths
#file_path=$(printf '%s' "${file_path//[/\\[}") # only one sample needed these two statements
#file_path=$(printf '%s' "${file_path//]/\\]}")
old_commit=${1#*blob/}
old_commit=${old_commit%%/*}

cd /tmp/workdir
git clone $clone_url 2>/dev/null # you will need to comment this out for the Programming-CookBook sample and instead follow https://stackoverflow.com/a/26558656
cd "./$clone_dir"

new_urls=""
new_dates=""
new_linenos=""
so_search=$(cat "$file_path" | grep -n "stackoverflow.com" -)
while read -r line; do
    so_match_lineno=${line%%:*}
    so_match=${line#*:}
    so_match=$(printf '%s' "${so_match//\\/\\\\}") # take care of backslashes in the input
    so_match=$(printf '%s' "${so_match//\*/\\\*}")
    so_match=$(printf '%s' "${so_match//[/\\[}")
    so_match=$(printf '%s' "${so_match//]/\\]}")
    gitlog_output=$(git --no-pager log -p --date=unix -L $so_match_lineno,$so_match_lineno:"$file_path") # bc this will follow renames, we need to grab the other "new" file_path later
    gitlog_match_lineno=$(echo -e "$gitlog_output" | grep -n -- "$so_match" - | tail -n 1 | cut -d ':' -f1)
    commit_lineno=$(echo -e "$gitlog_output" | sed -n "1, $gitlog_match_lineno p" - | grep -n "commit " - | grep ":commit" - | tail -n 1 | cut -d ':' -f1) #tail bc latest..oldest, we want oldest
    commit_line=$(echo -e "$gitlog_output" | sed -n "$commit_lineno, $commit_lineno p" -)
    commit_hash=${commit_line#commit\ } # final value
    date_lineno=$(("$commit_lineno+2"))
    date=$(echo -e "$gitlog_output" | sed -n "$date_lineno, $date_lineno p" -)
    date=${date#Date:\ \ \ } # final value
    new_filepath_lineno=$(echo -e "$gitlog_output" | sed -n "$commit_lineno, $gitlog_match_lineno p" | grep -n "diff --git" -)
    new_filepath_lineno=${new_filepath_lineno%:*}
    new_filepath_lineno=$(("$commit_lineno+$new_filepath_lineno-1")) # the location of the diff -git line
    new_filepath_lineno=$(("$new_filepath_lineno+2")) #the +++ line is only two lines ahead
    new_filepath_line=$(echo -e "$gitlog_output" | sed -n "$new_filepath_lineno, $new_filepath_lineno p" -)
    new_filepath=${new_filepath_line#\+\+\+ b/}

    new_url="${clone_url}/blob/${commit_hash}/${new_filepath}"
    new_urls+="${new_url}\n"
    new_dates+="${date}\n"
    new_linenos+="${so_match_lineno}\n"
done <<< "$so_search"

echo -e "$new_urls"
echo -e "$new_dates"
echo -e "$new_linenos"
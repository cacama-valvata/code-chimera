#!/bin/bash
set -euo pipefail

# example $1 = https://github.com/kristovatlas/multi-sig-check-demo/blob/78231bb8cff94323d0fff827f745db1905146028/examples/osx-config-check-master/app.py

clone_url=${1%/blob/*}
clone_dir=$(basename $clone_url)
file_path=${1#*blob/*/}
file_path=$(printf '%b' "${file_path//%/\\x}")
file_path=$(printf '%s' "${file_path// /\\ }") # take care of spaces in file paths
file_path=$(printf '%s' "${file_path//[/\\[}") # only one sample needed these two statements
file_path=$(printf '%s' "${file_path//]/\\]}")
old_commit=${1#*blob/}
old_commit=${old_commit%%/*}

cd /tmp/workdir
git clone $clone_url 2>/dev/null
cd "./$clone_dir"

gitlog_output=$(git --no-pager log -p --follow -- "$file_path")
#commit_lines=$(echo "$gitlog_output" | grep -n "commit " - | grep ":commit" - | cut -d ':' -f1)
so_lines=$(echo "$gitlog_output" | grep -n "stackoverflow.com" | cut -d ":" -f1)

new_urls=""
new_dates=""
while read -r line; do
    line_number=$(echo "$gitlog_output" | sed -n "1, $line p" - | grep -n "commit " - | grep ":commit" - | cut -d ':' -f1 | tail -n 1)
    date_line_number=$(("$line_number+2"))
    commit_line=$(echo "$gitlog_output" | sed -n "$line_number p")
    commit_hash=${commit_line#commit\ }
    date_line=$(echo "$gitlog_output" | sed -n "$date_line_number p")
    date=${date_line#Date:\ \ \ }
    # There's always at least line 1 so I'm not going to check for empty output before filtering to commit_hash
    new_url="${clone_url}/blob/${commit_hash}/${file_path}"
    new_urls+="${new_url}\n"
    new_dates+="${date}\n"
done <<< "$so_lines"

echo -e "$new_urls"
echo -e "$new_dates"
echo -e "$so_lines"
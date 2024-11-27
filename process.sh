# Weeeeee
set -euo pipefail

cd workdir
git clone $1
dir=$(basename $1)
cd $dir

git --no-pager log -p --follow $2

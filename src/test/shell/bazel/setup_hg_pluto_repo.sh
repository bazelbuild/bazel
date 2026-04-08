#!/usr/bin/env bash
#
# Copyright 2026 The Bazel Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Creates a reproducible test mercurial repository in the current directory. The commits match the
# equivalent git repositories used in the starlark_git_repository_test. After each commit group, the
# changeset id (SHA hash) is checked to verify that the commit metadata is reproducible.
set -e

# Ensure reproducibility by avoiding the mercurial config files.
export HGRCPATH=

hg init

# Commit 1 (tag: 0-initial)
echo "Pluto is a planet" > info
hg add

TZ=PDT+07:00 HGUSER="John Doe <john@example.com>" \
hg commit --date "2015-07-16 04:49:35" --message "Initial commit."

TZ=PDT+07:00 HGUSER="John Doe <john@example.com" \
hg tag --date "2015-07-16 04:49:35" --message "tag 0-initial" 0-initial

SHA=$(hg id --id --debug)
if [[ $SHA != "16a138603930599de37a8b5ee52ddf903b633c45" ]]; then
  echo "Hg SHA ($SHA) wrong for first pluto commit" >&2
  exit 1
fi

# Commit 2 (tag: 1-build)
cat <<EOF > BUILD
filegroup(
    name = "pluto",
    srcs = ["info"],
    visibility = ["//visibility:public"],
)
EOF
touch WORKSPACE
echo "Pluto is a dwarf planet" > info
hg add

TZ=PDT+07:00 HGUSER="John Doe <john@example.com>" \
hg commit --date "2015-07-16 04:50:53" --message "Add WORKSPACE and BUILD file. Update info because Pluto is no longer a planet."

TZ=PDT+07:00 HGUSER="John Doe <john@example.com" \
hg tag --date "2015-07-16 04:50:53" --message "tag 1-build" 1-build

SHA=$(hg id --id --debug)
if [[ $SHA != "baa0483a956434b5a104147d52de87eb03cb5654" ]]; then
  echo "Hg SHA ($SHA) wrong for second pluto commit" >&2
  exit 1
fi

# Commit 3 (tag: 2-subdir)
mkdir pluto
hg mv BUILD info pluto
TZ=UTC HGUSER="John Doe <john@example.com>" \
hg commit --date "2017-12-27 20:54:31" --message "Move pluto files into pluto subdirectory"
TZ=UTC HGUSER="John Doe <john@example.com" \
hg tag --date "2017-12-27 20:54:31" --message "tag 2-subdir" 2-subdir
SHA=$(hg id --id --debug)
if [[ $SHA != "a5c5a6bc585fe5a906aaf40a72f87b9835cfbbc9" ]]; then
  echo "Hg SHA ($SHA) wrong for third pluto commit" >&2
  exit 1
fi

# Commit 4 (tag: 3-subdir-bar)
hg rm pluto/BUILD WORKSPACE

TZ=EDT+05:00 HGUSER="John Doe <john@example.com>" \
hg commit --date "2017-12-29 20:33:35" --message "Remove BUILD and WORKSPACE files"

TZ=EDT+05:00 HGUSER="John Doe <john@example.com" \
hg tag --date "2017-12-29 20:33:35" --message "tag 3-subdir-bare" 3-subdir-bare

SHA=$(hg id --id --debug)
if [[ $SHA != "e0fd3d20e082378d786f69319f6b74117ac4dc88" ]]; then
  echo "Hg SHA ($SHA) wrong for fourth pluto commit" >&2
  exit 1
fi

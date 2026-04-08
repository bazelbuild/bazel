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

# Commit 1
echo 'genrule(name="g", srcs=[], outs=["go"], cmd="echo GIT 1 > $@")' > BUILD
touch WORKSPACE
hg add

TZ=UTC-01:00 HGUSER="John Doe <john@example.com>" \
hg commit --date "2015-11-25 13:19:52" --message "Initial checkin."

SHA=$(hg id --id --debug)
if [[ $SHA != "92872ece5144313d5f29000322ea77c9a7d7159a" ]]; then
  echo "Hg SHA ($SHA) wrong for first refetch commit" >&2
  exit 1
fi

# Commit 2
echo 'genrule(name="g", srcs=[], outs=["go"], cmd="echo GIT 2 > $@")' > BUILD
hg add

TZ=UTC-01:00 HGUSER="John Doe <john@example.com>" \
hg commit --date "2015-11-25 13:20:07" --message "Change 1 to 2"

SHA=$(hg id --id --debug)
if [[ $SHA != "e34c5d9fe274c05d6ff499d249a5461770b8be8e" ]]; then
  echo "Hg SHA ($SHA) wrong for second refetch commit" >&2
  exit 1
fi

# Commit 3
mkdir gdir
hg mv BUILD gdir
TZ=EST+05:00 HGUSER="John Doe <john@example.com>" \
hg commit --date "2017-12-29 18:36:06" --message "Move BUILD into a subdirectory"
SHA=$(hg id --id --debug)
if [[ $SHA != "ddfce8df599ad91e9f3416ed1a89f60321fa3f05" ]]; then
  echo "Hg SHA ($SHA) wrong for third refetch commit" >&2
  exit 1
fi

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
mkdir prefix-foo
echo "FOO=123" > prefix-foo/defs.bzl
touch prefix-foo/WORKSPACE
touch prefix-foo/BUILD
hg add .
TZ=UTC-01:00 HGUSER="Yun Peng <pcloudy@google.com>" \
hg commit --date "2020-11-23 17:31:04" --message "First commit"
SHA=$(hg id --id --debug)
if [[ $SHA != "ee31cd98d80d6c9b6a2e1dd0fdfeb27538da3163" ]]; then
  echo "Hg SHA ($SHA) wrong for first pluto commit" >&2
  exit 1
fi

#! /usr/bin/env bash
#
# Copyright 2015 Andrew Z Allen <me@andrewzallen.com>. All rights reserved.
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


# This script looks at the files changed in git against origin/master and
# queries for all build and test targets associated with those files.
#
# Running this script on a CI server should allow you to only test the targets
# that have changed since the last time your merged or fast forwarded.
#
# This script can be used to recreate the benefits that TAP provides to
# Google's developers as describe by Mike Bland on his article on Google's
# infrastructure.
# https://mike-bland.com/2012/10/01/tools.html#tools-tap-sponge
#
# "Every single change submitted to Google’s Perforce depot is built and
# tested, and only those targets affected by a particular change are
# built and tested"
#
# When your code is running on Gerrit (for example) and you have a fast-forward
# only policy on your repo you can be sure, by diffing from the origin/master..
# refspec, that you're listing all the files that are included in a review.
# This would let a team working on Gerrit have the verified bit be
# autopopulated without having the pain of having to wait for the whole world
# to compile.

REFSPEC="origin/master.."

# Go to the root of the repo
cd "$(git rev-parse --show-toplevel)"

# Get a list of the current files in package form by querying Bazel.
files=()
for file in $(git diff --name-only ${REFSPEC} ); do
  files+=($(bazel query $file))
  echo $(bazel query $file)
done

# Query for the associated buildables
buildables=$(bazel query --keep_going "kind(.*_binary, rdeps(//..., set(${files[@]})))")
# Run the tests if there were results
if [[ ! -z $buildables ]]; then
  echo "Building binaries"
  bazel build $buildables
fi

tests=$(bazel query --keep_going "kind(test, rdeps(//..., set(${files[@]})))")
# Run the tests if there were results
if [[ ! -z $tests ]]; then
  echo "Running tests"
  bazel test $tests
fi

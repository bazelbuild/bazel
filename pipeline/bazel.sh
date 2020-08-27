#!/bin/bash
set -euox pipefail

[[ ${1:-} == "postmerge" ]] && POST_MERGE="true" || POST_MERGE="false"

BAZEL_BIN=tools/bazel
HOST=$(uname -s | tr 'A-Z' 'a-z')
MACHINE=$(uname -m)

RELEASE_NAME=$(git describe --tags HEAD)

BAZEL_OPTS_COMMON=(
    --announce_rc
)

BAZEL_OPTS_BUILD=(
    ${BAZEL_OPTS_COMMON[@]}
    --remote_upload_local_results=true
    --embed_label=${RELEASE_NAME}
)

if [ "${POST_MERGE}" == "true" ]; then
    RELEASE_NAME=$(git describe --tags HEAD)
else
    RELEASE_NAME="precommit-${GERRIT_CHANGE_NUMBER}-${GERRIT_PATCHSET_NUMBER}-${BUILD_NUMBER}"
fi
# Clean/Expunge first, if any builds configured XCode incorrectly, i.e.
# it was not present at invocation, the repository rule will not recreate
# the required "osx_arch.bzl" file. Not very hermetic Bazel....
"${BAZEL_BIN}" clean --expunge "${BAZEL_OPTS_COMMON[@]}"

# Build release version
"${BAZEL_BIN}" build "${BAZEL_OPTS_BUILD[@]}" -- //src:bazel

mkdir -p uploads
cp "bazel-bin/src/bazel" "uploads/bazel-${RELEASE_NAME}-${HOST}-${MACHINE}"

#!/bin/bash

set -euo pipefail

# Parsing the flags.
while [[ -n "$@" ]]; do
  arg="$1"; shift
  val="$1"; shift
  case "$arg" in
    "--java_version") java_version="$val" ;;
    "--java_tools_version") java_tools_version="$val" ;;
    "--commit_hash") commit_hash="$val" ;;
    "--rc") timestamp="$val" ;;
    *) echo "Flag $arg is not recognized." && exit 1 ;;
  esac
done

for platform in linux windows darwin; do
  src_artifact=$(gsutil ls -lh gs://bazel-mirror/bazel_java_tools/tmp/build/${commit_hash}/java${java_version}/java_tools_javac${java_version}_${platform}* | sort -k 2 | grep "gs" | cut -d " " -f 7)
  dest_artifact="gs://bazel-mirror/bazel_java_tools/release_candidates/javac${java_version}/${java_tools_version}/java_tools_javac${java_version}_${platform}-${java_tools_version}-rc${rc}.zip"
  gsutil cp ${src_artifact} ${dest_artifact}
done



#!/bin/bash -e

readonly SDK_VERSION='1.11.297'
readonly MVN_REPO='repo1.maven.org/maven2'

fetch_artifact() {
  local artifact="$1"
  local src_jar="$artifact-$SDK_VERSION-sources.jar"

  wget "https://$MVN_REPO/com/amazonaws/$artifact/$SDK_VERSION/$src_jar"
  echo "$src_jar"
}

fetch_and_unpack() {
  local out_dir="$1"
  local core_jar
  local s3_jar

  core_jar="$(fetch_artifact 'aws-java-sdk-core')"
  s3_jar="$(fetch_artifact 'aws-java-sdk-s3')"

  # Unpack the core source first
  unzip "$core_jar" -x 'META-INF/*' -d "$out_dir"
  rm "$core_jar"

  # Grab Region from the S3 jar
  unzip "$s3_jar" 'com/amazonaws/services/s3/model/Region.java' -d "$out_dir"
  rm "$s3_jar"
}

patch_upstream() {
  echo "Applying (or attempting to apply) bazel patches against AWS SDK"

  while read -r patch; do
    echo "Patch: $patch"
    patch -p0 < "$patch"
  done < <(find patches-vs-$SDK_VERSION -name '*.patch')
}

update_sdk() {
  mkdir -p ./src/main/java
  fetch_and_unpack ./src/main/java
  patch_upstream
}

update_sdk

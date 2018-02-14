#!/usr/bin/env bash

set -eux -o pipefail

# Empty string is valid for package fix version
PACKAGE_FIX_VERSION="${PACKAGE_FIX_VERSION:-}"
if [[ -z "${VERSION+x}" ]]; then echo "VERSION is unset" && exit 1; else echo "VERSION is set."; fi
if [[ -z "${BAZEL_RELEASE_NOTES_URL+x}" ]]; then echo "BAZEL_RELEASE_NOTES_URL is unset" && exit 1; else echo "BAZEL_RELEASE_NOTES_URL is set."; fi
if [[ -z "${BAZEL_WINDOWS_ZIP_URL+x}" ]]; then echo "BAZEL_WINDOWS_ZIP_URL is unset" && exit 1; else echo "BAZEL_WINDOWS_ZIP_URL is set."; fi
if [[ -z "${BAZEL_WINDOWS_ZIP_SHA256+x}" ]]; then echo "BAZEL_WINDOWS_ZIP_SHA256 is unset" && exit 1; else echo "BAZEL_WINDOWS_ZIP_SHA256 is set."; fi
if [[ -z "${CHOCOLATEY_API_KEY+x}" ]]; then echo "CHOCOLATEY_API_KEY is unset" && exit 1; else echo "CHOCOLATEY_API_KEY is set."; fi

if [[ -n "${PACKAGE_FIX_VERSION}" ]]; then PACKAGE_FIX_VERSION=".${PACKAGE_FIX_VERSION}"; fi
pushd "bazel-package"
  package_name="bazel"
  # clean slate
  rm -rf "tools/LICENSE" "tools/params.txt" "${package_name}.nuspec" "${package_name}.${VERSION}.nupkg"
  ls -al

  # expand the bazel.nuspec.template file & write that to bazel.nuspec
  sed -e "s|\$(\$tvVersion)|${VERSION}|" < "${package_name}.nuspec.template" \
    | sed -e "s|\$(\$tvPackageFixVersion)|${PACKAGE_FIX_VERSION}|" \
    | sed -e "s|\$(\$tvReleaseNotesUri)|${BAZEL_RELEASE_NOTES_URL}|" \
    > ${package_name}.nuspec

  # create the tools/params.txt file
  cat > "tools/params.txt" << EOMultiLine
${BAZEL_WINDOWS_ZIP_URL}
${BAZEL_WINDOWS_ZIP_SHA256}
EOMultiLine

  # download license into tools/LICENSE
  curl -o "tools/LICENSE" "https://raw.githubusercontent.com/bazelbuild/bazel/master/LICENSE"
  # adjust the license to have the header
  echo -e "From: https://raw.githubusercontent.com/bazelbuild/bazel/master/LICENSE\n\n$(cat tools/LICENSE)" > "tools/LICENSE"

  ls -al
  pwd

  echo "running 'choco pack ./bazel.nuspec' inside docker..."
  # $ docker run --rm --volume $PWD:$PWD -w $PWD linuturk/mono-choco pack bazel.nuspec
  docker run \
    --rm \
    --volume "${PWD}:${PWD}" \
    --workdir "${PWD}" \
    choco \
    pack "${package_name}.nuspec"

  # echo "running 'choco push <package> --key <key>' inside docker..."
  # docker run \
  #   --rm \
  #   --mount "type=bind,src=$(pwd),dst=/work" \
  #   choco \
  #   push --verbose --debug "/work/${package_name}.${VERSION}${PACKAGE_FIX_VERSION}.nupkg" --timeout "30" --apikey="${CHOCOLATEY_API_KEY}"
popd

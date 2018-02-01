#!/usr/bin/env bash

set -eux -o pipefail

# Empty string is valid for package fix version
BAZEL_PACKAGE_FIX_VERSION="${BAZEL_PACKAGE_FIX_VERSION:-}"

# clean slate
rm -rf "tools/LICENSE" "tools/params.txt" "bazel.nuspec"

# expand the bazel.nuspec.template file & write that to bazel.nuspec
sed -e "s/\$(\$tvVersion)/${BAZEL_VERSION}/" < "bazel.nuspec.template" \
  | sed -e "s/\$(\$tvPackageFixVersion)/${BAZEL_PACKAGE_FIX_VERSION}/" \
  | sed -e "s/\$(\$tvReleaseNotesUri)/${BAZEL_RELEASE_NOTES_URL}/" \
  > bazel.nuspec

# cat bazel.nuspec.template \
#   | sed -e "s/\$(\$tvVersion)/${BAZEL_VERSION}/" \
#   | sed -e "s/\$(\$tvPackageFixVersion)/${BAZEL_PACKAGE_FIX_VERSION}/" \
#   | sed -e "s/\$(\$tvReleaseNotesUri)/${BAZEL_RELEASE_NOTES_URL}/" \
#   > bazel.nuspec

# create the tools/params.txt file
cat > "tools/params.txt" << EOMultiLine
${BAZEL_WINDOWS_EXE_URL}
${BAZEL_WINDOWS_EXE_SHA256}
EOMultiLine

# download license into tools/LICENSE
curl -o "tools/LICENSE" "https://raw.githubusercontent.com/bazelbuild/bazel/master/LICENSE"
# adjust the license to have the header
echo -e "From: https://raw.githubusercontent.com/bazelbuild/bazel/master/LICENSE\n\n$(cat tools/LICENSE)" > "tools/LICENSE"
# run `choco pack ./bazel.nuspec`
docker run \
  --rm \
  --mount type=bind,src="$(pwd)",dst=/work \
  linuturk/mono-choco \
  pack /work/bazel.nuspec
# run `choco push --key <key>`
docker run \
  --rm \
  --mount type=bind,src="$(pwd)",dst=/work \
  linuturk/mono-choco \
  push "${BAZEL_VERSION}.nupkg" --apikey="${BAZEL_CHOCOLATEY_API_KEY}"

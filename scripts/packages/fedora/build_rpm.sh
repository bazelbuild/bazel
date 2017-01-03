#!/bin/sh

# Copyright 2016 The Bazel Authors. All rights reserved.
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

set -eu

# Build a RPM archive from the Bazel sources.

# Usage: build_rpm.sh spec_file dest_file [other files]

spec_file=$1
shift
dest_file=$1
shift

echo "Building ${dest_file} from ${spec_file}."
WORK_DIR="${PWD}/bazel-fedora"

# Copy needed sources.
rm -rf ${WORK_DIR}
mkdir -p ${WORK_DIR}/SOURCES
mkdir -p ${WORK_DIR}/BUILD
cp $spec_file ${WORK_DIR}
for i in "$@"; do
  cp $i $WORK_DIR/BUILD
done

# Build the RPM.
rpmbuild \
  --define "_topdir ${WORK_DIR}" \
  --define "_tmppath /tmp" \
  -bb ${spec_file} > rpmbuild.log 2>&1
if [ $? -ne 0 ]; then
  err=$?
  echo "Error in rpmbuild:"
  cat rpmbuild.log
  exit $err
fi
out_file=$(grep '^Wrote:' rpmbuild.log | cut -d ' ' -f 2)

# Copy output back to the destination.
cp $out_file $dest_file
echo "Created $dest_file"


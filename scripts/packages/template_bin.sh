#!/bin/bash -e

# Copyright 2015 The Bazel Authors. All rights reserved.
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

# Bazel self-extractable installer

# Installation and etc prefix can be overriden from command line
install_prefix=${1:-"/usr/local"}
bazelrc=${2:-"/etc/bazel.bazelrc"}

progname="$0"

echo "Bazel installer"
echo "---------------"
echo
cat <<'EOF'
%release_info%
EOF

function usage() {
  echo "Usage: $progname [options]" >&2
  echo "Options are:" >&2
  echo "  --prefix=/some/path set the prefix path (default=/usr/local)." >&2
  echo "  --bazelrc= set the bazelrc path (default=/etc/bazel.bazelrc)." >&2
  echo "  --bin= set the binary folder path (default=%prefix%/bin)." >&2
  echo "  --base= set the base install path (default=%prefix%/lib/bazel)." >&2
  echo "  --user configure for user install, expands to" >&2
  echo '           `--bin=$HOME/bin --base=$HOME/.bazel --bazelrc=$HOME/.bazelrc`.' >&2
  exit 1
}

prefix="/usr/local"
bin="%prefix%/bin"
base="%prefix%/lib/bazel"
bazelrc="/etc/bazel.bazelrc"

for opt in "${@}"; do
  case $opt in
    --prefix=*)
      prefix="$(echo "$opt" | cut -d '=' -f 2-)"
      ;;
    --bazelrc=*)
      bazelrc="$(echo "$opt" | cut -d '=' -f 2-)"
      ;;
    --bin=*)
      bin="$(echo "$opt" | cut -d '=' -f 2-)"
      ;;
    --base=*)
      base="$(echo "$opt" | cut -d '=' -f 2-)"
      ;;
    --user)
      bin="$HOME/bin"
      base="$HOME/.bazel"
      bazelrc="$HOME/.bazelrc"
      ;;
    *)
      usage
      ;;
  esac
done

bin="${bin//%prefix%/${prefix}}"
base="${base//%prefix%/${prefix}}"
bazelrc="${bazelrc//%prefix%/${prefix}}"

function test_write() {
  local file="$1"
  while [ "$file" != "/" ] && [ -n "${file}" ] && [ ! -e "$file" ]; do
    file="$(dirname "${file}")"
  done
  [ -w "${file}" ] || {
    echo >&2
    echo "The Bazel installer must have write access to $1!" >&2
    echo >&2
    usage
  }
}

# Test for dependencies
# unzip
if ! which unzip >/dev/null; then
  echo >&2
  echo "unzip not found, please install the corresponding package." >&2
  echo "See http://bazel.io/docs/install.html for more information on" >&2
  echo "dependencies of Bazel." >&2
  exit 1
fi

# java
if [ -z "${JAVA_HOME-}" ]; then
  case "$(uname -s | tr 'A-Z' 'a-z')" in
    linux)
      JAVA_HOME="$(readlink -f $(which javac) 2>/dev/null | sed 's_/bin/javac__')" || true
      BASHRC="~/.bashrc"
      ;;
    freebsd)
      JAVA_HOME="/usr/local/openjdk8"
      BASHRC="~/.bashrc"
      ;;
    darwin)
      JAVA_HOME="$(/usr/libexec/java_home -v ${JAVA_VERSION}+ 2> /dev/null)" || true
      BASHRC="~/.bash_profile"
      ;;
  esac
fi
if [ ! -x "${JAVA_HOME}/bin/javac" ]; then
  echo >&2
  echo "Java not found, please install the corresponding package" >&2
  echo "See http://bazel.io/docs/install.html for more information on" >&2
  echo "dependencies of Bazel." >&2
  exit 1
fi

# Test for write access
test_write "${bin}"
test_write "${base}"
test_write "${bazelrc}"

# Do the actual installation
echo -n "Uncompressing."

# Cleaning-up, with some guards.
if [ -f "${bin}/bazel" ]; then
  rm -f "${bin}/bazel"
fi
if [ -d "${base}" -a -x "${base}/bin/bazel" ]; then
  rm -fr "${base}"
fi

mkdir -p ${bin} ${base} ${base}/bin ${base}/etc ${base}/base_workspace
echo -n .

unzip -q "${BASH_SOURCE[0]}" bazel bazel-real bazel-complete.bash -d "${base}/bin"
echo -n .
chmod 0755 "${base}/bin/bazel" "${base}/bin/bazel-real"
unzip -q "${BASH_SOURCE[0]}" -x bazel bazel-real bazel-complete.bash -d "${base}/base_workspace"
echo -n .
cat >"${base}/etc/bazel.bazelrc" <<EO
build --package_path %workspace%:${base}/base_workspace
fetch --package_path %workspace%:${base}/base_workspace
query --package_path %workspace%:${base}/base_workspace
EO
echo -n .
chmod -R og-w "${base}"
chmod -R og+rX "${base}"
chmod -R u+rwX "${base}"
echo -n .

ln -s "${base}/bin/bazel" "${bin}/bazel"
echo -n .

# Uncompress the bazel base install for faster startup time
"${bin}/bazel" help >/dev/null

if [ -f "${bazelrc}" ]; then
  echo
  echo "${bazelrc} already exists, ignoring. It is either a link to"
  echo "${base}/etc/bazel.bazelrc or that it's importing that file with:"
  echo "  import ${base}/etc/bazel.bazelrc"
else
  ln -s "${base}/etc/bazel.bazelrc" "${bazelrc}"
  echo .
fi

cat <<EOF

Bazel is now installed!

Make sure you have "${bin}" in your path. You can also activate bash
completion by adding the following line to your ${BASHRC}:
  source ${base}/bin/bazel-complete.bash

See http://bazel.io/docs/getting-started.html to start a new project!
EOF
exit 0

#!/bin/bash
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

readonly OUTPUT=${PWD}/$1
shift
readonly JEKYLL_BASE=${PWD}/$1
shift
readonly SKYLARK_RULE_DOCS=${PWD}/$1
shift
readonly BE_ZIP=${PWD}/$1
shift
readonly SL_ZIP=${PWD}/$1
shift
readonly CLR_HTML=${PWD}/$1

# Create temporary directory that is removed when this script exits.
readonly TMP=$(mktemp -d "${TMPDIR:-/tmp}/tmp.XXXXXXXX")
readonly OUT_DIR="$TMP/out"
trap "rm -rf ${TMP}" EXIT

function setup {
  mkdir -p "$OUT_DIR"
  cd "$OUT_DIR"
  tar -xf "${JEKYLL_BASE}"
}

# Unpack the Build Encyclopedia into docs/be
function unpack_build_encyclopedia {
  local be_dir="$OUT_DIR/versions/master/docs/be"
  mkdir -p "$be_dir"
  unzip -qq "$BE_ZIP" -d "$be_dir"
  mv "$be_dir/be-nav.html" "$OUT_DIR/_includes"

  # Create redirects to each page in the Build Encyclopedia.
  mkdir -p "$OUT_DIR/docs/be"
  for f in $(find "$OUT_DIR/versions/master/docs/be" -name "*.html"); do
    local filename=$(basename "$f")
    cat > "$OUT_DIR/docs/be/${filename}" <<EOF
---
layout: redirect
redirect: docs/be/${filename}
---
EOF
  done
}

# Unpack the Skylark Library into docs/skylark/lib
function unpack_skylark_library {
  local sl_dir="$OUT_DIR/versions/master/docs/skylark/lib"
  mkdir -p "$sl_dir"
  unzip -qq "$SL_ZIP" -d "$sl_dir"
  mv "$sl_dir/skylark-nav.html" "$OUT_DIR/_includes"

  # Create redirects to each page in the Skylark Library
  mkdir -p "$OUT_DIR/docs/skylark/lib"
  for f in $(find "$OUT_DIR/versions/master/docs/skylark/lib" -name "*.html"); do
    local filename=$(basename "$f")
    cat > "$OUT_DIR/docs/skylark/lib/${filename}" <<EOF
---
layout: redirect
redirect: docs/skylark/lib/${filename}
---
EOF
  done
}

function copy_skylark_rule_doc {
  local rule_family=$1
  local rule_family_name=$2
  local be_dir="$OUT_DIR/versions/master/docs/be"

  ( cat <<EOF
---
layout: documentation
title: ${rule_family_name} Rules
---
EOF
    cat "$TMP/skylark/$rule_family/README.md"; ) > "$be_dir/${rule_family}.md"
}

function unpack_skylark_rule_docs {
  local tmp_dir=$TMP/skylark
  mkdir -p $tmp_dir
  cd "$tmp_dir"
  tar -xf "${SKYLARK_RULE_DOCS}"
  copy_skylark_rule_doc docker "Docker"
  copy_skylark_rule_doc pkg "Packaging"
}

function process_doc {
  local f=$1
  local tempf=$(mktemp -t bazel-doc-XXXXXX)

  chmod +w $f
  cat "$f" | sed 's,\.md,.html,g;s,Blaze,Bazel,g;s,blaze,bazel,g' > "$tempf"
  cat "$tempf" > "$f"
}

function process_docs {
  for f in $(find "$OUT_DIR/versions/master/docs" -name "*.html"); do
    process_doc $f
  done
  for f in $(find "$OUT_DIR/versions/master/docs" -name "*.md"); do
    process_doc $f
  done
  for f in $(find "$OUT_DIR/designs" -name "*.md"); do
    process_doc $f
  done
}

function package_output {
  cd "$OUT_DIR"
  tar -hcf $OUTPUT $(find . -type f | sort)
}

function main {
  setup
  unpack_build_encyclopedia
  unpack_skylark_library
  unpack_skylark_rule_docs
  cp ${CLR_HTML} ${OUT_DIR}/versions/master/docs
  process_docs
  package_output
}
main

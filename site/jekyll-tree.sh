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

# This script constructs the final Jekyll tree by combining the static Jekyll
# site files with generated documentation, such as the Build Encyclopedia and
# Skylark Library. It then constructs the site directory structure for
# Bazel documentation at HEAD by moving all documentation into the
# /versions/master directory and adding redirects from the root of the site.
# This way, URLs of the form https://docs.bazel.build/foo.html will be
# redirected to https://docs.bazel.build/versions/master/foo.html.

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
shift
readonly REPO_ZIP="${PWD}/$1"

# Create temporary directory that is removed when this script exits.
readonly TMP=$(mktemp -d "${TMPDIR:-/tmp}/tmp.XXXXXXXX")
readonly OUT_DIR="$TMP/out"
trap "rm -rf ${TMP}" EXIT

readonly VERSION="${DOC_VERSION:-master}"
readonly VERSION_DIR="$OUT_DIR/versions/$VERSION"

# Unpacks the base Jekyll tree, Build Encyclopedia, etc.
function setup {
  mkdir -p "$OUT_DIR"
  cd "$OUT_DIR"
  tar -xf "${JEKYLL_BASE}"

  mkdir -p "$VERSION_DIR"
  mv "$OUT_DIR"/docs/* "$VERSION_DIR"
  rm -r "$OUT_DIR"/docs

  # Unpack the Build Encyclopedia into versions/master/be
  local be_dir="$VERSION_DIR/be"
  mkdir -p "$be_dir"
  unzip -qq "$BE_ZIP" -d "$be_dir"
  mv "$be_dir/be-nav.html" "$OUT_DIR/_includes"

  # Unpack the Skylark Library into versions/master/skylark/lib
  local sl_dir="$VERSION_DIR/skylark/lib"
  mkdir -p "$sl_dir"
  unzip -qq "$SL_ZIP" -d "$sl_dir"
  mv "$sl_dir/skylark-nav.html" "$OUT_DIR/_includes"

  # Unpack the the documentation for the repository rules to repo subdirectory
  local repo_dir="${VERSION_DIR}/repo"
  mkdir -p "${repo_dir}"
  unzip -qq "${REPO_ZIP}" -d "${repo_dir}"

  # Copy the command line reference.
  cp "$CLR_HTML" "$VERSION_DIR"
}

# Helper function for copying a Skylark rule doc.
function copy_skylark_rule_doc {
  local rule_family=$1
  local rule_family_name=$2
  local be_dir="$VERSION_DIR/be"

  ( cat <<EOF
---
layout: documentation
title: ${rule_family_name} Rules
---
EOF
    cat "$TMP/skylark/$rule_family/README.md"; ) > "$be_dir/${rule_family}.md"
}

# Copies the READMEs for Skylark rules bundled with Bazel.
function unpack_skylark_rule_docs {
  local tmp_dir=$TMP/skylark
  mkdir -p $tmp_dir
  cd "$tmp_dir"
  tar -xf "${SKYLARK_RULE_DOCS}"
  copy_skylark_rule_doc pkg "Packaging"
}

# Processes a documentation page, such as replacing Blaze with Bazel.
function process_doc {
  local f=$1
  local tempf=$(mktemp -t bazel-doc-XXXXXX)

  chmod +w $f
  cat "$f" | sed 's,\.md,.html,g;s,Blaze,Bazel,g;s,blaze,bazel,g' > "$tempf"
  cat "$tempf" > "$f"
}

# Performs fixup on each doc, such as replacing instances of 'blaze' with
# 'bazel'.
function process_docs {
  for f in $(find "$VERSION_DIR" -name "*.html"); do
    process_doc $f
  done
  for f in $(find "$VERSION_DIR" -name "*.md"); do
    process_doc $f
  done
}

# Generates a redirect for a documentation page under /versions/master.
function gen_redirect {
  local output_dir=$OUT_DIR/$(dirname $f)
  if [[ ! -d "$output_dir" ]]; then
    mkdir -p "$output_dir"
  fi

  local src_basename=$(basename $f)
  local md_basename="${src_basename%.*}.md"
  local html_file="${f%.*}.html"
  local redirect_file="$output_dir/$md_basename"
  if [[ -e "$redirect_file" ]]; then
    echo "Cannot create redirect file $redirect_file. File exists."
    exit 1
  fi
  cat > "$redirect_file" <<EOF
---
layout: redirect
redirect: /versions/$VERSION/$html_file
---
EOF
}

# During setup, all documentation under docs are moved to the /versions/master
# directory as the documentation from HEAD.
#
# This function henerates a redirect from the root of the site for the given
# doc page under /versions/master so that https://docs.bazel.build/foo.html
# will be redirected to https://docs.bazel.build/versions/master/foo.html
function gen_redirects {
  pushd "$VERSION_DIR" > /dev/null
  for f in $(find . -name "*.html" -type f); do
    gen_redirect $f
  done
  for f in $(find . -name "*.md" -type f); do
    gen_redirect $f
  done
  popd > /dev/null
}

# Creates a tar archive containing the final Jekyll tree.
function package_output {
  cd "$OUT_DIR"
  tar -hcf $OUTPUT $(find . -type f | sort)
}

function main {
  setup
  unpack_skylark_rule_docs
  process_docs
  gen_redirects
  package_output
}
main

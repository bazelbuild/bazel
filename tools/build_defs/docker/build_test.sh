#!/bin/bash

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

# Unit tests for docker_build

DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source ${DIR}/testenv.sh || { echo "testenv.sh not found!" >&2; exit 1; }

readonly PLATFORM="$(uname -s | tr 'A-Z' 'a-z')"
if [ "${PLATFORM}" = "darwin" ]; then
  readonly MAGIC_TIMESTAMP="$(date -r 0 "+%b %e  %Y")"
else
  readonly MAGIC_TIMESTAMP="$(date --date=@0 "+%F %R")"
fi

function EXPECT_CONTAINS() {
  local complete="${1}"
  local substring="${2}"
  local message="${3:-Expected '${substring}' not found in '${complete}'}"

  echo "${complete}" | grep -Fsq -- "${substring}" \
    || fail "$message"
}

function check_property() {
  local property="${1}"
  local tarball="${2}"
  local layer="${3}"
  local expected="${4}"
  local test_data="${TEST_DATA_DIR}/${tarball}.tar"

  local metadata="$(tar xOf "${test_data}" "./${layer}/json")"

  # This would be much more accurate if we had 'jq' everywhere.
  EXPECT_CONTAINS "${metadata}" "\"${property}\": ${expected}"
}

function check_no_property() {
  local property="${1}"
  local tarball="${2}"
  local layer="${3}"
  local test_data="${TEST_DATA_DIR}/${tarball}.tar"

  tar xOf "${test_data}" "./${layer}/json" >$TEST_log
  expect_not_log "\"${property}\":"

  # notop variant
  test_data="${TEST_DATA_DIR}/notop_${tarball}.tar"
  tar xOf "${test_data}" "./${layer}/json" >$TEST_log
  expect_not_log "\"${property}\":"
}

function check_size() {
  check_property Size "${@}"
}

function check_id() {
  check_property id "${@}"
}

function check_parent() {
  check_property parent "${@}"
}

function check_entrypoint() {
  input="$1"
  shift
  check_property Entrypoint "${input}" "${@}"
  check_property Entrypoint "notop_${input}" "${@}"
}

function check_cmd() {
  input="$1"
  shift
  check_property Cmd "${input}" "${@}"
  check_property Cmd "notop_${input}" "${@}"
}

function check_ports() {
  input="$1"
  shift
  check_property ExposedPorts "${input}" "${@}"
  check_property ExposedPorts "${input}" "${@}"
}

function check_volumes() {
  input="$1"
  shift
  check_property Volumes "${input}" "${@}"
  check_property Volumes "notop_${input}" "${@}"
}

function check_env() {
  input="$1"
  shift
  check_property Env "${input}" "${@}"
  check_property Env "notop_${input}" "${@}"
}

function check_layers_aux() {
  local input=${1}
  shift 1
  local expected_layers=(${*})

  local expected_layers_sorted=(
    $(for i in ${expected_layers[*]}; do echo $i; done | sort)
  )
  local test_data="${TEST_DATA_DIR}/${input}.tar"

  # Verbose output for testing.
  tar tvf "${test_data}"

  local actual_layers=(
    $(tar tvf ${test_data} | tr -s ' ' | cut -d' ' -f 4- | sort \
      | cut -d'/' -f 2 | grep -E '^[0-9a-f]+$' | sort | uniq))

  # Verbose output for testing.
  echo Expected: ${expected_layers_sorted[@]}
  echo Actual: ${actual_layers[@]}

  check_eq "${#expected_layers[@]}" "${#actual_layers[@]}"

  local index=0
  local parent=
  while [ "${index}" -lt "${#expected_layers[@]}" ]
  do
    # Check that the nth sorted layer matches
    check_eq "${expected_layers_sorted[$index]}" "${actual_layers[$index]}"

    # Grab the ordered layer and check it.
    local layer="${expected_layers[$index]}"

    # Verbose output for testing.
    echo Checking layer: "${layer}"

    local listing="$(tar xOf "${test_data}" "./${layer}/layer.tar" | tar tv)"

    # Check that all files in the layer, if any, have the magic timestamp
    check_eq "$(echo "${listing}" | grep -Fv "${MAGIC_TIMESTAMP}")" ""

    check_id "${input}" "${layer}" "\"${layer}\""

    # Check that the layer contains its predecessor as its parent in the JSON.
    if [[ -n "${parent}" ]]; then
      check_parent "${input}" "${layer}" "\"${parent}\""
    fi

    # Check that the layer's size metadata matches the layer's tarball's size.
    local layer_size=$(tar xOf "${test_data}" "./${layer}/layer.tar" | wc -c | xargs)
    check_size "${input}" "${layer}" "${layer_size}"

    index=$((index + 1))
    parent=$layer
  done
}

function check_layers() {
  local input=$1
  shift
  check_layers_aux "$input" "$@"
  check_layers_aux "notop_$input" "$@"
}

function test_gen_image() {
  grep -Fsq "./gen.out" "$TEST_DATA_DIR/gen_image.tar" \
    || fail "'./gen.out' not found in '$TEST_DATA_DIR/gen_image.tar'"
}

function test_files_base() {
  check_layers "files_base" \
    "240dd12c02aee796394ce18eee3108475f7d544294b17fc90ec54e983601fe1b"
}

function test_files_with_files_base() {
  check_layers "files_with_files_base" \
    "240dd12c02aee796394ce18eee3108475f7d544294b17fc90ec54e983601fe1b" \
    "a9fd8cab2b9831ca2a13f371c04667a7698ef3baa90f3e820c4568d774cc69ab"
}

function test_tar_base() {
  check_layers "tar_base" \
    "8b9e4db9dd4b990ee6d8adc2843ad64702ad9063ae6c22e8ca5f94aa54e71277"

  # Check that this layer doesn't have any entrypoint data by looking
  # for *any* entrypoint.
  check_no_property "Entrypoint" "tar_base" \
    "8b9e4db9dd4b990ee6d8adc2843ad64702ad9063ae6c22e8ca5f94aa54e71277"
}

function test_tar_with_tar_base() {
  check_layers "tar_with_tar_base" \
    "8b9e4db9dd4b990ee6d8adc2843ad64702ad9063ae6c22e8ca5f94aa54e71277" \
    "1cc81a2aaec2e3727d98d48bf9ba09d3ac96ef48adf5edae861d15dd0191dc40"
}

function test_files_with_tar_base() {
  check_layers "files_with_tar_base" \
    "8b9e4db9dd4b990ee6d8adc2843ad64702ad9063ae6c22e8ca5f94aa54e71277" \
    "b0fe5685bf89a2c875a93495a4e2b7a3f1fb6f27a9ac5dc2b174e7d74cb6fe27"
}

function test_tar_with_files_base() {
  check_layers "tar_with_files_base" \
    "240dd12c02aee796394ce18eee3108475f7d544294b17fc90ec54e983601fe1b" \
    "c1ec3719c075f54f65045f127a6cfd77675c547446cdda7193de81b9b59afb51"
}

function test_base_with_entrypoint() {
  check_layers "base_with_entrypoint" \
    "b5f7dbb1097fa15ac31e86a149b8de2efeaedd5b8f008ba88cf2ae51adf1bd00"

  check_entrypoint "base_with_entrypoint" \
    "b5f7dbb1097fa15ac31e86a149b8de2efeaedd5b8f008ba88cf2ae51adf1bd00" \
    '["/bar"]'

  # Check that the base layer has a port exposed.
  check_ports "base_with_entrypoint" \
    "b5f7dbb1097fa15ac31e86a149b8de2efeaedd5b8f008ba88cf2ae51adf1bd00" \
    '{"8080/tcp": {}}'
}

function test_derivative_with_shadowed_cmd() {
  check_layers "derivative_with_shadowed_cmd" \
    "b5f7dbb1097fa15ac31e86a149b8de2efeaedd5b8f008ba88cf2ae51adf1bd00" \
    "f1303868df182b4bdbf3abeba580e4fe9fe9b976615dcaa67ea167d954f364d3"
}

function test_derivative_with_cmd() {
  check_layers "derivative_with_cmd" \
    "b5f7dbb1097fa15ac31e86a149b8de2efeaedd5b8f008ba88cf2ae51adf1bd00" \
    "f1303868df182b4bdbf3abeba580e4fe9fe9b976615dcaa67ea167d954f364d3" \
    "e050a01061b34d8703a672e0c6e573911ba1925fadfbcaf76999ace937108cee"

  check_entrypoint "derivative_with_cmd" \
    "e050a01061b34d8703a672e0c6e573911ba1925fadfbcaf76999ace937108cee" \
    '["/bar"]'

  # Check that the middle layer has our shadowed arg.
  check_cmd "derivative_with_cmd" \
    "f1303868df182b4bdbf3abeba580e4fe9fe9b976615dcaa67ea167d954f364d3" \
    '["shadowed-arg"]'

  # Check that our topmost layer excludes the shadowed arg.
  check_cmd "derivative_with_cmd" \
    "e050a01061b34d8703a672e0c6e573911ba1925fadfbcaf76999ace937108cee" \
    '["arg1", "arg2"]'

  # Check that the topmost layer has the ports exposed by the bottom
  # layer, and itself.
  check_ports "derivative_with_cmd" \
    "e050a01061b34d8703a672e0c6e573911ba1925fadfbcaf76999ace937108cee" \
    '{"80/tcp": {}, "8080/tcp": {}}'
}

function test_derivative_with_volume() {
  check_layers "derivative_with_volume" \
    "f86da639a9346bec6d3a821ad1f716a177a8ff8f71d66f8b70238ce7e7ba51b8" \
    "839bbd055b732c784847b3ec112d88c94f3bb752147987daef916bc956f9adf0"

  # Check that the topmost layer has the ports exposed by the bottom
  # layer, and itself.
  check_volumes "derivative_with_volume" \
    "f86da639a9346bec6d3a821ad1f716a177a8ff8f71d66f8b70238ce7e7ba51b8" \
    '{"/logs": {}}'

  check_volumes "derivative_with_volume" \
    "839bbd055b732c784847b3ec112d88c94f3bb752147987daef916bc956f9adf0" \
    '{"/asdf": {}, "/blah": {}, "/logs": {}}'
}

function test_generated_tarball() {
  check_layers "generated_tarball" \
    "54b8328604115255cc76c12a2a51939be65c40bf182ff5a898a5fb57c38f7772"
}

function test_with_env() {
  check_layers "with_env" \
    "f86da639a9346bec6d3a821ad1f716a177a8ff8f71d66f8b70238ce7e7ba51b8" \
    "80b94376a90de45256c3e94c82bc3812bc5cbd05b7d01947f29e6805e8cd7018"

  check_env "with_env" \
    "80b94376a90de45256c3e94c82bc3812bc5cbd05b7d01947f29e6805e8cd7018" \
    '["bar=blah blah blah", "foo=/asdf"]'
}

function test_with_double_env() {
  check_layers "with_double_env" \
    "f86da639a9346bec6d3a821ad1f716a177a8ff8f71d66f8b70238ce7e7ba51b8" \
    "80b94376a90de45256c3e94c82bc3812bc5cbd05b7d01947f29e6805e8cd7018" \
    "548e1d847a1d051e3cb3af383b0ebe40d341c01c97e735ae5a78ee3e10353b93"

  # Check both the aggregation and the expansion of embedded variables.
  check_env "with_double_env" \
    "548e1d847a1d051e3cb3af383b0ebe40d341c01c97e735ae5a78ee3e10353b93" \
    '["bar=blah blah blah", "baz=/asdf blah blah blah", "foo=/asdf"]'
}

function test_with_double_env() {
  check_layers "with_double_env" \
    "f86da639a9346bec6d3a821ad1f716a177a8ff8f71d66f8b70238ce7e7ba51b8" \
    "80b94376a90de45256c3e94c82bc3812bc5cbd05b7d01947f29e6805e8cd7018" \
    "548e1d847a1d051e3cb3af383b0ebe40d341c01c97e735ae5a78ee3e10353b93"

  # Check both the aggregation and the expansion of embedded variables.
  check_env "with_double_env" \
    "548e1d847a1d051e3cb3af383b0ebe40d341c01c97e735ae5a78ee3e10353b93" \
    '["bar=blah blah blah", "baz=/asdf blah blah blah", "foo=/asdf"]'
}

function get_layer_listing() {
  local input=$1
  local layer=$2
  local test_data="${TEST_DATA_DIR}/${input}.tar"
  tar xOf "${test_data}" \
    "./${layer}/layer.tar" | tar tv | sed -e 's/^.*:00 //'
}

function test_data_path() {
  local no_data_path_sha="29bab1d66ea34ebbb9604704322458e0f06a3a9a7b0ac08df8b95a6c9c3d9320"
  local data_path_sha="9a5c6b96227af212d1300964aaeb0723e78e78de019de1ba2382603f3050558a"

  check_layers_aux "no_data_path_image" "${no_data_path_sha}"
  check_layers_aux "data_path_image" "${data_path_sha}"

  # Without data_path = "." the file will be inserted as `./test`
  # (since it is the path in the package) and with data_path = "."
  # the file will be inserted relatively to the testdata package
  # (so `./test/test`).
  check_eq $(get_layer_listing "no_data_path_image" "${no_data_path_sha}") \
    ./test
  check_eq $(get_layer_listing "data_path_image" "${data_path_sha}") \
    ./test/test
}

run_suite "build_test"

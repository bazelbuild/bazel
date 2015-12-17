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

function check_label() {
  input="$1"
  shift
  check_property Label "${input}" "${@}"
  check_property Label "notop_${input}" "${@}"
}

function check_workdir() {
  input="$1"
  shift
  check_property WorkingDir "${input}" "${@}"
  check_property WorkingDir "notop_${input}" "${@}"
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

function test_dummy_repository() {
  local layer="0279f3ce8b08d10506abcf452393b3e48439f5eca41b836fae59a0d509fbafea"
  local test_data="${TEST_DATA_DIR}/dummy_repository.tar"
  check_layers_aux "dummy_repository" "$layer"


  local repositories="$(tar xOf "${test_data}" "./repositories")"
  # This would really need to use `jq` instead.
  echo "${repositories}" | \
    grep -Esq -- "\"gcr.io/dummy/[a-zA-Z_]*_docker_testdata\": {" \
    || fail "Cannot find image in repository gcr.io/dummy in '${repositories}'"
  EXPECT_CONTAINS "${repositories}" "\"dummy_repository\": \"$layer\""
}

function test_files_base() {
  check_layers "files_base" \
    "82ca3945f7d07df82f274d7fafe83fd664c2154e5c64c988916ccd5b217bb710"
}

function test_files_with_files_base() {
  check_layers "files_with_files_base" \
    "82ca3945f7d07df82f274d7fafe83fd664c2154e5c64c988916ccd5b217bb710" \
    "84c0d09919ae8b06cb6b064d8cd5eab63341a46f11ccc7ecbe270ad3e1f52744"
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

function test_directory_with_tar_base() {
  check_layers "directory_with_tar_base" \
    "8b9e4db9dd4b990ee6d8adc2843ad64702ad9063ae6c22e8ca5f94aa54e71277" \
    "e56ddeb8279698484f50d480f71cb5380223ad0f451766b7b9a9348129d02542"
}

function test_files_with_tar_base() {
  check_layers "files_with_tar_base" \
    "8b9e4db9dd4b990ee6d8adc2843ad64702ad9063ae6c22e8ca5f94aa54e71277" \
    "f099727fa58f9b688e77b511b3cc728b86ae0e84d197b9330bd51082ad5589f2"
}

function test_workdir_with_tar_base() {
  check_layers "workdir_with_tar_base" \
    "8b9e4db9dd4b990ee6d8adc2843ad64702ad9063ae6c22e8ca5f94aa54e71277" \
    "f24cbe53bd1b78909c6dba0bd47016354f3488b35b85aeee68ecc423062b927e"
}

function test_tar_with_files_base() {
  check_layers "tar_with_files_base" \
    "82ca3945f7d07df82f274d7fafe83fd664c2154e5c64c988916ccd5b217bb710" \
    "bee1a325e4b51a1dcfd7e447987b4e130590815865ab22e8744878053d525f20"
}

function test_base_with_entrypoint() {
  check_layers "base_with_entrypoint" \
    "4acbeb0495918726c0107e372b421e1d2a6fd4825d58fc3f0b0b2a719fb3ce1b"

  check_entrypoint "base_with_entrypoint" \
    "4acbeb0495918726c0107e372b421e1d2a6fd4825d58fc3f0b0b2a719fb3ce1b" \
    '["/bar"]'

  # Check that the base layer has a port exposed.
  check_ports "base_with_entrypoint" \
    "4acbeb0495918726c0107e372b421e1d2a6fd4825d58fc3f0b0b2a719fb3ce1b" \
    '{"8080/tcp": {}}'
}

function test_derivative_with_shadowed_cmd() {
  check_layers "derivative_with_shadowed_cmd" \
    "4acbeb0495918726c0107e372b421e1d2a6fd4825d58fc3f0b0b2a719fb3ce1b" \
    "e35f57dc6c1e84ae67dcaaf3479a3a3c0f52ac4d194073bd6214e04c05beab42"
}

function test_derivative_with_cmd() {
  check_layers "derivative_with_cmd" \
    "4acbeb0495918726c0107e372b421e1d2a6fd4825d58fc3f0b0b2a719fb3ce1b" \
    "e35f57dc6c1e84ae67dcaaf3479a3a3c0f52ac4d194073bd6214e04c05beab42" \
    "186289545131e34510006ac79498078dcf41736a5eb9a36920a6b30d3f45bc01"

  check_entrypoint "derivative_with_cmd" \
    "186289545131e34510006ac79498078dcf41736a5eb9a36920a6b30d3f45bc01" \
    '["/bar"]'

  # Check that the middle layer has our shadowed arg.
  check_cmd "derivative_with_cmd" \
    "e35f57dc6c1e84ae67dcaaf3479a3a3c0f52ac4d194073bd6214e04c05beab42" \
    '["shadowed-arg"]'

  # Check that our topmost layer excludes the shadowed arg.
  check_cmd "derivative_with_cmd" \
    "186289545131e34510006ac79498078dcf41736a5eb9a36920a6b30d3f45bc01" \
    '["arg1", "arg2"]'

  # Check that the topmost layer has the ports exposed by the bottom
  # layer, and itself.
  check_ports "derivative_with_cmd" \
    "186289545131e34510006ac79498078dcf41736a5eb9a36920a6b30d3f45bc01" \
    '{"80/tcp": {}, "8080/tcp": {}}'
}

function test_derivative_with_volume() {
  check_layers "derivative_with_volume" \
    "125e7cfb9d4a6d803a57b88bcdb05d9a6a47ac0d6312a8b4cff52a2685c5c858" \
    "08424283ad3a7e020e210bec22b166d7ebba57f7ba2d0713c2fd7bd1e2038f88"

  # Check that the topmost layer has the ports exposed by the bottom
  # layer, and itself.
  check_volumes "derivative_with_volume" \
    "125e7cfb9d4a6d803a57b88bcdb05d9a6a47ac0d6312a8b4cff52a2685c5c858" \
    '{"/logs": {}}'

  check_volumes "derivative_with_volume" \
    "08424283ad3a7e020e210bec22b166d7ebba57f7ba2d0713c2fd7bd1e2038f88" \
    '{"/asdf": {}, "/blah": {}, "/logs": {}}'
}

function test_generated_tarball() {
  check_layers "generated_tarball" \
    "54b8328604115255cc76c12a2a51939be65c40bf182ff5a898a5fb57c38f7772"
}

function test_with_env() {
  check_layers "with_env" \
    "125e7cfb9d4a6d803a57b88bcdb05d9a6a47ac0d6312a8b4cff52a2685c5c858" \
    "42a1bd0f449f61a23b8a7776875ffb6707b34ee99c87d6428a7394f5e55e8624"

  check_env "with_env" \
    "42a1bd0f449f61a23b8a7776875ffb6707b34ee99c87d6428a7394f5e55e8624" \
    '["bar=blah blah blah", "foo=/asdf"]'
}

function test_with_double_env() {
  check_layers "with_double_env" \
    "125e7cfb9d4a6d803a57b88bcdb05d9a6a47ac0d6312a8b4cff52a2685c5c858" \
    "42a1bd0f449f61a23b8a7776875ffb6707b34ee99c87d6428a7394f5e55e8624" \
    "576a9fd9c690be04dc7aacbb9dbd1f14816e32dbbcc510f4d42325bbff7163dd"

  # Check both the aggregation and the expansion of embedded variables.
  check_env "with_double_env" \
    "576a9fd9c690be04dc7aacbb9dbd1f14816e32dbbcc510f4d42325bbff7163dd" \
    '["bar=blah blah blah", "baz=/asdf blah blah blah", "foo=/asdf"]'
}

function test_with_label() {
  check_layers "with_label" \
    "125e7cfb9d4a6d803a57b88bcdb05d9a6a47ac0d6312a8b4cff52a2685c5c858" \
    "eba6abda3d259ab6ed5f4d48b76df72a5193fad894d4ae78fbf0a363d8f9e8fd"

  check_label "with_label" \
    "eba6abda3d259ab6ed5f4d48b76df72a5193fad894d4ae78fbf0a363d8f9e8fd" \
    '["com.example.bar={\"name\": \"blah\"}", "com.example.baz=qux", "com.example.foo={\"name\": \"blah\"}"]'
}

function test_with_double_label() {
  check_layers "with_double_label" \
    "125e7cfb9d4a6d803a57b88bcdb05d9a6a47ac0d6312a8b4cff52a2685c5c858" \
    "eba6abda3d259ab6ed5f4d48b76df72a5193fad894d4ae78fbf0a363d8f9e8fd" \
    "bfe88fbb5e24fc5bff138f7a1923d53a2ee1bbc8e54b6f5d9c371d5f48b6b023" \

  check_label "with_double_label" \
    "bfe88fbb5e24fc5bff138f7a1923d53a2ee1bbc8e54b6f5d9c371d5f48b6b023" \
    '["com.example.bar={\"name\": \"blah\"}", "com.example.baz=qux", "com.example.foo={\"name\": \"blah\"}", "com.example.qux={\"name\": \"blah-blah\"}"]'
}

function get_layer_listing() {
  local input=$1
  local layer=$2
  local test_data="${TEST_DATA_DIR}/${input}.tar"
  tar xOf "${test_data}" \
    "./${layer}/layer.tar" | tar tv | sed -e 's/^.*:00 //'
}

function test_data_path() {
  local no_data_path_sha="451d182e5c71840f00ba9726dc0239db73a21b7e89e79c77f677e3f7c5c23d44"
  local data_path_sha="9a41c9e1709558f7ef06f28f66e9056feafa7e0f83990801e1b27c987278d8e8"
  local absolute_data_path_sha="f196c42ab4f3eb850d9655b950b824db2c99c01527703ac486a7b48bb2a34f44"
  local root_data_path_sha="19d7fd26d67bfaeedd6232dcd441f14ee163bc81c56ed565cc20e73311c418b6"

  check_layers_aux "no_data_path_image" "${no_data_path_sha}"
  check_layers_aux "data_path_image" "${data_path_sha}"
  check_layers_aux "absolute_data_path_image" "${absolute_data_path_sha}"
  check_layers_aux "root_data_path_image" "${root_data_path_sha}"

  # Without data_path = "." the file will be inserted as `./test`
  # (since it is the path in the package) and with data_path = "."
  # the file will be inserted relatively to the testdata package
  # (so `./test/test`).
  check_eq "$(get_layer_listing "no_data_path_image" "${no_data_path_sha}")" \
    './
./test'
  check_eq "$(get_layer_listing "data_path_image" "${data_path_sha}")" \
    './
./test/
./test/test'

  # With an absolute path for data_path, we should strip that prefix
  # from the files' paths. Since the testdata images are in
  # //tools/build_defs/docker/testdata and data_path is set to
  # "/tools/build_defs", we should have `docker` as the top-level
  # directory.
  check_eq "$(get_layer_listing "absolute_data_path_image" "${absolute_data_path_sha}")" \
    './
./docker/
./docker/testdata/
./docker/testdata/test/
./docker/testdata/test/test'

  # With data_path = "/", we expect the entire path from the repository
  # root.
  check_eq "$(get_layer_listing "root_data_path_image" "${root_data_path_sha}")" \
    "./
./tools/
./tools/build_defs/
./tools/build_defs/docker/
./tools/build_defs/docker/testdata/
./tools/build_defs/docker/testdata/test/
./tools/build_defs/docker/testdata/test/test"
}

function test_extras_with_deb() {
  local test_data="${TEST_DATA_DIR}/extras_with_deb.tar"
  local sha=$(tar xOf ${test_data} ./top)

  # The content of the layer should have no duplicate
  local layer_listing="$(get_layer_listing "extras_with_deb" "${sha}" | sort)"
  check_eq "${layer_listing}" \
"./
./etc/
./etc/nsswitch.conf
./tmp/
./usr/
./usr/bin/
./usr/bin/java -> /path/to/bin/java
./usr/titi"
}

run_suite "build_test"

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
  local image="${3}"
  local expected="${4}"
  local test_data="${TEST_DATA_DIR}/${tarball}.tar"

  local config="$(tar xOf "${test_data}" "./${image}.json")"

  # This would be much more accurate if we had 'jq' everywhere.
  EXPECT_CONTAINS "${config}" "\"${property}\": ${expected}"
}

function check_no_property() {
  local property="${1}"
  local tarball="${2}"
  local image="${3}"
  local test_data="${TEST_DATA_DIR}/${tarball}.tar"

  tar xOf "${test_data}" "./${image}.json" >$TEST_log
  expect_not_log "\"${property}\":"
}

function check_entrypoint() {
  input="$1"
  shift
  check_property Entrypoint "${input}" "${@}"
}

function check_cmd() {
  input="$1"
  shift
  check_property Cmd "${input}" "${@}"
}

function check_ports() {
  input="$1"
  shift
  check_property ExposedPorts "${input}" "${@}"
}

function check_volumes() {
  input="$1"
  shift
  check_property Volumes "${input}" "${@}"
}

function check_env() {
  input="$1"
  shift
  check_property Env "${input}" "${@}"
}

function check_label() {
  input="$1"
  shift
  check_property Label "${input}" "${@}"
}

function check_workdir() {
  input="$1"
  shift
  check_property WorkingDir "${input}" "${@}"
}

function check_user() {
  input="$1"
  shift
  check_property User "${input}" "${@}"
}

function check_images() {
  local input="$1"
  shift 1
  local expected_images=(${*})
  local test_data="${TEST_DATA_DIR}/${input}.tar"

  local manifest="$(tar xOf "${test_data}" "./manifest.json")"
  local manifest_images=(
    $(echo "${manifest}" | grep -Eo '"Config":[[:space:]]*"[^"]+"' \
      | sed -r -e 's#"Config":.*?"([0-9a-f]+)\.json"#\1#'))

  local manifest_parents=(
    $(echo "${manifest}" | grep -Eo '"Parent":[[:space:]]*"[^"]+"' \
      | sed -r -e 's#"Parent":.*?"sha256:([0-9a-f]+)"#\1#'))

  # Verbose output for testing.
  echo Expected: "${expected_images[@]}"
  echo Actual: "${manifest_images[@]}"
  echo Parents: "${manifest_parents[@]}"

  check_eq "${#expected_images[@]}" "${#manifest_images[@]}"

  local index=0
  while [ "${index}" -lt "${#expected_images[@]}" ]
  do
    # Check that the nth sorted layer matches
    check_eq "${expected_images[$index]}" "${manifest_images[$index]}"

    index=$((index + 1))
  done

  # Check that the image contains its predecessor as its parent in the manifest.
  check_eq "${#manifest_parents[@]}" "$((${#manifest_images[@]} - 1))"

  local index=0
  while [ "${index}" -lt "${#manifest_parents[@]}" ]
  do
    # Check that the nth sorted layer matches
    check_eq "${manifest_parents[$index]}" "${manifest_images[$index]}"

    index=$((index + 1))
  done
}

# The bottom manifest entry must contain all layers in order
function check_image_manifest_layers() {
  local input="$1"
  shift 1
  local expected_layers=(${*})
  local test_data="${TEST_DATA_DIR}/${input}.tar"

  local manifest="$(tar xOf "${test_data}" "./manifest.json")"
  local manifest_layers=(
    $(echo "${manifest}" | grep -Eo '"Layers":[[:space:]]*\[[^]]+\]' \
      | grep -Eo '\[.+\]' | tail -n 1 | tr ',' '\n' \
      | sed -r -e 's#.*"([0-9a-f]+)/layer\.tar".*#\1#'))

  # Verbose output for testing.
  echo Expected: "${expected_layers[@]}"
  echo Actual: "${manifest_layers[@]}"

  check_eq "${#expected_layers[@]}" "${#manifest_layers[@]}"

  local index=0
  while [ "${index}" -lt "${#expected_layers[@]}" ]
  do
    # Check that the nth sorted layer matches
    check_eq "${expected_layers[$index]}" "${manifest_layers[$index]}"

    index=$((index + 1))
  done
}

function check_layers_aux() {
  local input="$1"
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
  echo Expected: "${expected_layers_sorted[@]}"
  echo Actual: "${actual_layers[@]}"

  check_eq "${#expected_layers[@]}" "${#actual_layers[@]}"

  local index=0
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
    check_eq "$(echo "${listing}" | grep -Fv "${MAGIC_TIMESTAMP}" || true)" ""

    index=$((index + 1))
  done
}

function check_layers() {
  local input="$1"
  shift
  check_layers_aux "$input" "$@"
  check_image_manifest_layers "$input" "$@"
}

function test_gen_image() {
  grep -Fsq "./gen.out" "$TEST_DATA_DIR/gen_image.tar" \
    || fail "'./gen.out' not found in '$TEST_DATA_DIR/gen_image.tar'"
}

function test_dummy_repository() {
  local layer="0279f3ce8b08d10506abcf452393b3e48439f5eca41b836fae59a0d509fbafea"
  local test_data="${TEST_DATA_DIR}/dummy_repository.tar"
  check_layers_aux "dummy_repository" "$layer"
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
    "9fec194fd32c03350d6a6e60ee8ed7862471e8817aaa310306d9be6242b05d20"
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
    "d59ab78d94f88b906227b8696d3065b91c71a1c6045d5103f3572c1e6fe9a1a9" \
    '["/bar"]'

  # Check that the base layer has a port exposed.
  check_ports "base_with_entrypoint" \
    "d59ab78d94f88b906227b8696d3065b91c71a1c6045d5103f3572c1e6fe9a1a9" \
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

  check_images "derivative_with_cmd" \
    "d59ab78d94f88b906227b8696d3065b91c71a1c6045d5103f3572c1e6fe9a1a9" \
    "a37fcc5dfa513987ecec8a19ebe5d17568a7d6e696771c596b110fcc30a2d8a6" \
    "d3ea6e7cfc3e182a8ca43081db1e145f1bee8c5da5627639800c76abf61b5165"

  check_entrypoint "derivative_with_cmd" \
    "d59ab78d94f88b906227b8696d3065b91c71a1c6045d5103f3572c1e6fe9a1a9" \
    '["/bar"]'

  # Check that the middle image has our shadowed arg.
  check_cmd "derivative_with_cmd" \
    "a37fcc5dfa513987ecec8a19ebe5d17568a7d6e696771c596b110fcc30a2d8a6" \
    '["shadowed-arg"]'

  # Check that our topmost image excludes the shadowed arg.
  check_cmd "derivative_with_cmd" \
    "d3ea6e7cfc3e182a8ca43081db1e145f1bee8c5da5627639800c76abf61b5165" \
    '["arg1", "arg2"]'

  # Check that the topmost layer has the ports exposed by the bottom
  # layer, and itself.
  check_ports "derivative_with_cmd" \
    "d3ea6e7cfc3e182a8ca43081db1e145f1bee8c5da5627639800c76abf61b5165" \
    '{"80/tcp": {}, "8080/tcp": {}}'
}

function test_derivative_with_volume() {
  check_layers "derivative_with_volume" \
    "125e7cfb9d4a6d803a57b88bcdb05d9a6a47ac0d6312a8b4cff52a2685c5c858" \
    "08424283ad3a7e020e210bec22b166d7ebba57f7ba2d0713c2fd7bd1e2038f88"

  check_images "derivative_with_volume" \
    "da0f0e314eb3187877754fd5ee1e487b93c13dbabdba18f35d130324f3c9b76d" \
    "c872bf3f4c7eb5a01ae7ad6fae4c25e86ff2923bb1fe29be5edcdff1b31ed71a"

  # Check that the topmost layer has the ports exposed by the bottom
  # layer, and itself.
  check_volumes "derivative_with_volume" \
    "da0f0e314eb3187877754fd5ee1e487b93c13dbabdba18f35d130324f3c9b76d" \
    '{"/logs": {}}'

  check_volumes "derivative_with_volume" \
    "c872bf3f4c7eb5a01ae7ad6fae4c25e86ff2923bb1fe29be5edcdff1b31ed71a" \
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
    "87c0d91841f92847ec6c183810f720e5926dba0652eb5d52a807366825dd21c7" \
    '["bar=blah blah blah", "foo=/asdf"]'
}

function test_with_double_env() {
  check_layers "with_double_env" \
    "125e7cfb9d4a6d803a57b88bcdb05d9a6a47ac0d6312a8b4cff52a2685c5c858" \
    "42a1bd0f449f61a23b8a7776875ffb6707b34ee99c87d6428a7394f5e55e8624" \
    "576a9fd9c690be04dc7aacbb9dbd1f14816e32dbbcc510f4d42325bbff7163dd"

  # Check both the aggregation and the expansion of embedded variables.
  check_env "with_double_env" \
    "273d2a6cfc25001baf9d3f7c68770ec79a1671b8249d153e7611a4f80165ecda" \
    '["bar=blah blah blah", "baz=/asdf blah blah blah", "foo=/asdf"]'
}

function test_with_label() {
  check_layers "with_label" \
    "125e7cfb9d4a6d803a57b88bcdb05d9a6a47ac0d6312a8b4cff52a2685c5c858" \
    "eba6abda3d259ab6ed5f4d48b76df72a5193fad894d4ae78fbf0a363d8f9e8fd"

  check_label "with_label" \
    "83c007425faff33ac421329af9f6444b7250abfc12c28f188b47e97fb715c006" \
    '["com.example.bar={\"name\": \"blah\"}", "com.example.baz=qux", "com.example.foo={\"name\": \"blah\"}"]'
}

function test_with_double_label() {
  check_layers "with_double_label" \
    "125e7cfb9d4a6d803a57b88bcdb05d9a6a47ac0d6312a8b4cff52a2685c5c858" \
    "eba6abda3d259ab6ed5f4d48b76df72a5193fad894d4ae78fbf0a363d8f9e8fd" \
    "bfe88fbb5e24fc5bff138f7a1923d53a2ee1bbc8e54b6f5d9c371d5f48b6b023"

  check_label "with_double_label" \
    "8cfc89c83adf947cd2c18c11579559f1f48cf375a20364ec79eb14d6580dbf75" \
    '["com.example.bar={\"name\": \"blah\"}", "com.example.baz=qux", "com.example.foo={\"name\": \"blah\"}", "com.example.qux={\"name\": \"blah-blah\"}"]'
}

function test_with_user() {
  check_user "with_user" \
    "bd6666bdde7d4a837a0685d2861822507119f7f6e565acecbbbe93f1d0cc1974" \
    "\"nobody\""
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

run_suite "build_test_oci"

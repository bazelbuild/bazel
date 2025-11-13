#!/bin/sh
#
# Copyright 2017 The Bazel Authors. All rights reserved.
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

#
# Generates the Bash completion script for Bazel.
#
# At a minimum, you must pass the --bazel and --output flags to specify the path
# to the Bazel binary to use and the output file to generate.
#
# Callers can customize the completion script by passing additional files with
# the --prepend and --append flags, which are stitched together to generate the
# final completion script.  Prepended files can override built-in variables and
# appended files can override built-in functions.
#

set -e

die() {
  echo "${@}" 1>&2
  exit 1
}

get_optarg() {
  expr -- "${1}" : "[^=]*=\\(.*\\)"
}

append=
bazel=
javabase=
output=
prepend=
while [ ${#} -gt 0 ]; do
  case "${1}" in
    --append=*) append="${append} $(get_optarg "${1}")" ;;
    --bazel=*) bazel="$(get_optarg "${1}")" ;;
    --javabase=*) javabase="$(get_optarg "${1}")" ;;
    --output=*) output="$(get_optarg "${1}")" ;;
    --prepend=*) prepend="${prepend} $(get_optarg "${1}")" ;;
    --*) die "Unknown option ${1}" ;;
    *) break ;;
  esac
  shift
done
[ ${#} -eq 0 ] || die "No arguments allowed"
[ -n "${bazel}" ] || die "--bazel required but not provided"
[ -n "${output}" ] || die "--output required but not provided"

tempdir="$(mktemp -d "${TMPDIR:-/tmp}/generate_bash_completion.XXXXXXXX")"
trap "rm -rf '${tempdir}'" EXIT

touch "${tempdir}/WORKSPACE"
mkdir "${tempdir}/root"

[ -z "${prepend}" ] || cat ${prepend} >>"${tempdir}/output"

server_javabase_flag=
[ -z "${javabase}" ] || server_javabase_flag="--server_javabase=${javabase}"
"${bazel}" --batch --output_user_root="${tempdir}/root" ${server_javabase_flag} \
    help completion >>"${tempdir}/output"

[ -z "${append}" ] || cat ${append} >>"${tempdir}/output"

rm -f "${output}"
mv "${tempdir}/output" "${output}"

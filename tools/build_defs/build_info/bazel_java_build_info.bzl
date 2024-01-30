# Copyright 2023 The Bazel Authors. All rights reserved.
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

"""A rule for getting transliterated build info files for Java."""

def _transform_date_string(date, timestamp):
    # Date should always be in "yyyy MMM d HH mm ss EEE" format.
    # For example: 2023 Jul 26 01 58 54 Fri
    tokens = date.split(" ")
    if len(tokens) != 7:
        fail("date string does not have a proper format: " + date + "\nExpected: yyyy MMM d HH mm ss EEE(2023 Jul 26 01 58 54 Fri)")
    date_format = "{day} {month} {day_num} {hour}:{minute}:{second} {year} ({timestamp})"
    return date_format.format(
        day = tokens[6],
        month = tokens[1],
        day_num = tokens[2],
        hour = tokens[3],
        minute = tokens[4],
        second = tokens[5],
        year = tokens[0],
        timestamp = timestamp,
    )

def _add_backslashes_to_string(value):
    tokens = value.split(":")
    return "\\:".join(tokens)

def _add_backslashes_to_dict_values(substitutions):
    # We are doing this because the values of substitutions dict
    # is written to .properties file for Java.
    for k, v in substitutions.items():
        substitutions[k] = _add_backslashes_to_string(v)
    return substitutions

def _transform_version(version_file_contents):
    # We assume that FORMATTED_DATE and BUILD_TIMESTAMP are always present
    # in the workspace status file.
    if "BUILD_TIMESTAMP" not in version_file_contents.keys() or "FORMATTED_DATE" not in version_file_contents.keys():
        fail("timestamp information is missing from workspace status file, BUILD_TIMESTAMP or FORMATTED_DATE keys not found.")
    substitutions = {
        "{build.time}": _transform_date_string(version_file_contents["FORMATTED_DATE"], version_file_contents["BUILD_TIMESTAMP"]),
        "{build.timestamp}": version_file_contents["BUILD_TIMESTAMP"],
        "{build.timestamp.as.int}": version_file_contents["BUILD_TIMESTAMP"],
    }
    return _add_backslashes_to_dict_values(substitutions)

def _transform_info(info_file_contents):
    return _add_backslashes_to_dict_values({"{build.label}": info_file_contents.get("BUILD_EMBED_LABEL", "")})

def _get_build_info_files(actions, version_template, info_template, redacted_file, stamp):
    outputs = []
    if stamp:
        version_file = actions.transform_version_file(transform_func = _transform_version, template = version_template, output_file_name = "volatile_file.properties")
        info_file = actions.transform_info_file(transform_func = _transform_info, template = info_template, output_file_name = "non_volatile_file.properties")
        outputs.append(info_file)
        outputs.append(version_file)
    else:
        output_redacted_file = actions.declare_file("redacted_file.properties")
        actions.symlink(output = output_redacted_file, target_file = redacted_file)
        outputs.append(output_redacted_file)
    return outputs

def _impl(ctx):
    output_groups = {
        "non_redacted_build_info_files": depset(_get_build_info_files(
            ctx.actions,
            ctx.file._version_template,
            ctx.file._info_template,
            ctx.file._redacted_file,
            True,
        )),
        "redacted_build_info_files": depset(_get_build_info_files(
            ctx.actions,
            ctx.file._version_template,
            ctx.file._info_template,
            ctx.file._redacted_file,
            False,
        )),
    }
    return OutputGroupInfo(**output_groups)

bazel_java_build_info = rule(
    implementation = _impl,
    attrs = {
        "_version_template": attr.label(
            default = "@bazel_tools//tools/build_defs/build_info/templates:volatile_file.properties.template",
            allow_single_file = True,
        ),
        "_info_template": attr.label(
            default = "@bazel_tools//tools/build_defs/build_info/templates:non_volatile_file.properties.template",
            allow_single_file = True,
        ),
        "_redacted_file": attr.label(
            default = "@bazel_tools//tools/build_defs/build_info/templates:redacted_file.properties.template",
            allow_single_file = True,
        ),
    },
)

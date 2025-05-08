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

"""A rule for getting transliterated build info files for C++."""

def _transform_version(version_file_contents):
    if "BUILD_TIMESTAMP" not in version_file_contents.keys():
        fail("timestamp information is missing from workspace status file, BUILD_TIMESTAMP key not found.")
    return {
        "{BUILD_SCM_REVISION}": version_file_contents.get("BUILD_SCM_REVISION", "0"),
        "{BUILD_SCM_STATUS}": version_file_contents.get("BUILD_SCM_STATUS", ""),
        "{BUILD_TIMESTAMP}": version_file_contents["BUILD_TIMESTAMP"],
    }

def _transform_info(info_file_contents):
    return {
        "{BUILD_EMBED_LABEL}": info_file_contents.get("BUILD_EMBED_LABEL", ""),
        "{BUILD_HOST}": info_file_contents.get("BUILD_HOST", "hostname"),
        "{BUILD_USER}": info_file_contents.get("BUILD_USER", "username"),
    }

def _get_build_info_files(actions, version_template, info_template, redacted_file, stamp):
    outputs = []
    if stamp:
        version_file = actions.transform_version_file(transform_func = _transform_version, template = version_template, output_file_name = "volatile_file.h")
        info_file = actions.transform_info_file(transform_func = _transform_info, template = info_template, output_file_name = "non_volatile_file.h")
        outputs.append(info_file)
        outputs.append(version_file)
    else:
        output_redacted_file = actions.declare_file("redacted_file.h")
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

bazel_cc_build_info = rule(
    implementation = _impl,
    attrs = {
        "_version_template": attr.label(
            default = "@bazel_tools//tools/build_defs/build_info/templates:volatile_file.h.template",
            allow_single_file = True,
        ),
        "_info_template": attr.label(
            default = "@bazel_tools//tools/build_defs/build_info/templates:non_volatile_file.h.template",
            allow_single_file = True,
        ),
        "_redacted_file": attr.label(
            default = "@bazel_tools//tools/build_defs/build_info/templates:redacted_file.h.template",
            allow_single_file = True,
        ),
    },
)

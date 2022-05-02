# Copyright 2021 The Bazel Authors. All rights reserved.
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

"""cc_proto_library rule."""

load(":common/cc/semantics.bzl", "semantics")

cc_common = _builtins.toplevel.cc_common
CcInfo = _builtins.toplevel.CcInfo
ProtoInfo = _builtins.toplevel.ProtoInfo

def _rule_impl(ctx):
    if len(ctx.attr.deps) == 0:
        fail("no deps attribute found; expected one.")

    if len(ctx.attr.deps) > 1:
        # actually label_list is used only for consistency with other deps attributes.
        fail("more than one deps attribute found; expected only one.")
    dep = ctx.attr.deps[0]

    files = semantics.get_proto_cc_files(dep)

    cc_files = [f for f in files if f.basename.endswith("pb.cc") or
                                    f.basename.endswith("pb.h") or f.basename.endswith("proto.h")]

    cc_files_provider = semantics.get_cc_files_provider(cc_files)
    files_provider = DefaultInfo(files = depset(cc_files))

    if CcInfo in dep:
        cc_info_provider = dep[CcInfo]
    else:
        fail("proto_library rule must generate CcInfo (have cc_api_version > 0).")

    return [cc_files_provider, files_provider, cc_info_provider]

def _create_cc_proto_library_rule():
    aspects = semantics.get_proto_aspects()
    return rule(
        output_to_genfiles = True,
        implementation = _rule_impl,
        attrs = {
            "deps": attr.label_list(
                # aspects = [_cc_proto_aspect], todo(dbabkin): return aspet after fix b/123988498
                # TODO(cmita): use Starlark aspect after b/145508948, or when proto_library
                # doesn't need to propagate
                aspects = aspects,
                allow_rules = ["proto_library"],
                providers = [ProtoInfo, CcInfo],  # todo(dbabkin): remove CcInfo after fix b/123988498
            ),
        },
    )

cc_proto_library = _create_cc_proto_library_rule()

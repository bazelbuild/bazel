# Copyright 2016 The Bazel Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http:#www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

_kind_to_kind_id = {
  "android_binary"  : 0,
  "android_library" : 1,
  "android_test" :    2,
  "android_roboelectric_test" : 3,
  "java_library" : 4,
  "java_test" : 5,
  "java_import" : 6,
  "java_binary" : 7,
  "proto_library" : 8,
  "android_sdk" : 9,
  "java_plugin" : 10,
}

_unrecognized_rule = -1;

def get_kind(target, ctx):
  return _kind_to_kind_id.get(ctx.rule.kind, _unrecognized_rule)

def _aspect_impl(target, ctx):
  ide_info_text = set()
  kind = get_kind(target, ctx)
  if kind != _unrecognized_rule:
      info = struct(
          label = str(target.label),
          kind = kind,
          # build_file = ???
      )
      output = ctx.new_file(target.label.name + ".aswb-build.txt")
      ctx.file_action(output, info.to_proto())
      ide_info_text += set([output])
  for dep in ctx.rule.attr.deps:
    ide_info_text += dep.android_studio_info_files

  return struct(
      output_groups = {
        "ide-info-text" : ide_info_text
      },
      android_studio_info_files = ide_info_text
    )

intellij_info_aspect = aspect(implementation = _aspect_impl,
    attr_aspects = [
          "deps",
          "exports",
          "_robolectric", # From android_robolectric_test
          "_junit", # From android_robolectric_test
          "binary_under_test", #  From android_test
          "java_lib",# From proto_library
          "_proto1_java_lib", # From proto_library
    ]
)
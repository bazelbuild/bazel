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

DEPENDENCY_ATTRIBUTES = [
  "deps",
  "exports",
  "_robolectric", # From android_robolectric_test
  "_junit", # From android_robolectric_test
  "binary_under_test", #  From android_test
  "java_lib",# From proto_library
  "_proto1_java_lib", # From proto_library
  "runtime_deps",
]

def get_kind(target, ctx):
  return _kind_to_kind_id.get(ctx.rule.kind, _unrecognized_rule)

def is_java_rule(target, ctx):
  return ctx.rule.kind != "android_sdk";

def struct_omit_none(**kwargs):
    d = {name: kwargs[name] for name in kwargs if kwargs[name] != None}
    return struct(**d)

def artifact_location(file):
  if file == None:
    return None
  return struct_omit_none(
      relative_path = file.short_path,
      is_source = file.is_source,
      root_execution_path_fragment = file.root.path if not file.is_source else None
  )

def build_file_artifact_location(build_file_path):
  return struct(
      relative_path = build_file_path,
      is_source = True,
  )


def library_artifact(java_output):
  if java_output == None or java_output.class_jar == None:
    return None
  return struct_omit_none(
        jar = artifact_location(java_output.class_jar),
        interface_jar = artifact_location(java_output.ijar),
        source_jar = artifact_location(java_output.source_jar),
  )

def annotation_processing_jars(annotation_processing):
  return struct_omit_none(
        jar = artifact_location(annotation_processing.class_jar),
        source_jar = artifact_location(annotation_processing.source_jar),
  )

def java_rule_ide_info(target, ctx):
  if hasattr(ctx.rule.attr, "srcs"):
     sources = [artifact_location(file)
                for src in ctx.rule.attr.srcs
                for file in src.files]
  else:
     sources = []

  jars = [library_artifact(output) for output in target.java.outputs.jars]
  ide_resolve_files = set([jar
       for output in target.java.outputs.jars
       for jar in [output.class_jar, output.ijar, output.source_jar]
       if jar != None and not jar.is_source])

  gen_jars = []
  if target.java.annotation_processing and target.java.annotation_processing.enabled:
    gen_jars = [annotation_processing_jars(target.java.annotation_processing)]
    ide_resolve_files = ide_resolve_files | set([ jar
        for jar in [target.java.annotation_processing.class_jar,
                    target.java.annotation_processing.source_jar]
        if jar != None and not jar.is_source])

  jdeps = artifact_location(target.java.outputs.jdeps)

  return (struct_omit_none(
                 sources = sources,
                 jars = jars,
                 jdeps = jdeps,
                 generated_jars = gen_jars
          ),
          ide_resolve_files)


def _aspect_impl(target, ctx):
  kind = get_kind(target, ctx)
  rule_attrs = ctx.rule.attr

  ide_info_text = set()
  ide_resolve_files = set()
  all_deps = []

  for attr_name in DEPENDENCY_ATTRIBUTES:
    if hasattr(rule_attrs, attr_name):
      deps = getattr(rule_attrs, attr_name)
      for dep in deps:
        ide_info_text = ide_info_text | dep.intellij_info_files.ide_info_text
        ide_resolve_files = ide_resolve_files | dep.intellij_info_files.ide_resolve_files
      all_deps += [str(dep.label) for dep in deps]

  if kind != _unrecognized_rule:
    if is_java_rule(target, ctx):
      java_rule_ide_info, java_ide_resolve_files = java_rule_ide_info(target, ctx)
      ide_resolve_files = ide_resolve_files | java_ide_resolve_files
      info = struct(
          label = str(target.label),
          kind = kind,
          dependencies = all_deps,
          build_file_artifact_location = build_file_artifact_location(ctx.build_file_path),
          java_rule_ide_info = java_rule_ide_info,
          tags = ctx.rule.attr.tags,
      )
    else:
      info = struct(
          label = str(target.label),
          kind = kind,
          dependencies = all_deps,
          build_file_artifact_location = build_file_artifact_location(ctx.build_file_path),
      )
  output = ctx.new_file(target.label.name + ".aswb-build.txt")
  ctx.file_action(output, info.to_proto())
  ide_info_text += set([output])

  return struct(
      output_groups = {
        "ide-info-text" : ide_info_text,
        "ide-resolve" : ide_resolve_files,
      },
      intellij_info_files = struct(
        ide_info_text = ide_info_text,
        ide_resolve_files = ide_resolve_files,
      )
    )

intellij_info_aspect = aspect(implementation = _aspect_impl,
    attr_aspects = DEPENDENCY_ATTRIBUTES
)
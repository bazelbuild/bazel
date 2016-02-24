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

DEPS = [
  "deps",
  "exports",
  "_robolectric", # From android_robolectric_test
  "_junit", # From android_robolectric_test
  "binary_under_test", #  From android_test
  "java_lib",# From proto_library
  "_proto1_java_lib", # From proto_library
]

RUNTIME_DEPS = [
  "runtime_deps",
]

ALL_DEPS = DEPS + RUNTIME_DEPS

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

def source_directory_tuple(resource_file):
  return (
      str(android_common.resource_source_directory(resource_file)),
      resource_file.is_source,
      resource_file.root.path if not resource_file.is_source else None
  )

def all_unique_source_directories(resources):
  # Sets can contain tuples, but cannot conntain structs.
  # Use set of tuples to unquify source directories.
  source_directory_tuples = set([source_directory_tuple(file) for file in resources])
  return [struct_omit_none(relative_path = relative_path,
                           is_source = is_source,
                           root_execution_path_fragment = root_execution_path_fragment)
          for (relative_path, is_source, root_execution_path_fragment) in source_directory_tuples]

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

def jars_from_output(output):
  if output == None:
    return []
  return [jar
          for jar in [output.class_jar, output.ijar, output.source_jar]
          if jar != None and not jar.is_source]

def java_rule_ide_info(target, ctx):
  if not hasattr(target, "java"):
    return (None, set())
  if hasattr(ctx.rule.attr, "srcs"):
     sources = [artifact_location(file)
                for src in ctx.rule.attr.srcs
                for file in src.files]
  else:
     sources = []

  jars = [library_artifact(output) for output in target.java.outputs.jars]
  ide_resolve_files = set([jar
       for output in target.java.outputs.jars
       for jar in jars_from_output(output)])

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

def android_rule_ide_info(target, ctx):
  if not hasattr(target, 'android'):
    return (None, set())
  ide_resolve_files = set(jars_from_output(target.android.idl.output))
  return (struct_omit_none(
            java_package = target.android.java_package,
            manifest = artifact_location(target.android.manifest),
            apk = artifact_location(target.android.apk),
            has_idl_sources = target.android.idl.output != None,
            idl_jar = library_artifact(target.android.idl.output),
            generate_resource_class = target.android.defines_resources,
            resources = all_unique_source_directories(target.android.resources),
        ),
        ide_resolve_files)

def collect_labels(rule_attrs, attr_list):
  return set([str(dep.label)
      for attr_name in attr_list
      if hasattr(rule_attrs, attr_name)
      for dep in getattr(rule_attrs, attr_name)])

def collect_export_deps(rule_attrs):
  result = set()
  for attr_name in DEPS:
    if hasattr(rule_attrs, attr_name):
      for dep in getattr(rule_attrs, attr_name):
        result = result | dep.export_deps
  return result


def _aspect_impl(target, ctx):
  kind = get_kind(target, ctx)
  rule_attrs = ctx.rule.attr

  compiletime_deps = collect_labels(rule_attrs, DEPS) | collect_export_deps(rule_attrs)
  runtime_deps = collect_labels(rule_attrs, RUNTIME_DEPS)

  ide_info_text = set()
  ide_resolve_files = set()

  for attr_name in ALL_DEPS:
    if hasattr(rule_attrs, attr_name):
      for dep in getattr(rule_attrs, attr_name):
        ide_info_text = ide_info_text | dep.intellij_info_files.ide_info_text
        ide_resolve_files = ide_resolve_files | dep.intellij_info_files.ide_resolve_files

  (java_rule_ide_info, java_ide_resolve_files) = java_rule_ide_info(target, ctx)
  ide_resolve_files = ide_resolve_files | java_ide_resolve_files

  (android_rule_ide_info, android_ide_resolve_files) = android_rule_ide_info(target, ctx)
  ide_resolve_files = ide_resolve_files | android_ide_resolve_files

  export_deps = set()
  if hasattr(target, "java"):
    export_deps = set([str(l) for l in target.java.transitive_exports])
    # Empty android libraries export all their dependencies.
    if ctx.rule.kind == "android_library" and \
            (not hasattr(rule_attrs, "src") or not ctx.rule.attr.src):
      export_deps = export_deps | compiletime_deps

  info = struct_omit_none(
      label = str(target.label),
      kind = kind,
      dependencies = list(compiletime_deps),
      runtime_deps = list(runtime_deps),
      build_file_artifact_location = build_file_artifact_location(ctx.build_file_path),
      java_rule_ide_info = java_rule_ide_info,
      android_rule_ide_info = android_rule_ide_info,
      tags = ctx.rule.attr.tags,
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
      ),
      export_deps = export_deps,
    )

intellij_info_aspect = aspect(implementation = _aspect_impl,
    attr_aspects = ALL_DEPS
)
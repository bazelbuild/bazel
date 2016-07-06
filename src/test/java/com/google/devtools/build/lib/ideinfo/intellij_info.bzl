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

# Implementaion of AndroidStudio-specific information collecting aspect.

# A map to convert JavaApiFlavor to ProtoLibraryLegacyJavaIdeInfo.ApiFlavor
_api_flavor_to_id = {
    "FLAVOR_NONE": 0,
    "FLAVOR_IMMUTABLE": 1,
    "FLAVOR_MUTABLE": 2,
    "FLAVOR_BOTH": 3,
}

# Compile-time dependency attributes, grouped by type.
DEPS = struct(
    label = [
        "binary_under_test",  #  From android_test
        "java_lib",  # From proto_library
        "_proto1_java_lib",  # From proto_library
        "_junit",  # From android_robolectric_test
        "_cc_toolchain",  # From C rules
        "module_target",
        "_java_toolchain",  # From java rules
    ],
    label_list = [
        "deps",
        "exports",
        "_robolectric",  # From android_robolectric_test
    ],
)

# Run-time dependency attributes, grouped by type.
RUNTIME_DEPS = struct(
    label = [],
    label_list = [
        "runtime_deps",
    ],
)

# All dependency attributes along which the aspect propagates, grouped by type.
ALL_DEPS = struct(
    label = DEPS.label + RUNTIME_DEPS.label,
    label_list = DEPS.label_list + RUNTIME_DEPS.label_list,
)

LEGACY_RESOURCE_ATTR = "resources"

##### Helpers

def struct_omit_none(**kwargs):
    """A replacement for standard `struct` function that omits the fields with None value."""
    d = {name: kwargs[name] for name in kwargs if kwargs[name] != None}
    return struct(**d)

def artifact_location(file):
  """Creates an ArtifactLocation proto from a File."""
  if file == None:
    return None
  return struct_omit_none(
      relative_path = file.short_path,
      is_source = file.is_source,
      root_execution_path_fragment = file.root.path if not file.is_source else None,
  )

def source_directory_tuple(resource_file):
  """Creates a tuple of (source directory, is_source, root execution path)."""
  return (
      str(android_common.resource_source_directory(resource_file)),
      resource_file.is_source,
      resource_file.root.path if not resource_file.is_source else None
  )

def all_unique_source_directories(resources):
  """Builds a list of ArtifactLocation protos.

  This is done for all source directories for a list of Android resources.
  """
  # Sets can contain tuples, but cannot contain structs.
  # Use set of tuples to unquify source directories.
  source_directory_tuples = set([source_directory_tuple(file) for file in resources])
  return [struct_omit_none(relative_path = relative_path,
                           is_source = is_source,
                           root_execution_path_fragment = root_execution_path_fragment)
          for (relative_path, is_source, root_execution_path_fragment) in source_directory_tuples]

def build_file_artifact_location(build_file_path):
  """Creates an ArtifactLocation proto representing a location of a given BUILD file."""
  return struct(
      relative_path = build_file_path,
      is_source = True,
  )

def library_artifact(java_output):
  """Creates a LibraryArtifact representing a given java_output."""
  if java_output == None or java_output.class_jar == None:
    return None
  return struct_omit_none(
        jar = artifact_location(java_output.class_jar),
        interface_jar = artifact_location(java_output.ijar),
        source_jar = artifact_location(java_output.source_jar),
  )

def annotation_processing_jars(annotation_processing):
  """Creates a LibraryArtifact representing Java annotation processing jars."""
  return struct_omit_none(
        jar = artifact_location(annotation_processing.class_jar),
        source_jar = artifact_location(annotation_processing.source_jar),
  )

def jars_from_output(output):
  """Collect jars for ide-resolve-files from Java output."""
  if output == None:
    return []
  return [jar
          for jar in [output.class_jar, output.ijar, output.source_jar]
          if jar != None and not jar.is_source]

# TODO(salguarnieri) Remove once skylark provides the path safe string from a PathFragment.
def replace_empty_path_with_dot(pathString):
  return "." if len(pathString) == 0 else pathString

def sources_from_rule(context):
  """
  Get the list of sources from a rule as artifact locations.

  Returns the list of sources as artifact locations for a rule or an empty list if no sources are
  present.
  """

  if hasattr(context.rule.attr, "srcs"):
    return [artifact_location(file)
            for src in context.rule.attr.srcs
            for file in src.files]
  return []

def collect_targets_from_attrs(rule_attrs, attrs):
  """Returns a list of targets from the given attributes."""
  list_deps = [dep for attr_name in attrs.label_list
               if hasattr(rule_attrs, attr_name)
               for dep in getattr(rule_attrs, attr_name)]

  scalar_deps = [getattr(rule_attrs, attr_name) for attr_name in attrs.label
                 if hasattr(rule_attrs, attr_name)]

  return [dep for dep in (list_deps + scalar_deps) if is_valid_aspect_target(dep)]

def collect_transitive_exports(targets):
  """Build a union of all export dependencies."""
  result = set()
  for dep in targets:
    result = result | dep.export_deps
  return result

def get_legacy_resource_dep(rule_attrs):
  """Gets the legacy 'resources' attribute."""
  legacy_resource_target = None
  if hasattr(rule_attrs, LEGACY_RESOURCE_ATTR):
    dep = getattr(rule_attrs, LEGACY_RESOURCE_ATTR)
    # resources can sometimes be a list attribute, in which case we don't want it
    if dep and is_valid_aspect_target(dep):
      legacy_resource_target = dep
  return legacy_resource_target

def targets_to_labels(targets):
  """Returns a set of label strings for the given targets."""
  return set([str(target.label) for target in targets])

def list_omit_none(value):
  """Returns a list of the value, or the empty list if None."""
  return [value] if value else []

def is_valid_aspect_target(target):
  """Returns whether the target has had the aspect run on it."""
  return hasattr(target, "intellij_aspect")

super_secret_rule_name = "".join(["gen", "m", "p", "m"])  # Take that, leak test
is_bazel = not hasattr(native, super_secret_rule_name)
def tool_label(label_str):
  """Returns a label that points to a blaze/bazel tool.

  Will be removed once the aspect is migrated out of core.
  """
  return Label("@bazel_tools" + label_str if is_bazel else label_str)

##### Builders for individual parts of the aspect output

def build_c_rule_ide_info(target, ctx):
  """Build CRuleIdeInfo.

  Returns a pair of (CRuleIdeInfo proto, a set of ide-resolve-files).
  (or (None, empty set) if the rule is not a C rule).
  """
  if not hasattr(target, "cc"):
    return (None, set())

  sources = sources_from_rule(ctx)

  rule_includes = []
  if hasattr(ctx.rule.attr, "includes"):
    rule_includes = ctx.rule.attr.includes
  rule_defines = []
  if hasattr(ctx.rule.attr, "defines"):
    rule_defines = ctx.rule.attr.defines
  rule_copts = []
  if hasattr(ctx.rule.attr, "copts"):
    rule_copts = ctx.rule.attr.copts

  cc_provider = target.cc

  c_rule_ide_info = struct_omit_none(
      source = sources,
      rule_include = rule_includes,
      rule_define = rule_defines,
      rule_copt = rule_copts,
      transitive_include_directory = cc_provider.include_directories,
      transitive_quote_include_directory = cc_provider.quote_include_directories,
      transitive_define = cc_provider.defines,
      transitive_system_include_directory = cc_provider.system_include_directories,
  )
  ide_resolve_files = cc_provider.transitive_headers
  return (c_rule_ide_info, ide_resolve_files)

def build_c_toolchain_ide_info(target, ctx):
  """Build CToolchainIdeInfo.

  Returns a pair of (CToolchainIdeInfo proto, a set of ide-resolve-files).
  (or (None, empty set) if the rule is not a cc_toolchain rule).
  """

  if ctx.rule.kind != "cc_toolchain":
    return (None, set())

  # This should exist because we requested it in our aspect definition.
  cc_fragment = ctx.fragments.cpp

  c_toolchain_ide_info = struct_omit_none(
      target_name = cc_fragment.target_gnu_system_name,
      base_compiler_option = cc_fragment.compiler_options(ctx.features),
      c_option = cc_fragment.c_options,
      cpp_option = cc_fragment.cxx_options(ctx.features),
      link_option = cc_fragment.link_options,
      unfiltered_compiler_option = cc_fragment.unfiltered_compiler_options(ctx.features),
      preprocessor_executable = replace_empty_path_with_dot(
          str(cc_fragment.preprocessor_executable)),
      cpp_executable = str(cc_fragment.compiler_executable),
      built_in_include_directory = [str(d)
                                    for d in cc_fragment.built_in_include_directories],
  )
  return (c_toolchain_ide_info, set())

def build_java_rule_ide_info(target, ctx):
  """
  Build JavaRuleIdeInfo.

  Returns a pair of (JavaRuleIdeInfo proto, a set of ide-info-files, a set of ide-resolve-files).
  (or (None, empty set, empty set) if the rule is not Java rule).
  """
  if not hasattr(target, "java") or ctx.rule.kind == "proto_library":
    return (None, set(), set())

  sources = sources_from_rule(ctx)

  jars = [library_artifact(output) for output in target.java.outputs.jars]
  ide_resolve_files = set([jar
       for output in target.java.outputs.jars
       for jar in jars_from_output(output)])

  gen_jars = []
  if target.java.annotation_processing and target.java.annotation_processing.enabled:
    gen_jars = [annotation_processing_jars(target.java.annotation_processing)]
    ide_resolve_files = ide_resolve_files | set([
        jar for jar in [target.java.annotation_processing.class_jar,
                        target.java.annotation_processing.source_jar]
        if jar != None and not jar.is_source])

  jdeps = artifact_location(target.java.outputs.jdeps)

  package_manifest = build_java_package_manifest(target, ctx)
  ide_info_files = set([package_manifest]) if package_manifest else set()

  java_rule_ide_info = struct_omit_none(
      sources = sources,
      jars = jars,
      jdeps = jdeps,
      generated_jars = gen_jars,
      package_manifest = artifact_location(package_manifest),
  )
  return (java_rule_ide_info, ide_info_files, ide_resolve_files)

def build_java_package_manifest(target, ctx):
  """Builds a java package manifest and returns the output file."""
  source_files = java_sources_for_package_manifest(ctx)
  if not source_files:
    return None

  output = ctx.new_file(target.label.name + ".manifest")

  args = []
  args += ["--output_manifest", output.path]
  args += ["--sources"]
  args += [":".join([f.root.path + "," + f.path for f in source_files])]
  argfile = ctx.new_file(ctx.configuration.bin_dir,
                         target.label.name + ".manifest.params")
  ctx.file_action(output=argfile, content="\n".join(args))

  ctx.action(
      inputs = source_files + [argfile],
      outputs = [output],
      executable = ctx.executable._package_parser,
      arguments = ["@" + argfile.path],
      mnemonic = "JavaPackageManifest",
      progress_message = "Parsing java package strings for " + str(target.label),
  )
  return output

def java_sources_for_package_manifest(ctx):
  """Get the list of non-generated java sources to go in the package manifest."""

  if hasattr(ctx.rule.attr, "srcs"):
    return [f
            for src in ctx.rule.attr.srcs
            for f in src.files
            if f.is_source and f.basename.endswith(".java")]
  return []

def build_android_rule_ide_info(target, ctx, legacy_resource_label):
  """Build AndroidRuleIdeInfo.

  Returns a pair of (AndroidRuleIdeInfo proto, a set of ide-resolve-files).
  (or (None, empty set) if the rule is not Android rule).
  """
  if not hasattr(target, "android"):
    return (None, set())

  android_rule_ide_info = struct_omit_none(
      java_package = target.android.java_package,
      manifest = artifact_location(target.android.manifest),
      apk = artifact_location(target.android.apk),
      dependency_apk = [artifact_location(apk) for apk in target.android.apks_under_test],
      has_idl_sources = target.android.idl.output != None,
      idl_jar = library_artifact(target.android.idl.output),
      generate_resource_class = target.android.defines_resources,
      resources = all_unique_source_directories(target.android.resources),
      resource_jar = library_artifact(target.android.resource_jar),
      legacy_resources = legacy_resource_label,
  )
  ide_resolve_files = set(jars_from_output(target.android.idl.output))
  return (android_rule_ide_info, ide_resolve_files)

def build_test_info(target, ctx):
  """Build TestInfo"""
  if not is_test_rule(ctx):
    return None
  return struct_omit_none(
           size = ctx.rule.attr.size,
         )

def is_test_rule(ctx):
  kind_string = ctx.rule.kind
  return kind_string.endswith("_test")

def build_proto_library_legacy_java_ide_info(target, ctx):
  """Build ProtoLibraryLegacyJavaIdeInfo."""
  if not hasattr(target, "proto_legacy_java"):
    return None
  proto_info = target.proto_legacy_java.legacy_info
  return struct_omit_none(
    api_version = proto_info.api_version,
    api_flavor = _api_flavor_to_id[proto_info.api_flavor],
    jars1 = [library_artifact(output) for output in proto_info.jars1],
    jars_mutable = [library_artifact(output) for output in proto_info.jars_mutable],
    jars_immutable = [library_artifact(output) for output in proto_info.jars_immutable],
  )

def build_java_toolchain_ide_info(target):
  """Build JavaToolchainIdeInfo."""
  if not hasattr(target, "java_toolchain"):
    return None
  toolchain_info = target.java_toolchain
  return struct_omit_none(
      source_version = toolchain_info.source_version,
      target_version = toolchain_info.target_version,
  )

##### Main aspect function

def _aspect_impl_helper(target, ctx, for_test):
  """Aspect implementation function."""
  rule_attrs = ctx.rule.attr

  # Collect direct dependencies
  direct_dep_targets = collect_targets_from_attrs(rule_attrs, DEPS)

  # Add exports from direct dependencies
  exported_deps_from_deps = collect_transitive_exports(direct_dep_targets)
  compiletime_deps = targets_to_labels(direct_dep_targets) | exported_deps_from_deps

  # Propagate my own exports
  export_deps = set()
  if hasattr(target, "java"):
    export_deps = set([str(l) for l in target.java.transitive_exports])
    # Empty android libraries export all their dependencies.
    if ctx.rule.kind == "android_library":
      if not hasattr(rule_attrs, "srcs") or not ctx.rule.attr.srcs:
        export_deps = export_deps | compiletime_deps

  # runtime_deps
  runtime_dep_targets = collect_targets_from_attrs(rule_attrs, RUNTIME_DEPS)
  runtime_deps = targets_to_labels(runtime_dep_targets)

  # resources
  legacy_resource_target = get_legacy_resource_dep(rule_attrs)
  legacy_resource_label = str(legacy_resource_target.label) if legacy_resource_target else None

  # Roll up files from my prerequisites
  prerequisites = direct_dep_targets + runtime_dep_targets + list_omit_none(legacy_resource_target)
  ide_info_text = set()
  ide_resolve_files = set()
  intellij_infos = dict()
  for dep in prerequisites:
    ide_info_text = ide_info_text | dep.intellij_info_files.ide_info_text
    ide_resolve_files = ide_resolve_files | dep.intellij_info_files.ide_resolve_files
    if for_test:
      intellij_infos.update(dep.intellij_infos)

  # Collect C-specific information
  (c_rule_ide_info, c_ide_resolve_files) = build_c_rule_ide_info(target, ctx)
  ide_resolve_files = ide_resolve_files | c_ide_resolve_files

  (c_toolchain_ide_info, c_toolchain_ide_resolve_files) = build_c_toolchain_ide_info(target, ctx)
  ide_resolve_files = ide_resolve_files | c_toolchain_ide_resolve_files

  # Collect Java-specific information
  (java_rule_ide_info, java_ide_info_files, java_ide_resolve_files) = build_java_rule_ide_info(
      target, ctx)
  ide_info_text = ide_info_text | java_ide_info_files
  ide_resolve_files = ide_resolve_files | java_ide_resolve_files

  # Collect Android-specific information
  (android_rule_ide_info, android_ide_resolve_files) = build_android_rule_ide_info(
      target, ctx, legacy_resource_label)
  ide_resolve_files = ide_resolve_files | android_ide_resolve_files

  # legacy proto_library support
  proto_library_legacy_java_ide_info = build_proto_library_legacy_java_ide_info(target, ctx)

  # java_toolchain
  java_toolchain_ide_info = build_java_toolchain_ide_info(target)

  # Collect test info
  test_info = build_test_info(target, ctx)

  # Build RuleIdeInfo proto
  info = struct_omit_none(
      label = str(target.label),
      kind_string = ctx.rule.kind,
      dependencies = list(compiletime_deps),
      runtime_deps = list(runtime_deps),
      build_file_artifact_location = build_file_artifact_location(ctx.build_file_path),
      c_rule_ide_info = c_rule_ide_info,
      c_toolchain_ide_info = c_toolchain_ide_info,
      java_rule_ide_info = java_rule_ide_info,
      android_rule_ide_info = android_rule_ide_info,
      tags = ctx.rule.attr.tags,
      test_info = test_info,
      proto_library_legacy_java_ide_info = proto_library_legacy_java_ide_info,
      java_toolchain_ide_info = java_toolchain_ide_info,
  )

  # Output the ide information file.
  output = ctx.new_file(target.label.name + ".intellij-build.txt")
  ctx.file_action(output, info.to_proto())
  ide_info_text = ide_info_text | set([output])
  if for_test:
    intellij_infos[str(target.label)] = info
  else:
    intellij_infos = None

  # Return providers.
  return struct_omit_none(
      intellij_aspect = True,
      output_groups = {
        "ide-info-text" : ide_info_text,
        "ide-resolve" : ide_resolve_files,
      },
      intellij_info_files = struct(
        ide_info_text = ide_info_text,
        ide_resolve_files = ide_resolve_files,
      ),
      intellij_infos = intellij_infos,
      export_deps = export_deps,
    )

def _aspect_impl(target, ctx):
  return _aspect_impl_helper(target, ctx, for_test=False)

def _test_aspect_impl(target, ctx):
  return _aspect_impl_helper(target, ctx, for_test=True)

def _aspect_def(impl):
  return aspect(
      attrs = {
          "_package_parser": attr.label(
              default = tool_label("//tools/android:PackageParser"),
              cfg = HOST_CFG,
              executable = True,
              allow_files = True),
      },
      attr_aspects = ALL_DEPS.label + ALL_DEPS.label_list + [LEGACY_RESOURCE_ATTR],
      fragments = ["cpp"],
      implementation = impl,
  )


intellij_info_aspect = _aspect_def(_aspect_impl)
intellij_info_test_aspect = _aspect_def(_test_aspect_impl)

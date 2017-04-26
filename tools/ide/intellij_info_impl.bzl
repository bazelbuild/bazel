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

"""Implementation of IntelliJ-specific information collecting aspect."""

# Compile-time dependency attributes, grouped by type.
DEPS = [
    "_cc_toolchain",  # From C rules
    "_java_toolchain",  # From java rules
    "deps",
    "exports",
    "_robolectric",  # From android_robolectric_test
    "_android_sdk",  # from android rules
    "aidl_lib",  # from android_sdk
    "_scalalib",  # From scala rules
    "_scalacompiler",  # From scala rules
    "_scalareflect",  # From scala rules
    "_scalatest",  # From scala_test rules
    "_scalatest_reporter",  # From scala_test rules
    "_scalaxml",  # From scala_test rules
]

# Run-time dependency attributes, grouped by type.
RUNTIME_DEPS = [
    "runtime_deps",
]

PREREQUISITE_DEPS = []

# Dependency type enum
COMPILE_TIME = 0
RUNTIME = 1

##### Helpers

def struct_omit_none(**kwargs):
  """A replacement for standard `struct` function that omits the fields with None value."""
  d = {name: kwargs[name] for name in kwargs if kwargs[name] != None}
  return struct(**d)

def artifact_location(f):
  """Creates an ArtifactLocation proto from a File."""
  if f == None:
    return None

  return to_artifact_location(
      f.path,
      f.root.path if not f.is_source else "",
      f.is_source,
      is_external_artifact(f.owner),
  )

def to_artifact_location(exec_path, root_exec_path_fragment, is_source, is_external):
  """Derives workspace path from other path fragments, and creates an ArtifactLocation proto."""
  # Bazel 0.4.4 has directory structure:
  # exec_path = (root_fragment)? + (external/repo_name)? + relative_path
  # Bazel 0.4.5 has planned directory structure:
  # exec_path = (../repo_name)? + (root_fragment)? + relative_path
  # Handle both cases by trying to strip the external workspace prefix before and after removing
  # root_exec_path_fragment.
  relative_path = strip_external_workspace_prefix(exec_path)
  relative_path = strip_root_exec_path_fragment(relative_path, root_exec_path_fragment)
  # Remove this line when Bazel 0.4.4 and earlier no longer need to be supported.
  relative_path = strip_external_workspace_prefix(relative_path)

  root_exec_path_fragment = exec_path[:-(len("/" + relative_path))]

  return struct_omit_none(
      relative_path = relative_path,
      is_source = is_source,
      is_external = is_external,
      root_execution_path_fragment = root_exec_path_fragment,
      is_new_external_version = True,
  )

def strip_root_exec_path_fragment(path, root_fragment):
  if root_fragment and path.startswith(root_fragment + "/"):
    return path[len(root_fragment + "/"):]
  return path

def strip_external_workspace_prefix(path):
  """Either 'external/workspace_name/' or '../workspace_name/'."""
  # Label.EXTERNAL_PATH_PREFIX is due to change from 'external' to '..' in Bazel 0.4.5.
  # This code is for forwards and backwards compatibility.
  # Remove the 'external/' check when Bazel 0.4.4 and earlier no longer need to be supported.
  if path.startswith("../") or path.startswith("external/"):
    return "/".join(path.split("/")[2:])
  return path

def is_external_artifact(label):
  """Determines whether a label corresponds to an external artifact."""
  # Label.EXTERNAL_PATH_PREFIX is due to change from 'external' to '..' in Bazel 0.4.5.
  # This code is for forwards and backwards compatibility.
  # Remove the 'external' check when Bazel 0.4.4 and earlier no longer need to be supported.
  return label.workspace_root.startswith("external") or label.workspace_root.startswith("..")

def source_directory_tuple(resource_file):
  """Creates a tuple of (exec_path, root_exec_path_fragment, is_source, is_external)."""
  relative_path = str(android_common.resource_source_directory(resource_file))
  root_exec_path_fragment = resource_file.root.path if not resource_file.is_source else None
  return (
      relative_path if resource_file.is_source else root_exec_path_fragment + relative_path,
      root_exec_path_fragment,
      resource_file.is_source,
      is_external_artifact(resource_file.owner)
  )

def all_unique_source_directories(resources):
  """Builds a list of unique ArtifactLocation protos."""
  # Sets can contain tuples, but cannot contain structs.
  # Use set of tuples to unquify source directories.
  source_directory_tuples = set([source_directory_tuple(f) for f in resources])
  return [to_artifact_location(
      exec_path,
      root_path_fragment,
      is_source,
      is_external)
          for (exec_path, root_path_fragment, is_source, is_external) in source_directory_tuples]

def build_file_artifact_location(ctx):
  """Creates an ArtifactLocation proto representing a location of a given BUILD file."""
  return to_artifact_location(
      ctx.build_file_path,
      ctx.build_file_path,
      True,
      is_external_artifact(ctx.label)
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
  """Collect jars for intellij-resolve-files from Java output."""
  if output == None:
    return []
  return [jar
          for jar in [output.class_jar, output.ijar, output.source_jar]
          if jar != None and not jar.is_source]

# TODO(salguarnieri) Remove once skylark provides the path safe string from a PathFragment.
def replace_empty_path_with_dot(path):
  return path or "."

def sources_from_target(ctx):
  """Get the list of sources from a target as artifact locations."""

  if hasattr(ctx.rule.attr, "srcs"):
    return [artifact_location(f)
            for src in ctx.rule.attr.srcs
            for f in src.files]
  return []

def _collect_target_from_attr(rule_attrs, attr_name, result):
  """Collects the targets from the given attr into the result."""
  if not hasattr(rule_attrs, attr_name):
    return
  attr_value = getattr(rule_attrs, attr_name)
  type_name = type(attr_value)
  if type_name == "Target":
    result.append(attr_value)
  elif type_name == "list":
    result.extend(attr_value)

def collect_targets_from_attrs(rule_attrs, attrs):
  """Returns a list of targets from the given attributes."""
  result = []
  for attr_name in attrs:
    _collect_target_from_attr(rule_attrs, attr_name, result)
  return [target for target in result if is_valid_aspect_target(target)]

def targets_to_labels(targets):
  """Returns a set of label strings for the given targets."""
  return set([str(target.label) for target in targets])

def list_omit_none(value):
  """Returns a list of the value, or the empty list if None."""
  return [value] if value else []

def is_valid_aspect_target(target):
  """Returns whether the target has had the aspect run on it."""
  return hasattr(target, "intellij_info")

def get_aspect_ids(ctx, target):
  """Returns the all aspect ids, filtering out self."""
  aspect_ids = None
  if hasattr(ctx, "aspect_ids"):
    aspect_ids = ctx.aspect_ids
  elif hasattr(target, "aspect_ids"):
    aspect_ids = target.aspect_ids
  else:
    return None
  return [aspect_id for aspect_id in aspect_ids if "intellij_info_aspect" not in aspect_id]

def make_target_key(label, aspect_ids):
  """Returns a TargetKey proto struct from a target."""
  return struct_omit_none(
      label = str(label),
      aspect_ids = tuple(aspect_ids) if aspect_ids else None
  )

def make_dep(dep, dependency_type):
  """Returns a Dependency proto struct."""
  return struct(
      target = dep.intellij_info.target_key,
      dependency_type = dependency_type,
  )

def make_deps(deps, dependency_type):
  """Returns a list of Dependency proto structs."""
  return [make_dep(dep, dependency_type) for dep in deps]

def make_dep_from_label(label, dependency_type):
  """Returns a Dependency proto struct from a label."""
  return struct(
      target = struct(label = str(label)),
      dependency_type = dependency_type,
  )

##### Builders for individual parts of the aspect output

def build_py_ide_info(target, ctx):
  """Build PyIdeInfo."""
  if not hasattr(target, "py"):
    return (None, set())

  sources = sources_from_target(ctx)
  transitive_sources = target.py.transitive_sources

  py_ide_info = struct_omit_none(
      sources = sources,
  )
  return (py_ide_info, transitive_sources)

def build_c_ide_info(target, ctx):
  """Build CIdeInfo."""
  if not hasattr(target, "cc"):
    return (None, set())

  sources = sources_from_target(ctx)

  target_includes = []
  if hasattr(ctx.rule.attr, "includes"):
    target_includes = ctx.rule.attr.includes
  target_defines = []
  if hasattr(ctx.rule.attr, "defines"):
    target_defines = ctx.rule.attr.defines
  target_copts = []
  if hasattr(ctx.rule.attr, "copts"):
    target_copts = ctx.rule.attr.copts

  cc_provider = target.cc

  c_ide_info = struct_omit_none(
      source = sources,
      target_include = target_includes,
      target_define = target_defines,
      target_copt = target_copts,
      transitive_include_directory = cc_provider.include_directories,
      transitive_quote_include_directory = cc_provider.quote_include_directories,
      transitive_define = cc_provider.defines,
      transitive_system_include_directory = cc_provider.system_include_directories,
  )
  intellij_resolve_files = cc_provider.transitive_headers
  return (c_ide_info, intellij_resolve_files)

def build_c_toolchain_ide_info(ctx):
  """Build CToolchainIdeInfo."""

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

def get_java_provider(target):
  if hasattr(target, "proto_java"):
    return target.proto_java
  if hasattr(target, "java"):
    return target.java
  return None

def build_java_ide_info(target, ctx, semantics):
  """Build JavaIdeInfo."""
  java = get_java_provider(target)
  if not java:
    return (None, set(), set())

  java_semantics = semantics.java if hasattr(semantics, "java") else None
  if java_semantics and java_semantics.skip_target(target, ctx):
    return (None, set(), set())

  ide_info_files = set()
  sources = sources_from_target(ctx)

  jars = [library_artifact(output) for output in java.outputs.jars]
  output_jars = [jar for output in java.outputs.jars for jar in jars_from_output(output)]
  intellij_resolve_files = set(output_jars)

  gen_jars = []
  if java.annotation_processing and java.annotation_processing.enabled:
    gen_jars = [annotation_processing_jars(java.annotation_processing)]
    intellij_resolve_files = intellij_resolve_files | set([
        jar for jar in [java.annotation_processing.class_jar,
                        java.annotation_processing.source_jar]
        if jar != None and not jar.is_source])

  jdeps = artifact_location(java.outputs.jdeps)

  java_sources, gen_java_sources, srcjars = divide_java_sources(ctx)

  if java_semantics:
    srcjars = java_semantics.filter_source_jars(target, ctx, srcjars)

  package_manifest = None
  if java_sources:
    package_manifest = build_java_package_manifest(ctx, target, java_sources, ".java-manifest")
    ide_info_files = ide_info_files | set([package_manifest])

  filtered_gen_jar = None
  if java_sources and (gen_java_sources or srcjars):
    filtered_gen_jar, filtered_gen_resolve_files = build_filtered_gen_jar(
        ctx,
        target,
        java,
        gen_java_sources,
        srcjars
    )
    intellij_resolve_files = intellij_resolve_files | filtered_gen_resolve_files

  java_ide_info = struct_omit_none(
      sources = sources,
      jars = jars,
      jdeps = jdeps,
      generated_jars = gen_jars,
      package_manifest = artifact_location(package_manifest),
      filtered_gen_jar = filtered_gen_jar,
      main_class = ctx.rule.attr.main_class if hasattr(ctx.rule.attr, "main_class") else None,
  )
  return (java_ide_info, ide_info_files, intellij_resolve_files)

def _package_manifest_file_argument(f):
  artifact = artifact_location(f)
  is_external = "1" if is_external_artifact(f.owner) else "0"
  return artifact.root_execution_path_fragment + "," + artifact.relative_path + "," + is_external

def build_java_package_manifest(ctx, target, source_files, suffix):
  """Builds the java package manifest for the given source files."""
  output = ctx.new_file(target.label.name + suffix)

  args = []
  args += ["--output_manifest", output.path]
  args += ["--sources"]
  args += [":".join([_package_manifest_file_argument(f) for f in source_files])]
  argfile = ctx.new_file(ctx.configuration.bin_dir,
                         target.label.name + suffix + ".params")
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

def build_filtered_gen_jar(ctx, target, java, gen_java_sources, srcjars):
  """Filters the passed jar to contain only classes from the given manifest."""
  jar_artifacts = []
  source_jar_artifacts = []
  for jar in java.outputs.jars:
    if jar.ijar:
      jar_artifacts.append(jar.ijar)
    elif jar.class_jar:
      jar_artifacts.append(jar.class_jar)
    if jar.source_jar:
      source_jar_artifacts.append(jar.source_jar)

  filtered_jar = ctx.new_file(target.label.name + "-filtered-gen.jar")
  filtered_source_jar = ctx.new_file(target.label.name + "-filtered-gen-src.jar")
  args = []
  args += ["--filter_jars"]
  args += [":".join([jar.path for jar in jar_artifacts])]
  args += ["--filter_source_jars"]
  args += [":".join([jar.path for jar in source_jar_artifacts])]
  args += ["--filtered_jar", filtered_jar.path]
  args += ["--filtered_source_jar", filtered_source_jar.path]
  if gen_java_sources:
    args += ["--keep_java_files"]
    args += [":".join([java_file.path for java_file in gen_java_sources])]
  if srcjars:
    args += ["--keep_source_jars"]
    args += [":".join([source_jar.path for source_jar in srcjars])]
  ctx.action(
      inputs = jar_artifacts + source_jar_artifacts + gen_java_sources + srcjars,
      outputs = [filtered_jar, filtered_source_jar],
      executable = ctx.executable._jar_filter,
      arguments = args,
      mnemonic = "JarFilter",
      progress_message = "Filtering generated code for " + str(target.label),
  )
  output_jar = struct(
      jar=artifact_location(filtered_jar),
      source_jar=artifact_location(filtered_source_jar),
  )
  intellij_resolve_files = set([filtered_jar, filtered_source_jar])
  return output_jar, intellij_resolve_files

def divide_java_sources(ctx):
  """Divide sources into plain java, generated java, and srcjars."""

  java_sources = []
  gen_java_sources = []
  srcjars = []
  if hasattr(ctx.rule.attr, "srcs"):
    srcs = ctx.rule.attr.srcs
    for src in srcs:
      for f in src.files:
        if f.basename.endswith(".java"):
          if f.is_source:
            java_sources.append(f)
          else:
            gen_java_sources.append(f)
        elif f.basename.endswith(".srcjar"):
          srcjars.append(f)

  return java_sources, gen_java_sources, srcjars

def build_android_ide_info(target, ctx, semantics):
  """Build AndroidIdeInfo."""
  if not hasattr(target, "android"):
    return (None, set())

  android_semantics = semantics.android if hasattr(semantics, "android") else None
  extra_ide_info = android_semantics.extra_ide_info(target, ctx) if android_semantics else {}

  android = target.android
  android_ide_info = struct_omit_none(
      java_package = android.java_package,
      idl_import_root = android.idl.import_root if hasattr(android.idl, "import_root") else None,
      manifest = artifact_location(android.manifest),
      apk = artifact_location(android.apk),
      dependency_apk = [artifact_location(apk) for apk in android.apks_under_test],
      has_idl_sources = android.idl.output != None,
      idl_jar = library_artifact(android.idl.output),
      generate_resource_class = android.defines_resources,
      resources = all_unique_source_directories(android.resources),
      resource_jar = library_artifact(android.resource_jar),
      **extra_ide_info
  )
  intellij_resolve_files = set(jars_from_output(android.idl.output))

  if android.manifest and not android.manifest.is_source:
    intellij_resolve_files = intellij_resolve_files | set([android.manifest])

  return (android_ide_info, intellij_resolve_files)

def build_android_sdk_ide_info(ctx):
  if ctx.rule.kind != "android_sdk":
    return None
  android_jar_file = list(ctx.rule.attr.android_jar.files)[0]
  return struct(
      android_jar = artifact_location(android_jar_file),
  )

def build_scala_ide_info(target, ctx):
  """Build ScalaIdeInfo."""
  if not hasattr(target, "scala"):
    return (None, set(), set())

  ide_info_files = set()
  scala_sources = []
  if hasattr(ctx.rule.attr, "srcs"):
    scala_sources = [f for src in ctx.rule.attr.srcs for f in src.files if f.is_source]

  package_manifest = None
  if scala_sources:
    package_manifest = build_java_package_manifest(ctx, target, scala_sources, ".scala-manifest")
    ide_info_files = ide_info_files | set([package_manifest])

  scala = target.scala
  class_jar = None
  ijar = None
  intellij_resolve_files = set()
  if scala.outputs:
    class_jar = scala.outputs.class_jar
    ijar = scala.outputs.ijar
    intellij_resolve_files = set([class_jar, ijar])

  scala_ide_info = struct_omit_none(
      sources = sources_from_target(ctx),
      jars = [struct_omit_none(
          jar = artifact_location(class_jar),
          interface_jar = artifact_location(ijar),
      )],
      package_manifest = artifact_location(package_manifest),
      main_class = ctx.rule.attr.main_class if hasattr(ctx.rule.attr, "main_class") else None,
  )
  return (scala_ide_info, ide_info_files, intellij_resolve_files)


def build_test_info(ctx):
  """Build TestInfo."""
  if not is_test_rule(ctx):
    return None
  return struct_omit_none(
      size = ctx.rule.attr.size,
  )

def is_test_rule(ctx):
  kind_string = ctx.rule.kind
  return kind_string.endswith("_test")

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

def intellij_info_aspect_impl(target, ctx, semantics):
  """Aspect implementation function."""
  tags = ctx.rule.attr.tags
  if "no-ide" in tags:
    return struct()

  rule_attrs = ctx.rule.attr

  # Collect direct dependencies
  direct_dep_targets = collect_targets_from_attrs(
      rule_attrs, semantics_extra_deps(DEPS, semantics, "extra_deps"))
  direct_deps = make_deps(direct_dep_targets, COMPILE_TIME)

  # Add exports from direct dependencies
  exported_deps_from_deps = []
  for dep in direct_dep_targets:
    exported_deps_from_deps = exported_deps_from_deps + dep.intellij_info.export_deps

  # Combine into all compile time deps
  compiletime_deps = direct_deps + exported_deps_from_deps

  # Propagate my own exports
  export_deps = []
  if hasattr(target, "java"):
    transitive_exports = target.java.transitive_exports
    export_deps = [make_dep_from_label(label, COMPILE_TIME) for label in transitive_exports]
    # Empty android libraries export all their dependencies.
    if ctx.rule.kind == "android_library":
      if not hasattr(rule_attrs, "srcs") or not ctx.rule.attr.srcs:
        export_deps = export_deps + compiletime_deps
  export_deps = list(set(export_deps))

  # runtime_deps
  runtime_dep_targets = collect_targets_from_attrs(
      rule_attrs, semantics_extra_deps(RUNTIME_DEPS, semantics, "extra_runtime_deps"))
  runtime_deps = make_deps(runtime_dep_targets, RUNTIME)
  all_deps = list(set(compiletime_deps + runtime_deps))

  # extra prerequisites
  extra_prerequisite_targets = collect_targets_from_attrs(
      rule_attrs, semantics_extra_deps(PREREQUISITE_DEPS, semantics, "extra_prerequisites"))

  # Roll up files from my prerequisites
  prerequisites = direct_dep_targets + runtime_dep_targets + extra_prerequisite_targets
  intellij_info_text = set()
  intellij_resolve_files = set()
  intellij_compile_files = target.output_group("files_to_compile_INTERNAL_")
  for dep in prerequisites:
    intellij_info_text = intellij_info_text | dep.intellij_info.intellij_info_text
    intellij_resolve_files = intellij_resolve_files | dep.intellij_info.intellij_resolve_files

  # Collect python-specific information
  (py_ide_info, py_resolve_files) = build_py_ide_info(target, ctx)
  intellij_resolve_files = intellij_resolve_files | py_resolve_files

  # Collect C-specific information
  (c_ide_info, c_resolve_files) = build_c_ide_info(target, ctx)
  intellij_resolve_files = intellij_resolve_files | c_resolve_files

  (c_toolchain_ide_info, c_toolchain_resolve_files) = build_c_toolchain_ide_info(ctx)
  intellij_resolve_files = intellij_resolve_files | c_toolchain_resolve_files

  # Collect Java-specific information
  (java_ide_info, java_ide_info_files, java_resolve_files) = build_java_ide_info(
      target, ctx, semantics)
  intellij_info_text = intellij_info_text | java_ide_info_files
  intellij_resolve_files = intellij_resolve_files | java_resolve_files

  # Collect Android-specific information
  (android_ide_info, android_resolve_files) = build_android_ide_info(
      target, ctx, semantics)
  intellij_resolve_files = intellij_resolve_files | android_resolve_files
  android_sdk_ide_info = build_android_sdk_ide_info(ctx)

  # Collect Scala-specific information
  (scala_ide_info, scala_ide_info_files, scala_resolve_files) = build_scala_ide_info(
      target, ctx)
  intellij_info_text = intellij_info_text | scala_ide_info_files
  intellij_resolve_files = intellij_resolve_files | scala_resolve_files

  # java_toolchain
  java_toolchain_ide_info = build_java_toolchain_ide_info(target)

  # Collect test info
  test_info = build_test_info(ctx)

  file_name = target.label.name
  aspect_ids = get_aspect_ids(ctx, target)
  if aspect_ids:
    aspect_hash = hash(".".join(aspect_ids))
    file_name = file_name + "-" + str(aspect_hash)
  file_name = file_name + ".intellij-info.txt"

  # Any extra ide info
  extra_ide_info = {}
  if hasattr(semantics, "extra_ide_info"):
    extra_ide_info = semantics.extra_ide_info(target, ctx)

  # Build TargetIdeInfo proto
  target_key = make_target_key(target.label, aspect_ids)
  info = struct_omit_none(
      key = target_key,
      kind_string = ctx.rule.kind,
      deps = list(all_deps),
      build_file_artifact_location = build_file_artifact_location(ctx),
      c_ide_info = c_ide_info,
      c_toolchain_ide_info = c_toolchain_ide_info,
      java_ide_info = java_ide_info,
      android_ide_info = android_ide_info,
      android_sdk_ide_info = android_sdk_ide_info,
      scala_ide_info = scala_ide_info,
      tags = tags,
      test_info = test_info,
      java_toolchain_ide_info = java_toolchain_ide_info,
      py_ide_info = py_ide_info,
      **extra_ide_info
  )

  # Output the ide information file.
  output = ctx.new_file(file_name)
  ctx.file_action(output, info.to_proto())
  intellij_info_text = intellij_info_text | set([output])

  # Return providers.
  return struct_omit_none(
      output_groups = {
          "intellij-info-text" : intellij_info_text,
          "intellij-resolve" : intellij_resolve_files,
          "intellij-compile": intellij_compile_files,
      },
      intellij_info = struct(
          target_key = target_key,
          intellij_info_text = intellij_info_text,
          intellij_resolve_files = intellij_resolve_files,
          export_deps = export_deps,
      ),
  )

def semantics_extra_deps(base, semantics, name):
  if not hasattr(semantics, name):
    return base
  extra_deps = getattr(semantics, name)
  return base + extra_deps

def make_intellij_info_aspect(aspect_impl, semantics):
  """Creates the aspect given the semantics."""
  tool_label = semantics.tool_label
  deps = semantics_extra_deps(DEPS, semantics, "extra_deps")
  runtime_deps = semantics_extra_deps(RUNTIME_DEPS, semantics, "extra_runtime_deps")
  prerequisite_deps = semantics_extra_deps(PREREQUISITE_DEPS, semantics, "extra_prerequisites")

  attr_aspects = deps + runtime_deps + prerequisite_deps

  return aspect(
      attrs = {
          "_package_parser": attr.label(
              default = tool_label("//tools/android:PackageParser"),
              cfg = "host",
              executable = True,
              allow_files = True),
          "_jar_filter": attr.label(
              default = tool_label("//tools/android:JarFilter"),
              cfg = "host",
              executable = True,
              allow_files = True),
      },
      attr_aspects = attr_aspects,
      fragments = ["cpp"],
      implementation = aspect_impl,
      required_aspect_providers = ["proto_java"],
  )

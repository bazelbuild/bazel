# Copyright 2015 The Bazel Authors. All rights reserved.
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

"""Rules for supporting the Scala language."""


_scala_filetype = FileType([".scala"])

def _adjust_resources_path(path):
  dir_1, dir_2, rel_path = path.partition("resources")
  if rel_path:
    return dir_1 + dir_2, rel_path
  (dir_1,dir_2,rel_path) = path.partition("java")
  if rel_path:
    return dir_1 + dir_2, rel_path
  return "", path

def _compile(ctx, jars, buildijar):
  res_cmd = ""
  for f in ctx.files.resources:
    c_dir, res_path = _adjust_resources_path(f.path)
    change_dir = "-C " + c_dir if c_dir else ""
    res_cmd = "\n{jar} uf {out} " + change_dir + " " + res_path
  ijar_cmd = ""
  if buildijar:
    ijar_cmd = "\n{ijar} {out} {ijar_out}".format(
      ijar=ctx.file._ijar.path,
      out=ctx.outputs.jar.path,
      ijar_out=ctx.outputs.ijar.path)
  cmd = """
set -e
mkdir -p {out}_tmp
{scalac} {scala_opts} {jvm_flags} -classpath "{jars}" $@ -d {out}_tmp
# Make jar file deterministic by setting the timestamp of files
find {out}_tmp -exec touch -t 198001010000 {{}} \;
touch -t 198001010000 {manifest}
{jar} cmf {manifest} {out} -C {out}_tmp .
""" + ijar_cmd + res_cmd
  cmd = cmd.format(
      scalac=ctx.file._scalac.path,
      scala_opts=" ".join(ctx.attr.scalacopts),
      jvm_flags=" ".join(["-J" + flag for flag in ctx.attr.jvm_flags]),
      out=ctx.outputs.jar.path,
      manifest=ctx.outputs.manifest.path,
      jar=ctx.file._jar.path,
      ijar=ctx.file._ijar.path,
      jars=":".join([j.path for j in jars]),)
  outs = [ctx.outputs.jar]
  if buildijar:
    outs.extend([ctx.outputs.ijar])
  ctx.action(
      inputs=list(jars) +
          ctx.files.srcs +
          ctx.files.resources +
          ctx.files._jdk +
          ctx.files._scalasdk +
          [ctx.outputs.manifest, ctx.file._jar, ctx.file._ijar],
      outputs=outs,
      command=cmd,
      progress_message="scala %s" % ctx.label,
      arguments=[f.path for f in ctx.files.srcs])

def _write_manifest(ctx):
  # TODO(bazel-team): I don't think this classpath is what you want
  manifest = "Class-Path: %s\n" % ctx.file._scalalib.path
  if getattr(ctx.attr, "main_class", ""):
    manifest += "Main-Class: %s\n" % ctx.attr.main_class

  ctx.file_action(
      output = ctx.outputs.manifest,
      content = manifest)


def _write_launcher(ctx, jars):
  content = """#!/bin/bash
cd $0.runfiles
{java} -cp {cp} {name} "$@"
"""
  content = content.format(
      java=ctx.file._java.path,
      name=ctx.attr.main_class,
      deploy_jar=ctx.outputs.jar.path,
      cp=":".join([j.short_path for j in jars]))
  ctx.file_action(
      output=ctx.outputs.executable,
      content=content)

def _args_for_suites(suites):
  args = ["-o"]
  for suite in suites:
    args.extend(["-s", suite])
  return args

def _write_test_launcher(ctx, jars):
  content = """#!/bin/bash
cd $0.runfiles
{java} -cp {cp} {name} {args} "$@"
"""
  content = content.format(
      java=ctx.file._java.path,
      name=ctx.attr.main_class,
      args=' '.join(_args_for_suites(ctx.attr.suites)),
      deploy_jar=ctx.outputs.jar.path,
      cp=":".join([j.short_path for j in jars]))
  ctx.file_action(
      output=ctx.outputs.executable,
      content=content)

def _collect_comp_run_jars(ctx):
  compile_jars = set()
  runtime_jars = set()
  for target in ctx.attr.deps:
    if hasattr(target, "runtime_jar_files"):
      runtime_jars += target.runtime_jar_files
    if hasattr(target, "interface_jar_files"):
      compile_jars += target.interface_jar_files
    if hasattr(target, "java"):
      runtime_jars += target.java.transitive_runtime_deps
      #see JavaSkylarkApiProvider.java, this is just the compile-time deps
      compile_jars += target.java.transitive_deps
  return (compile_jars, runtime_jars)

def _scala_library_impl(ctx):
  (cjars, rjars) = _collect_comp_run_jars(ctx)
  _write_manifest(ctx)
  _compile(ctx, cjars, True)

  cjars += [ctx.outputs.ijar]
  rjars += [ctx.outputs.jar]
  runfiles = ctx.runfiles(
      files = list(rjars),
      collect_data = True)
  return struct(
      runtime_jar_files=rjars,
      interface_jar_files=cjars,
      runfiles=runfiles)

def _scala_macro_library_impl(ctx):
  (cjars, rjars) = _collect_comp_run_jars(ctx)
  _write_manifest(ctx)
  _compile(ctx, cjars, False)

  rjars += [ctx.outputs.jar]
  # macro code needs to be available at compiletime
  cjars += [ctx.outputs.jar]
  runfiles = ctx.runfiles(
      files = list(rjars),
      collect_data = True)
  return struct(
      runtime_jar_files=rjars,
      interface_jar_files=cjars,
      runfiles=runfiles)

# Common code shared by all scala binary implementations.
def _scala_binary_common(ctx, cjars, rjars):
  _write_manifest(ctx)
  _compile(ctx, cjars, False)

  runfiles = ctx.runfiles(
      files = list(rjars) + [ctx.outputs.executable] + [ctx.file._java] + ctx.files._jdk,
      collect_data = True)
  return struct(
      files=set([ctx.outputs.executable]),
      runfiles=runfiles)

def _scala_binary_impl(ctx):
  (cjars, rjars) = _collect_comp_run_jars(ctx)
  cjars += [ctx.file._scalareflect]
  rjars += [ctx.outputs.jar, ctx.file._scalalib, ctx.file._scalareflect]
  _write_launcher(ctx, rjars)
  return _scala_binary_common(ctx, cjars, rjars)

def _scala_test_impl(ctx):
  (cjars, rjars) = _collect_comp_run_jars(ctx)
  cjars += [ctx.file._scalareflect, ctx.file._scalatest, ctx.file._scalaxml]
  rjars += [ctx.outputs.jar, ctx.file._scalalib, ctx.file._scalareflect, ctx.file._scalatest, ctx.file._scalaxml]
  _write_test_launcher(ctx, rjars)
  return _scala_binary_common(ctx, cjars, rjars)

_implicit_deps = {
  "_ijar": attr.label(executable=True, default=Label("//tools/defaults:ijar"), single_file=True, allow_files=True),
  "_scalac": attr.label(executable=True, default=Label("@scala//:bin/scalac"), single_file=True, allow_files=True),
  "_scalalib": attr.label(default=Label("@scala//:lib/scala-library.jar"), single_file=True, allow_files=True),
  "_scalaxml": attr.label(default=Label("@scala//:lib/scala-xml_2.11-1.0.4.jar"), single_file=True, allow_files=True),
  "_scalasdk": attr.label(default=Label("@scala//:sdk"), allow_files=True),
  "_scalareflect": attr.label(default=Label("@scala//:lib/scala-reflect.jar"), single_file=True, allow_files=True),
  "_jar": attr.label(executable=True, default=Label("@bazel_tools//tools/jdk:jar"), single_file=True, allow_files=True),
  "_jdk": attr.label(default=Label("//tools/defaults:jdk"), allow_files=True),
}

# Common attributes reused across multiple rules.
_common_attrs = {
  "srcs": attr.label_list(
      allow_files=_scala_filetype,
      non_empty=True),
  "deps": attr.label_list(),
  "data": attr.label_list(allow_files=True, cfg=DATA_CFG),
  "resources": attr.label_list(allow_files=True),
  "scalacopts":attr.string_list(),
  "jvm_flags": attr.string_list(),
}

scala_library = rule(
  implementation=_scala_library_impl,
  attrs={
      "main_class": attr.string(),
      } + _implicit_deps + _common_attrs,
  outputs={
      "jar": "%{name}_deploy.jar",
      "ijar": "%{name}_ijar.jar",
      "manifest": "%{name}_MANIFEST.MF",
      },
)

scala_macro_library = rule(
  implementation=_scala_macro_library_impl,
  attrs={
      "main_class": attr.string(),
      "_scala-reflect": attr.label(default=Label("@scala//:lib/scala-reflect.jar"), single_file=True, allow_files=True),
      } + _implicit_deps + _common_attrs,
  outputs={
      "jar": "%{name}_deploy.jar",
      "manifest": "%{name}_MANIFEST.MF",
      },
)

scala_binary = rule(
  implementation=_scala_binary_impl,
  attrs={
      "main_class": attr.string(mandatory=True),
      "_java": attr.label(executable=True, default=Label("@bazel_tools//tools/jdk:java"), single_file=True, allow_files=True),
      } + _implicit_deps + _common_attrs,
  outputs={
      "jar": "%{name}_deploy.jar",
      "manifest": "%{name}_MANIFEST.MF",
      },
  executable=True,
)

scala_test = rule(
  implementation=_scala_test_impl,
  attrs={
      "main_class": attr.string(default="org.scalatest.tools.Runner"),
      "suites": attr.string_list(non_empty=True, mandatory=True),
      "_scalatest": attr.label(executable=True, default=Label("@scalatest//file"), single_file=True, allow_files=True),
      "_java": attr.label(executable=True, default=Label("@bazel_tools//tools/jdk:java"), single_file=True, allow_files=True),
      } + _implicit_deps + _common_attrs,
  outputs={
      "jar": "%{name}_deploy.jar",
      "manifest": "%{name}_MANIFEST.MF",
      },
  executable=True,
  test=True,
)

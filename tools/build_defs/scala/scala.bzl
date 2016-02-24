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

def _collect_jars(targets):
  """Compute the runtime and compile-time dependencies from the given targets"""
  compile_jars = set()  # not transitive
  runtime_jars = set()  # this is transitive
  for target in targets:
    found = False
    if hasattr(target, "scala"):
      compile_jars += [target.scala.outputs.ijar]
      compile_jars += target.scala.transitive_compile_exports
      runtime_jars += target.scala.transitive_runtime_deps
      runtime_jars += target.scala.transitive_runtime_exports
      found = True
    if hasattr(target, "java"):
      # see JavaSkylarkApiProvider.java, this is just the compile-time deps
      # this should be improved in bazel 0.1.5 to get outputs.ijar
      # compile_jars += [target.java.outputs.ijar]
      compile_jars += target.java.transitive_deps
      runtime_jars += target.java.transitive_runtime_deps
      found = True
    if not found:
      # support http_file pointed at a jar. http_jar uses ijar, which breaks scala macros
      runtime_jars += target.files
      compile_jars += target.files
  return struct(compiletime = compile_jars, runtime = runtime_jars)

def _lib(ctx, use_ijar):
  jars = _collect_jars(ctx.attr.deps)
  (cjars, rjars) = (jars.compiletime, jars.runtime)
  _write_manifest(ctx)
  _compile(ctx, cjars, use_ijar)

  rjars += [ctx.outputs.jar]
  rjars += _collect_jars(ctx.attr.runtime_deps).runtime

  ijar = None
  if use_ijar:
    ijar = ctx.outputs.ijar
  else:
    # macro code needs to be available at compile-time, so set ijar == jar
    ijar = ctx.outputs.jar

  texp = _collect_jars(ctx.attr.exports)
  scalaattr = struct(outputs = struct(ijar=ijar, class_jar=ctx.outputs.jar),
                     transitive_runtime_deps = rjars,
                     transitive_compile_exports = texp.compiletime,
                     transitive_runtime_exports = texp.runtime
                     )
  runfiles = ctx.runfiles(
      files = list(rjars),
      collect_data = True)
  return struct(
      scala = scalaattr,
      runfiles=runfiles)

def _scala_library_impl(ctx):
  return _lib(ctx, True)

def _scala_macro_library_impl(ctx):
  return _lib(ctx, False)  # don't build the ijar for macros

# Common code shared by all scala binary implementations.
def _scala_binary_common(ctx, cjars, rjars):
  _write_manifest(ctx)
  _compile(ctx, cjars, False)  # no need to build an ijar for an executable

  runfiles = ctx.runfiles(
      files = list(rjars) + [ctx.outputs.executable] + [ctx.file._java] + ctx.files._jdk,
      collect_data = True)
  return struct(
      files=set([ctx.outputs.executable]),
      runfiles=runfiles)

def _scala_binary_impl(ctx):
  jars = _collect_jars(ctx.attr.deps)
  (cjars, rjars) = (jars.compiletime, jars.runtime)
  cjars += [ctx.file._scalareflect]
  rjars += [ctx.outputs.jar, ctx.file._scalalib, ctx.file._scalareflect]
  rjars += _collect_jars(ctx.attr.runtime_deps).runtime
  _write_launcher(ctx, rjars)
  return _scala_binary_common(ctx, cjars, rjars)

def _scala_test_impl(ctx):
  jars = _collect_jars(ctx.attr.deps)
  (cjars, rjars) = (jars.compiletime, jars.runtime)
  cjars += [ctx.file._scalareflect, ctx.file._scalatest, ctx.file._scalaxml]
  rjars += [ctx.outputs.jar, ctx.file._scalalib, ctx.file._scalareflect, ctx.file._scalatest, ctx.file._scalaxml]
  rjars += _collect_jars(ctx.attr.runtime_deps).runtime
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
  "runtime_deps": attr.label_list(),
  "data": attr.label_list(allow_files=True, cfg=DATA_CFG),
  "resources": attr.label_list(allow_files=True),
  "scalacopts":attr.string_list(),
  "jvm_flags": attr.string_list(),
}

scala_library = rule(
  implementation=_scala_library_impl,
  attrs={
      "main_class": attr.string(),
      "exports": attr.label_list(allow_files=False),
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
      "exports": attr.label_list(allow_files=False),
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

SCALA_BUILD_FILE = """
# scala.BUILD
exports_files([
  "bin/scala",
  "bin/scalac",
  "bin/scaladoc",
  "lib/akka-actor_2.11-2.3.10.jar",
  "lib/config-1.2.1.jar",
  "lib/jline-2.12.1.jar",
  "lib/scala-actors-2.11.0.jar",
  "lib/scala-actors-migration_2.11-1.1.0.jar",
  "lib/scala-compiler.jar",
  "lib/scala-continuations-library_2.11-1.0.2.jar",
  "lib/scala-continuations-plugin_2.11.7-1.0.2.jar",
  "lib/scala-library.jar",
  "lib/scala-parser-comscala-2.11.7/binators_2.11-1.0.4.jar",
  "lib/scala-reflect.jar",
  "lib/scala-swing_2.11-1.0.2.jar",
  "lib/scala-xml_2.11-1.0.4.jar",
  "lib/scalap-2.11.7.jar",
])

filegroup(
    name = "sdk",
    srcs = glob(["**"]),
    visibility = ["//visibility:public"],
)
"""

def scala_repositories():
  native.new_http_archive(
    name = "scala",
    strip_prefix = "scala-2.11.7",
    sha256 = "ffe4196f13ee98a66cf54baffb0940d29432b2bd820bd0781a8316eec22926d0",
    url = "https://downloads.typesafe.com/scala/2.11.7/scala-2.11.7.tgz",
    build_file_content = SCALA_BUILD_FILE,
  )
  native.http_file(
    name = "scalatest",
    url = "https://oss.sonatype.org/content/groups/public/org/scalatest/scalatest_2.11/2.2.6/scalatest_2.11-2.2.6.jar",
    sha256 = "f198967436a5e7a69cfd182902adcfbcb9f2e41b349e1a5c8881a2407f615962",
  )

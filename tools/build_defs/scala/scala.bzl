# Copyright 2015 Google Inc. All rights reserved.
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

# TODO(bazel-team): Add local_repository to properly declare the dependency.
_scala_library_path = "/usr/share/java/scala-library.jar"
_scalac_path = "/usr/bin/scalac"

def _compile(ctx, jars):
  cmd = """
set -e
mkdir -p {out}_tmp
{scalac} -classpath "{jars}" $@ -d {out}_tmp
# Make jar file deterministic by setting the timestamp of files
touch -t 198001010000 $(find .)
jar cmf {manifest} {out} -C {out}_tmp .
"""
  cmd = cmd.format(
      scalac=_scalac_path,
      out=ctx.outputs.jar.path,
      manifest=ctx.outputs.manifest.path,
      jars=":".join([j.path for j in jars]))

  ctx.action(
      inputs=list(jars) + ctx.files.srcs + [ctx.outputs.manifest],
      outputs=[ctx.outputs.jar],
      command=cmd,
      progress_message="scala %s" % ctx.label,
      arguments=[f.path for f in ctx.files.srcs])


def _write_manifest(ctx):
  cp = "/usr/share/java/scala-library.jar"
  manifest = "Class-Path: %s\n" % cp
  if getattr(ctx.attr, "main_class", ""):
    manifest += "Main-Class: %s\n" % ctx.attr.main_class

  ctx.file_action(
      output = ctx.outputs.manifest,
      content = manifest)


def _write_launcher(ctx, jars):
  content = """#!/bin/bash
cd $0.runfiles
java -cp {cp} {name} "$@"
"""
  content = content.format(
      name=ctx.attr.main_class,
      deploy_jar=ctx.outputs.jar.path,
      cp=":".join([j.short_path for j in jars]))
  ctx.file_action(
      output=ctx.outputs.executable,
      content=content)


def _collect_jars(ctx):
  jars = set()
  for target in ctx.attr.deps:
    if hasattr(target, "jar_files"):
      jars += target.jar_files
    elif hasattr(target, "java"):
      jars += target.java.transitive_runtime_deps
  return jars


def _scala_library_impl(ctx):
  jars = _collect_jars(ctx)
  _write_manifest(ctx)
  _compile(ctx, jars)

  jars += [ctx.outputs.jar]
  runfiles = ctx.runfiles(
      files = list(jars),
      collect_data = True)
  return struct(
      files=jars,
      jar_files=jars,
      runfiles=runfiles)


def _scala_binary_impl(ctx):
  jars = _collect_jars(ctx)
  _write_manifest(ctx)
  _compile(ctx, jars)

  jars += [ctx.outputs.jar]
  _write_launcher(ctx, jars)

  runfiles = ctx.runfiles(
      files = list(jars) + [ctx.outputs.executable],
      collect_data = True)
  return struct(
      files=set([ctx.outputs.executable]),
      runfiles=runfiles)


scala_library = rule(
  implementation=_scala_library_impl,
  attrs={
      "main_class": attr.string(),
      "srcs": attr.label_list(
          allow_files=_scala_filetype,
          non_empty=True),
      "deps": attr.label_list(),
      "data": attr.label_list(allow_files=True, cfg=DATA_CFG),
      },
  outputs={
      "jar": "%{name}_deploy.jar",
      "manifest": "%{name}_MANIFEST.MF",
      },
)

scala_binary = rule(
  implementation=_scala_binary_impl,
  attrs={
      "main_class": attr.string(mandatory=True),
      "srcs": attr.label_list(
          allow_files=_scala_filetype,
          non_empty=True),
      "deps": attr.label_list(),
      "data": attr.label_list(allow_files=True, cfg=DATA_CFG),
      },
  outputs={
      "jar": "%{name}_deploy.jar",
      "manifest": "%{name}_MANIFEST.MF",
      },
  executable=True,
)

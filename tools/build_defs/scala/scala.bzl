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
mkdir -p {out}_tmp
{scalac} -classpath "{jars}" $@ -d {out}_tmp &&
# Make jar file deterministic by setting the timestamp of files
touch -t 198001010000 $(find .)
jar cmf {manifest} {out} -C {out}_tmp .
"""
  cmd = cmd.format(
      scalac=_scalac_path,
      out=ctx.outputs.jar.path,
      manifest=ctx.outputs.manifest.path,
      jars=':'.join([j.path for j in jars]))

  ctx.action(
      inputs=list(jars) + ctx.files.srcs + [ctx.outputs.manifest],
      outputs=[ctx.outputs.jar],
      command=cmd,
      progress_message="scala %s" % ctx.label,
      arguments=[f.path for f in ctx.files.srcs])


def _write_manifest(ctx):
  manifest = """Main-Class: {main_class}
Class-Path: {cp}
"""
  manifest = manifest.format(
      main_class=ctx.attr.main_class,
      cp=_scala_library_path)

  ctx.file_action(
      output = ctx.outputs.manifest,
      content = manifest)


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

  all_jars = jars + [ctx.outputs.jar]

  runfiles = ctx.runfiles(
      files = list(all_jars) + [ctx.outputs.jar],
      collect_data = True)
  return struct(
      files=all_jars,
      jar_files=all_jars,
      runfiles=runfiles)


scala_library = rule(
  implementation=_scala_library_impl,
  attrs={
      "main_class": attr.string(mandatory=True),
      "srcs": attr.label_list(allow_files=_scala_filetype),
      "deps": attr.label_list(),
      "data": attr.label_list(allow_files=True, cfg=DATA_CFG),
      },
  outputs={
      "jar": "%{name}_deploy.jar",
      "manifest": "%{name}_MANIFEST.MF",
      },
)

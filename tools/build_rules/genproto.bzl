# Copyright 2014 Google Inc. All rights reserved.
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

# This is a quick and dirty rule to make Bazel compile itself.  It
# only supports Java.


# TODO(bazel-team): unify the OSS Java rules and load from another
# file.
jar_filetype = FileType([".jar"])

proto_filetype = FileType([".proto"])

def genproto_impl(ctx):
  src = ctx.file.src
  proto_compiler = ctx.file._proto_compiler
  proto_dep = ctx.file._proto_dep
  class_jar = ctx.outputs.java
  proto_output = class_jar.path + ".proto_output"
  build_output = class_jar.path + ".build_output"

  inputs = [src, proto_dep, proto_compiler]
  proto_compiler_path = proto_compiler.path

  javapath = "tools/jdk/jdk/bin/"
  cmd = ("set -e;" +
         "rm -rf " + proto_output + ";" +
         "mkdir " + proto_output + ";" +
         "rm -rf " + build_output + ";" +
         "mkdir " + build_output + "\n" +
         proto_compiler_path + " --java_out=" +
         proto_output +" " + src.path + "\n" +
         "JAVA_FILES=$(find " + proto_output + " -name '*.java')\n" +
         javapath + "javac" + " -classpath " + proto_dep.path +
         " ${JAVA_FILES} -d " + build_output + "\n" +
         javapath + "jar cf " + class_jar.path + "  -C " + build_output + " .\n")
  ctx.action(
      inputs = inputs,
      outputs = [class_jar],
      mnemonic = 'CompileProtos',
      command = cmd,
      use_default_shell_env = True)

  return struct(compile_time_jars = set([class_jar]),
                runtime_jars = set([class_jar, proto_dep], order="link"))


genproto = rule(genproto_impl,
   # There should be a flag like gen_java, and only generate the jar if it's
   # set. Skylark needs a bit of improvement first (concat structs).
   attrs = {
       "src": attr.label(allow_files=proto_filetype, single_file=True),
       # TODO(bazel-team): this should be a hidden attribute with a default
       # value, but Skylark needs to support select first.
       "_proto_compiler": attr.label(
           default=Label("//third_party:protoc"),
           allow_files=True,
           single_file=True),
       "_proto_dep": attr.label(
           default=Label("//third_party:protobuf"),
           single_file=True,
           allow_files=jar_filetype,
           ),
   },
   outputs = {"java": "lib%{name}.jar"},
)

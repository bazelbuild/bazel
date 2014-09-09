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

java_filetype = filetype([".java"])
jar_filetype = filetype([".jar"])

# This is a quick and dirty rule to make Bazel compile itself. It's not
# production ready.

def java_library_impl(ctx):
  class_jar = ctx.outputs["class_jar"]
  manifest = ctx.outputs["manifest"]
  jars = jar_filetype.filter(ctx.files("deps", "TARGET"))
  build_output = Files.exec_path(class_jar) + ".build_output"
  java_home = ctx.attr.java_home
  main_class = ctx.attr.main_class
  srcdirs = ctx.attr.srcdirs
  sources = ctx.files("srcs", "TARGET")

  ctx.create_action(
    inputs = [],
    outputs = [manifest],
    mnemonic = 'manifest',
    command = ("echo 'Main-Class: " + main_class + "' > " +
              Files.exec_path(manifest)),
    use_default_shell_env = True)

  sources_param_file = ctx.param_file(
      ctx.configuration.bin_dir, class_jar, "-2.params")
  ctx.create_file_action(
      output = sources_param_file,
      content = Files.join_exec_paths("\n", sources),
      executable = False)

  # Cleaning build output directory
  cmd = "set -e;rm -rf " + build_output + ";mkdir " + build_output + "\n"
  # Java compilation
  cmd += ("/usr/bin/javac -classpath " +
         Files.join_exec_paths(":", jars) + " -sourcepath " +
         ":".join(srcdirs) + " -d " + build_output + " @" +
         Files.exec_path(sources_param_file) + "\n")

  # TODO(bazel-team): this deploy jar action should be only in binaries
  for jar in jars:
    cmd += "unzip -qn " + Files.exec_path(jar) + " -d " + build_output + "\n"
  cmd += ("/usr/bin/jar cmf " + Files.exec_path(manifest) + " " +
         Files.exec_path(class_jar) + " -C " + build_output + " .\n")

  ctx.create_action(
    inputs = jars + [manifest, sources_param_file],
    outputs = [class_jar],
    mnemonic='Javac',
    command=cmd,
    use_default_shell_env=True)

  return struct(files_to_build = nset("STABLE_ORDER", [class_jar, manifest]))


java_library = rule(java_library_impl,
    attr = {
       "srcs": Attr.label_list(file_types=java_filetype),
       "deps": Attr.label_list(file_types=NO_FILE),
       "java_home": Attr.string(),
       "srcdirs": Attr.string_list(),
       "main_class": Attr.string(),
   },
   implicit_outputs={
       "class_jar": "lib%{name}.jar",
       "manifest": "%{name}_MANIFEST.MF"
   }
)

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
  class_jar = ctx.outputs.class_jar
  # TODO(bazel-team): use simple set here, no need for nset
  compile_time_jars = nset("STABLE_ORDER")
  runtime_jars = nset("LINK_ORDER")
  for dep in ctx.targets("deps", "TARGET"):
    compile_time_jars += [dep.compile_time_jar]
    runtime_jars += dep.runtime_jars

  jars = jar_filetype.filter(ctx.files("jars", "TARGET"))
  compile_time_jars += jars
  runtime_jars += jars
  compile_time_jar_list = list(compile_time_jars)

  build_output = class_jar.path + ".build_output"
  sources = ctx.files("srcs", "TARGET")

  sources_param_file = ctx.new_file(
      ctx.configuration.bin_dir, class_jar, "-2.params")
  ctx.file_action(
      output = sources_param_file,
      content = files.join_exec_paths("\n", sources),
      executable = False)

  # Cleaning build output directory
  cmd = "set -e;rm -rf " + build_output + ";mkdir " + build_output + "\n"
  cmd += "/usr/bin/javac"
  if compile_time_jar_list:
    cmd += " -classpath " + files.join_exec_paths(":", compile_time_jar_list)
  cmd += " -d " + build_output + " @" + sources_param_file.path + "\n"
  cmd += ("/usr/bin/jar cf " + class_jar.path + " -C " + build_output + " .\n" +
         "touch " + build_output + "\n")

  ctx.action(
    inputs = sources + compile_time_jar_list + [sources_param_file],
    outputs = [class_jar],
    mnemonic='Javac',
    command=cmd,
    use_default_shell_env=True)

  runfiles = ctx.runfiles([DATA])

  return struct(files_to_build = nset("STABLE_ORDER", [class_jar]),
                compile_time_jar = class_jar,
                runtime_jars = runtime_jars + [class_jar],
                runfiles = runfiles)


java_library = rule(java_library_impl,
    attr = {
       "data": attr.label_list(file_types=ANY_FILE, rule_classes=NO_RULE,
          cfg=DATA_CFG),
       "srcs": attr.label_list(file_types=java_filetype),
       "jars": attr.label_list(file_types=jar_filetype),
       "deps": attr.label_list(file_types=NO_FILE,
           providers = ["compile_time_jar", "runtime_jars"]),
   },
   outputs={
       "class_jar": "lib%{name}.jar",
   }
)

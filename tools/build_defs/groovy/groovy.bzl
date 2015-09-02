# Copyright 2015 Erik Kuefler. All rights reserved.
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


def _groovy_jar_impl(ctx):
  """Creates a .jar file from Groovy sources. Users should rely on
  groovy_library instead of using this rule directly.
  """
  class_jar = ctx.outputs.class_jar
  build_output = class_jar.path + ".build_output"

  # Extract all transitive dependencies
  # TODO(bazel-team): get transitive dependencies from other groovy libraries
  all_deps = set(ctx.files.deps)
  for this_dep in ctx.attr.deps:
    if hasattr(this_dep, "java"):
      all_deps += this_dep.java.transitive_runtime_deps

  # Compile all files in srcs with groovyc
  cmd = "rm -rf %s; mkdir -p %s\n" % (build_output, build_output)
  cmd += "%s %s -d %s %s\n" % (
      ctx.file._groovyc.short_path,
      "-cp " + ":".join([dep.path for dep in all_deps]) if len(all_deps) != 0 else "",
      build_output,
      " ".join([src.path for src in ctx.files.srcs]),
  )

  # Jar them together to produce a single output. To make this work we have
  # to cd into the output directory, run find to locate all of the generated
  # class files, pass the result to cut to trim the leading "./", then pass
  # the resulting paths to the zipper.
  cmd += "root=`pwd`\n"
  cmd += "cd %s; $root/%s c ../%s `find . -name '*.class' | cut -c 3-`\n" % (
      build_output,
      ctx.executable._jar.path,
      class_jar.basename,
  )
  cmd += "cd $root\n"

  # Clean up temporary output
  cmd += "rm -rf %s" % build_output

  # Execute the command
  ctx.action(
      inputs = ctx.files.srcs + ctx.files.deps + ctx.files._jar,
      outputs = [class_jar],
      mnemonic = "Groovyc",
      command = "set -e;" + cmd,
      use_default_shell_env = True,
  )

_groovy_jar = rule(
    implementation = _groovy_jar_impl,
    attrs = {
        "srcs": attr.label_list(
            mandatory=False,
            allow_files=FileType([".groovy"])),
        "deps": attr.label_list(
            mandatory=False,
            allow_files=FileType([".jar"])),
        "_groovyc": attr.label(
            default=Label("//external:groovyc"),
            single_file=True),
        "_jar": attr.label(
            default=Label("//third_party/ijar:zipper"),
            executable=True,
            single_file=True),
        },
    outputs = {
        "class_jar": "lib%{name}.jar",
        },
)

def groovy_library(name, srcs=[], deps=[], **kwargs):
  """Rule analagous to java_library that accepts .groovy sources instead of
  .java sources. The result is wrapped in a java_import so that java rules may
  depend on it.
  """
  _groovy_jar(
      name = name + "-impl",
      srcs = srcs,
      deps = deps,
  )
  native.java_import(
      name = name,
      jars = [name + "-impl"],
      **kwargs
  )


def groovy_and_java_library(name, srcs=[], deps=[], **kwargs):
  """Accepts .groovy and .java srcs to create a groovy_library and a
  java_library. The groovy_library will depend on the java_library, so the
  Groovy code may reference the Java code but not vice-versa.
  """
  groovy_deps = deps
  jars = []

  # Put all .java sources in a java_library
  java_srcs = [src for src in srcs if src.endswith(".java")]
  if java_srcs:
    native.java_library(
        name = name + "-java",
        srcs = java_srcs,
        deps = deps,
    )
    groovy_deps += [name + "-java"]
    jars += ["lib"  + name + "-java.jar"]

  # Put all .groovy sources in a groovy_library depending on the java_library
  groovy_srcs = [src for src in srcs if src.endswith(".groovy")]
  if groovy_srcs:
    _groovy_jar(
        name = name + "-groovy",
        srcs = groovy_srcs,
        deps = groovy_deps,
    )
    jars += ["lib" + name + "-groovy.jar"]

  # Output a java_import combining both libraries
  native.java_import(
      name = name,
      jars = jars,
      **kwargs
  )


def groovy_binary(name, main_class, srcs=[], deps=[], **kwargs):
  """Rule analagous to java_binary that accepts .groovy sources instead of .java
  sources.
  """
  all_deps = deps
  if srcs:
    groovy_library(
        name = name + "-lib",
        srcs = srcs,
        deps = deps,
    )
    all_deps += [name + "-lib"]

  native.java_binary(
      name = name,
      main_class = main_class,
      runtime_deps = all_deps,
      **kwargs
  )

def path_to_class(path):
  if path.startswith("src/test/groovy/"):
    return path[len("src/test/groovy/") : path.index(".groovy")].replace('/', '.')
  elif path.startswith("src/test/java/"):
    return path[len("src/test/java/") : path.index(".groovy")].replace('/', '.')
  else:
    fail("groovy_test sources must be under src/test/java or src/test/groovy")

def groovy_test_impl(ctx):
  # Extract all transitive dependencies
  all_deps = set(ctx.files.deps + ctx.files._implicit_deps)
  for this_dep in ctx.attr.deps:
    if hasattr(this_dep, 'java'):
      all_deps += this_dep.java.transitive_runtime_deps

  # Infer a class name from each src file
  classes = [path_to_class(src.path) for src in ctx.files.srcs]

  # Write a file that executes JUnit on the inferred classes
  cmd = "%s %s -cp %s org.junit.runner.JUnitCore %s\n" % (
      ctx.executable._java.path,
    " ".join(ctx.attr.jvm_flags),
    ":".join([dep.short_path for dep in all_deps]),
    " ".join(classes),
  )
  ctx.file_action(
    output = ctx.outputs.executable,
    content = cmd
  )

  # Return all dependencies needed to run the tests
  return struct(
    runfiles=ctx.runfiles(files=list(all_deps) + ctx.files._java),
  )

groovy_test = rule(
  implementation = groovy_test_impl,
  attrs = {
    "srcs": attr.label_list(mandatory=True, allow_files=FileType([".groovy"])),
    "deps": attr.label_list(allow_files=FileType([".jar"])),
    "jvm_flags": attr.string_list(),
    "_java": attr.label(
      default=Label("@local-jdk//:java"),
      executable=True,
      single_file=True),
    "_implicit_deps": attr.label_list(default=[
      Label("//external:groovy"),
      Label("//external:junit"),
    ]),
  },
  test = True,
)

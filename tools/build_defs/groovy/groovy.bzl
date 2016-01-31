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

  # Set up the output directory and set JAVA_HOME
  cmd = "rm -rf %s\n" % build_output
  cmd += "mkdir -p %s\n" % build_output
  cmd += "export JAVA_HOME=external/local_jdk\n"

  # Set GROOVY_HOME by scanning through the groovy SDK to find the license file,
  # which should be at the root of the SDK.
  for file in ctx.files._groovysdk:
    if file.basename == "CLI-LICENSE.txt":
      cmd += "export GROOVY_HOME=%s\n" % file.dirname
      break

  # Compile all files in srcs with groovyc
  cmd += "$GROOVY_HOME/bin/groovyc %s -d %s %s\n" % (
      "-cp " + ":".join([dep.path for dep in all_deps]) if len(all_deps) != 0 else "",
      build_output,
      " ".join([src.path for src in ctx.files.srcs]),
  )

  # Jar them together to produce a single output. To make this work we have
  # to cd into the output directory, run find to locate all of the generated
  # class files, pass the result to cut to trim the leading "./", then pass
  # the resulting paths to the zipper.
  cmd += "root=`pwd`\n"
  cmd += "cd %s; $root/%s Cc ../%s `find . -name '*.class' | cut -c 3-`\n" % (
      build_output,
      ctx.executable._zipper.path,
      class_jar.basename,
  )
  cmd += "cd $root\n"

  # Clean up temporary output
  cmd += "rm -rf %s" % build_output

  # Execute the command
  ctx.action(
      inputs = (
          ctx.files.srcs
          + list(all_deps)
          + ctx.files._groovysdk
          + ctx.files._jdk
          + ctx.files._zipper),
      outputs = [class_jar],
      mnemonic = "Groovyc",
      command = "set -e;" + cmd,
      use_default_shell_env = True,
  )

_groovy_jar = rule(
    attrs = {
        "srcs": attr.label_list(
            non_empty = True,
            allow_files = FileType([".groovy"]),
        ),
        "deps": attr.label_list(
            mandatory = False,
            allow_files = FileType([".jar"]),
        ),
        "_groovysdk": attr.label(
            default = Label("//external:groovy-sdk"),
        ),
        "_jdk": attr.label(
            default = Label("//tools/defaults:jdk"),
        ),
        "_zipper": attr.label(
            default = Label("//third_party/ijar:zipper"),
            executable = True,
            single_file = True,
        ),
    },
    outputs = {
        "class_jar": "lib%{name}.jar",
    },
    implementation = _groovy_jar_impl,
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
      deps = deps,
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
      deps = deps,
      **kwargs
  )

def groovy_binary(name, main_class, srcs=[], deps=[], **kwargs):
  """Rule analagous to java_binary that accepts .groovy sources instead of .java
  sources.
  """
  all_deps = deps + ["//external:groovy"]
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

def _groovy_test_impl(ctx):
  # Collect jars from the Groovy sdk
  groovy_sdk_jars = [file
      for file in ctx.files._groovysdk
      if file.basename.endswith(".jar")
  ]

  # Extract all transitive dependencies
  all_deps = set(ctx.files.deps + ctx.files._implicit_deps + groovy_sdk_jars)
  for this_dep in ctx.attr.deps:
    if hasattr(this_dep, 'java'):
      all_deps += this_dep.java.transitive_runtime_deps

  # Infer a class name from each src file
  classes = [path_to_class(src.path) for src in ctx.files.srcs]

  # Write a file that executes JUnit on the inferred classes
  cmd = "external/local-jdk/bin/java %s -cp %s org.junit.runner.JUnitCore %s\n" % (
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
    runfiles=ctx.runfiles(files=list(all_deps) + ctx.files.data + ctx.files._jdk),
  )

_groovy_test = rule(
    attrs = {
        "srcs": attr.label_list(
            mandatory = True,
            allow_files = FileType([".groovy"]),
        ),
        "deps": attr.label_list(allow_files = FileType([".jar"])),
        "data": attr.label_list(allow_files = True),
        "jvm_flags": attr.string_list(),
        "_groovysdk": attr.label(
            default = Label("//external:groovy-sdk"),
        ),
        "_jdk": attr.label(
            default = Label("//tools/defaults:jdk"),
        ),
        "_implicit_deps": attr.label_list(default = [
            Label("//external:junit"),
        ]),
    },
    test = True,
    implementation = _groovy_test_impl,
)

def groovy_test(
    name,
    deps=[],
    srcs=[],
    data=[],
    resources=[],
    jvm_flags=[],
    size="medium",
    tags=[]):
  # Create an extra jar to hold the resource files if any were specified
  all_deps = deps
  if resources:
    native.java_library(
      name = name + "-resources",
      resources = resources,
    )
    all_deps += [name + "-resources"]

  _groovy_test(
    name = name,
    size = size,
    tags = tags,
    srcs = srcs,
    deps = all_deps,
    data = data,
    jvm_flags = jvm_flags,
  )

def groovy_junit_test(
    name,
    tests,
    deps=[],
    groovy_srcs=[],
    java_srcs=[],
    data=[],
    resources=[],
    jvm_flags=[],
    size="small",
    tags=[]):
  groovy_lib_deps = deps + ["//external:junit"]
  test_deps = deps + ["//external:junit"]

  if len(tests) == 0:
    fail("Must provide at least one file in tests")

  # Put all Java sources into a Java library
  if java_srcs:
    native.java_library(
      name = name + "-javalib",
      srcs = java_srcs,
      deps = deps + ["//external:junit"],
    )
    groovy_lib_deps += [name + "-javalib"]
    test_deps += [name + "-javalib"]

  # Put all tests and Groovy sources into a Groovy library
  groovy_library(
    name = name + "-groovylib",
    srcs = tests + groovy_srcs,
    deps = groovy_lib_deps,
  )
  test_deps += [name + "-groovylib"]

  # Create a groovy test
  groovy_test(
    name = name,
    deps = test_deps,
    srcs = tests,
    data = data,
    resources = resources,
    jvm_flags = jvm_flags,
    size = size,
    tags = tags,
  )

def spock_test(
    name,
    specs,
    deps=[],
    groovy_srcs=[],
    java_srcs=[],
    data=[],
    resources=[],
    jvm_flags=[],
    size="small",
    tags=[]):
  groovy_lib_deps = deps + [
    "//external:junit",
    "//external:spock",
  ]
  test_deps = deps + [
    "//external:junit",
    "//external:spock",
  ]

  if len(specs) == 0:
    fail("Must provide at least one file in specs")

  # Put all Java sources into a Java library
  if java_srcs:
    native.java_library(
      name = name + "-javalib",
      srcs = java_srcs,
      deps = deps + [
        "//external:junit",
        "//external:spock",
      ],
    )
    groovy_lib_deps += [name + "-javalib"]
    test_deps += [name + "-javalib"]

  # Put all specs and Groovy sources into a Groovy library
  groovy_library(
    name = name + "-groovylib",
    srcs = specs + groovy_srcs,
    deps = groovy_lib_deps,
  )
  test_deps += [name + "-groovylib"]

  # Create a groovy test
  groovy_test(
    name = name,
    deps = test_deps,
    srcs = specs,
    data = data,
    resources = resources,
    jvm_flags = jvm_flags,
    size = size,
    tags = tags,
  )

def groovy_repositories():
  native.new_http_archive(
    name = "groovy-sdk-artifact",
    url = "http://dl.bintray.com/groovy/maven/apache-groovy-binary-2.4.4.zip",
    sha256 = "a7cc1e5315a14ea38db1b2b9ce0792e35174161141a6a3e2ef49b7b2788c258c",
    build_file = "groovy.BUILD",
  )
  native.bind(
    name = "groovy-sdk",
    actual = "@groovy-sdk-artifact//:sdk",
  )
  native.bind(
    name = "groovy",
    actual = "@groovy-sdk-artifact//:groovy",
  )

  native.maven_jar(
    name = "junit-artifact",
    artifact = "junit:junit:4.12",
  )
  native.bind(
    name = "junit",
    actual = "@junit-artifact//jar",
  )

  native.maven_jar(
    name = "spock-artifact",
    artifact = "org.spockframework:spock-core:0.7-groovy-2.0",
  )
  native.bind(
    name = "spock",
    actual = "@spock-artifact//jar",
  )

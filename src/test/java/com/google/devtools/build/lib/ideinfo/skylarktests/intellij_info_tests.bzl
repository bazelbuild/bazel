# Copyright 2016 The Bazel Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http:#www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Tests for intellij_info.bzl

load(":skylarktests/testing.bzl",
     "start_test",
     "end_test",
     "fail_test",
     "assert_equals",
     "assert_contains_all",
     "assert_true")


load(":intellij_info.bzl", "intellij_info_test_aspect")

def test(impl):
    return rule(impl,
        attrs = {
            'targets' : attr.label_list(aspects = [intellij_info_test_aspect]),
        },
        test = True,
    )

def _source_paths(env, artifact_locations):
    for f in artifact_locations:
        assert_true(env, f.is_source, '%s is not source' % f)
    return [f.relative_path for f in artifact_locations]

def _jar_string(library_artifact, name):
    if hasattr(library_artifact, name):
      return "<%s:%s>" % (name, getattr(library_artifact, name).relative_path)
    else:
      return ""

def _library_artifact_string(env, library_artifact):
    return _jar_string(library_artifact, 'jar') + \
           _jar_string(library_artifact, 'interface_jar') + \
           _jar_string(library_artifact, 'source_jar')

def _jar_expected_string(base, jar, interface_jar, source_jar):
    s = ""
    if jar:
        s += "<jar:%s>" % (base + "/" + jar)
    if interface_jar:
        s += "<interface_jar:%s>" % (base + "/" + interface_jar)
    if source_jar:
        s += "<source_jar:%s>" % (base + "/" + source_jar)
    return s

################################################

def _test_simple_java_library(ctx):
    env = start_test(ctx)
    infos = ctx.attr.targets[0].intellij_infos
    info = infos[str(ctx.label.relative(":simple1"))]
    if not info:
        fail_test(env, "info not found")
        end_test(ctx, env)
        return
    assert_equals(env,
            ctx.label.package + "/BUILD",
            info.build_file_artifact_location.relative_path)

    assert_equals(env,
                True,
                info.build_file_artifact_location.is_source)

    assert_equals(env, "java_library", info.kind_string)

    assert_equals(env,
            [ctx.label.package + "/skylarktests/testfiles/Simple1.java"],
            _source_paths(env, info.java_rule_ide_info.sources))

    assert_equals(env,
            [_jar_expected_string(ctx.label.package,
                                 "libsimple1.jar", "libsimple1-ijar.jar", "libsimple1-src.jar")],
            [_library_artifact_string(env, a) for a in info.java_rule_ide_info.jars])

    assert_equals(env,
            ctx.label.package + "/libsimple1.jdeps",
            info.java_rule_ide_info.jdeps.relative_path)

    end_test(env)

test_simple_java_library_rule_test = test(_test_simple_java_library)

def test_simple_java_library():
    native.java_library(name = "simple1", srcs = ["skylarktests/testfiles/Simple1.java"])
    test_simple_java_library_rule_test(name = "test_simple_java_library",
        targets = [":simple1"]
    )

################################################
def _test_java_library_with_dependencies(ctx):
    env = start_test(ctx)
    infos = ctx.attr.targets[0].intellij_infos
    info_simple = infos[str(ctx.label.relative(":simple2"))]
    info_complex = infos[str(ctx.label.relative(":complex2"))]
    assert_equals(env,
            [ctx.label.package + "/skylarktests/testfiles/Complex2.java"],
            _source_paths(env, info_complex.java_rule_ide_info.sources))
    assert_contains_all(env,
                        [str(ctx.label.relative(":simple2"))],
                        info_complex.dependencies)
    end_test(env)

test_java_library_with_dependencies_rule_test = test(_test_java_library_with_dependencies)

def test_java_library_with_dependencies():
    native.java_library(name = "simple2", srcs = ["skylarktests/testfiles/Simple2.java"])
    native.java_library(name = "complex2",
                        srcs = ["skylarktests/testfiles/Complex2.java"],
                        deps = [":simple2"])
    test_java_library_with_dependencies_rule_test(name = "test_java_library_with_dependencies",
        targets = [":complex2"]
    )

def skylark_tests():
  test_simple_java_library()
  test_java_library_with_dependencies()

  native.test_suite(name = "skylark_tests",
                    tests = [":test_simple_java_library",
                             ":test_java_library_with_dependencies"])


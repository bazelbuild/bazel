// Copyright 2011 The Bazel Authors. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package com.google.devtools.build.lib.analysis.test;

import static com.google.common.truth.Truth.assertThat;
import static com.google.devtools.build.lib.actions.util.ActionsTestUtil.prettyArtifactNames;

import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for collection of instrumented files, {@link InstrumentedFilesCollector}. */
@RunWith(JUnit4.class)
public class InstrumentedFilesCollectorTest extends BuildViewTestCase {

  @Before
  public final void initializeConfiguration() throws Exception {
    useConfiguration("--collect_code_coverage");
  }

  @Test
  public void testCollectsSourceFiles_pythonTest() throws Exception {
    analysisMock.pySupport().setup(mockToolsConfig);
    useConfiguration("--collect_code_coverage");
    scratch.file(
        "foo/BUILD",
        """
        load("@rules_python//python:py_library.bzl", "py_library")
        load("@rules_python//python:py_test.bzl", "py_test")

        py_library(
            name = "foo",
            srcs = ["foo.py"],
            data = [
                "foo.js",
                "foo_test.js",
            ],
        )

        py_test(
            name = "foo_test",
            srcs = ["foo_test.py"],
            data = [
                "bar.js",
                "bar_test.js",
            ],
            deps = [":foo"],
        )
        """);

    InstrumentedFilesInfo target = getInstrumentedFilesProvider("//foo:foo_test");

    assertThat(prettyArtifactNames(target.getInstrumentedFiles())).containsExactly("foo/foo.py");
    assertThat(target.getInstrumentationMetadataFiles().toList()).isEmpty();
  }

  @Test
  public void testCollectsSourceFiles_ccLibraryTargets() throws Exception {
    scratch.file(
        "foo/BUILD",
        """
        load("@rules_cc//cc:cc_library.bzl", "cc_library")
        load("@rules_cc//cc:cc_test.bzl", "cc_test")

        cc_library(
            name = "impl",
            srcs = [
                "impl.cc",
                "impl.h",
            ],
            data = [
                "foo.js",
                "foo_test.js",
            ],
        )

        cc_test(
            name = "test",
            srcs = ["test.cc"],
            deps = [":impl"],
        )
        """);

    useConfiguration("--collect_code_coverage", "--instrument_test_targets");

    InstrumentedFilesInfo target = getInstrumentedFilesProvider("//foo:test");

    assertThat(prettyArtifactNames(target.getInstrumentedFiles()))
        .containsExactly("foo/impl.cc", "foo/impl.h", "foo/test.cc");
    assertThat(prettyArtifactNames(target.getInstrumentationMetadataFiles()))
        .containsExactly("foo/_objs/test/test.gcno", "foo/_objs/impl/impl.gcno");
  }

  @Test
  public void testInstrumentTestTargets() throws Exception {
    scratch.file(
        "foo/BUILD",
        """
        load("@rules_cc//cc:cc_library.bzl", "cc_library")
        load("@rules_cc//cc:cc_test.bzl", "cc_test")

        cc_library(
            name = "impl",
            srcs = ["impl.cc"],
        )

        cc_test(
            name = "test",
            srcs = ["test.cc"],
            deps = [":impl"],
        )
        """);

    // Test targets matched by instrumentation filter are included
    useConfiguration(
        "--collect_code_coverage", "--instrumentation_filter=//foo", "--instrument_test_targets");

    InstrumentedFilesInfo target = getInstrumentedFilesProvider("//foo:test");

    assertThat(prettyArtifactNames(target.getInstrumentedFiles()))
        .containsExactly("foo/impl.cc", "foo/test.cc");
  }

  @Test
  public void testNoInstrumentTestTargets() throws Exception {
    scratch.file(
        "foo/BUILD",
        """
        load("@rules_cc//cc:cc_library.bzl", "cc_library")
        load("@rules_cc//cc:cc_test.bzl", "cc_test")

        cc_library(
            name = "impl",
            srcs = ["impl.cc"],
        )

        cc_test(
            name = "test",
            srcs = ["test.cc"],
            deps = [":impl"],
        )
        """);

    // Test targets that would be matched by instrumentation filter are still excluded
    useConfiguration("--collect_code_coverage", "--instrumentation_filter=//foo");

    InstrumentedFilesInfo target = getInstrumentedFilesProvider("//foo:test");

    assertThat(prettyArtifactNames(target.getInstrumentedFiles())).containsExactly("foo/impl.cc");
  }

  @Test
  public void testCollectsSourceFiles_ccBinaryInGenrule() throws Exception {
    useConfiguration("--collect_code_coverage");
    scratch.file(
        "foo/BUILD",
        """
        load("@rules_cc//cc:cc_binary.bzl", "cc_binary")

        cc_binary(
            name = "bin",
            srcs = [
                "bin.cc",
                "bin.h",
            ],
            data = [
                "foo.js",
                "foo_test.js",
            ],
        )

        genrule(
            name = "gen",
            srcs = [":bin"],
            outs = ["gen"],
            cmd = " ...",
        )
        """);
    InstrumentedFilesInfo target = getInstrumentedFilesProvider("//foo:gen");

    // srcs is a source attribute, so stuff from ctx.files.srcs should be included if
    // :gen is matched by --instrumentation_filter. Transitive source files for stuff in srcs
    // (like bin.cc and bin.h) and associated metadata files should not be included.
    // cc_binary doesn't provide source files as outputs, so it doesn't configure
    // FileProvider.
    assertThat(prettyArtifactNames(target.getInstrumentedFiles())).isEmpty();
    assertThat(prettyArtifactNames(target.getInstrumentationMetadataFiles())).isEmpty();
  }

  @Test
  public void testCollectsSourceFiles_ccBinaryInFilegroup() throws Exception {
    useConfiguration("--collect_code_coverage");
    scratch.file(
        "foo/BUILD",
        """
        load("@rules_cc//cc:cc_binary.bzl", "cc_binary")

        cc_binary(
            name = "bin",
            srcs = [
                "bin.cc",
                "bin.h",
            ],
            data = [
                "foo.js",
                "foo_test.js",
            ],
        )

        filegroup(
            name = "fg",
            srcs = [":bin"],
        )
        """);
    InstrumentedFilesInfo target = getInstrumentedFilesProvider("//foo:fg");

    // For filegroup, everything is a dependency attribute, since we don't know whether it's
    // used by something that provides runtime dependencies or not. If it's consumed via a
    // dependency attribute for its conusumer, its transitive dependencies should be considered
    // for coverage. (Nothing on filegroup is a source attribute because it _processes_ no
    // sources itself.)
    assertThat(prettyArtifactNames(target.getInstrumentedFiles()))
        .containsExactly("foo/bin.cc", "foo/bin.h");
    assertThat(prettyArtifactNames(target.getInstrumentationMetadataFiles()))
        .containsExactly("foo/_objs/bin/bin.gcno");
  }

  @Test
  public void testGeneratedFilesAreNotCollected() throws Exception {
    scratch.file(
        "foo/BUILD",
        """
        load("@rules_cc//cc:cc_library.bzl", "cc_library")

        genrule(
            name = "src_gen",
            outs = ["gen.cc"],
            cmd = "",
        )

        cc_library(
            name = "foo",
            srcs = [":src_gen"],
        )
        """);
    useConfiguration("--collect_code_coverage");

    InstrumentedFilesInfo target = getInstrumentedFilesProvider("//foo:foo");

    assertThat(prettyArtifactNames(target.getInstrumentedFiles())).isEmpty();
  }

  @Test
  public void testGeneratedFilesAreCollectedWithFlag() throws Exception {
    scratch.file(
        "foo/BUILD",
        """
        load("@rules_cc//cc:cc_library.bzl", "cc_library")

        genrule(
            name = "src_gen",
            outs = ["gen.cc"],
            cmd = "",
        )

        cc_library(
            name = "foo",
            srcs = [":src_gen"],
        )
        """);
    useConfiguration(
        "--collect_code_coverage", "--experimental_collect_code_coverage_for_generated_files");

    InstrumentedFilesInfo target = getInstrumentedFilesProvider("//foo:foo");

    assertThat(prettyArtifactNames(target.getInstrumentedFiles())).containsExactly("foo/gen.cc");
  }

  @Test
  public void testCoverageEnvironmentPrefersTopLevelTarget() throws Exception {
    scratch.file(
        "test/rule.bzl",
        """
        def _impl(ctx):
          return [
            coverage_common.instrumented_files_info(
              ctx,
              dependency_attributes = ["deps"],
              coverage_environment = ctx.attr.env
            )
          ]

        my_rule = rule(
          implementation = _impl,
          attrs = {
            "deps" : attr.label_list(),
            "env" : attr.string_dict(),
          },
        )
        """);
    scratch.file(
        "test/BUILD",
        """
        load(":rule.bzl", "my_rule")
        my_rule(name = "top", env = {"k" : "from_top"}, deps = [":child"])
        my_rule(name = "child", env = {"k" : "from_child"}, deps = [":grandchild"])
        my_rule(name = "grandchild", env = {"k" : "from_grandchild"})
        """);
    useConfiguration("--collect_code_coverage");

    ImmutableMap<String, String> envDict =
        getInstrumentedFilesProvider("//test:top").getCoverageEnvironment();

    assertThat(envDict).containsExactly("k", "from_top");
  }

  private InstrumentedFilesInfo getInstrumentedFilesProvider(String label) throws Exception {
    return getConfiguredTarget(label).get(InstrumentedFilesInfo.STARLARK_CONSTRUCTOR);
  }
}

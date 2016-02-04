// Copyright 2016 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.java.turbine;

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.fail;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Iterables;

import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;

@RunWith(JUnit4.class)
public class TurbineOptionsTest {

  @Rule public final TemporaryFolder tmpFolder = new TemporaryFolder();

  static final ImmutableList<String> BASE_ARGS =
      ImmutableList.of(
          "--output",
          "out.jar",
          "--temp_dir",
          "_tmp",
          "--target_label",
          "//java/com/google/test",
          "--rule_kind",
          "java_library");

  @Test
  public void exhaustiveArgs() throws Exception {
    String[] lines = {
      "--output",
      "out.jar",
      "--source_jars",
      "sources1.srcjar",
      "sources2.srcjar",
      "--temp_dir",
      "_tmp",
      "--processors",
      "com.foo.MyProcessor",
      "com.foo.OtherProcessor",
      "--processorpath",
      "libproc1.jar:libproc2.jar",
      "--classpath",
      "lib1.jar:lib2.jar",
      "--bootclasspath",
      "rt.jar:zipfs.jar",
      "--javacopts",
      "-source",
      "8",
      "-target",
      "8",
      "--sources",
      "Source1.java",
      "Source2.java",
      "--output_deps",
      "out.jdeps",
      "--target_label",
      "//java/com/google/test",
      "--rule_kind",
      "java_library",
    };

    TurbineOptions options =
        TurbineOptionsParser.parse(Iterables.concat(BASE_ARGS, Arrays.asList(lines)));

    assertThat(options.outputFile()).isEqualTo("out.jar");
    assertThat(options.sourceJars())
        .containsExactly("sources1.srcjar", "sources2.srcjar")
        .inOrder();
    assertThat(options.tempDir()).isEqualTo("_tmp");
    assertThat(options.processors())
        .containsExactly("com.foo.MyProcessor", "com.foo.OtherProcessor")
        .inOrder();
    assertThat(options.processorPath()).containsExactly("libproc1.jar", "libproc2.jar").inOrder();
    assertThat(options.classPath()).containsExactly("lib1.jar", "lib2.jar").inOrder();
    assertThat(options.bootClassPath()).containsExactly("rt.jar", "zipfs.jar").inOrder();
    assertThat(options.javacOpts()).containsExactly("-source", "8", "-target", "8").inOrder();
    assertThat(options.sources()).containsExactly("Source1.java", "Source2.java");
    assertThat(options.outputDeps()).hasValue("out.jdeps");
    assertThat(options.targetLabel()).isEqualTo("//java/com/google/test");
    assertThat(options.ruleKind()).isEqualTo("java_library");
  }

  @Test
  public void strictJavaDepsArgs() throws Exception {
    String[] lines = {
      "--strict_java_deps",
      "OFF",
      "--direct_dependency",
      "blaze-out/foo/libbar.jar",
      "//foo/bar",
      "--indirect_dependency",
      "blaze-out/foo/libbaz1.jar",
      "//foo/baz1",
      "--indirect_dependency",
      "blaze-out/foo/libbaz2.jar",
      "//foo/baz2",
      "--deps_artifacts",
      "foo.jdeps",
      "bar.jdeps",
      "",
    };

    TurbineOptions options =
        TurbineOptionsParser.parse(Iterables.concat(BASE_ARGS, Arrays.asList(lines)));

    assertThat(options.strictDepsMode()).isEqualTo("OFF");
    assertThat(options.targetLabel()).isEqualTo("//java/com/google/test");
    assertThat(options.directJarsToTargets())
        .containsExactlyEntriesIn(ImmutableMap.of("blaze-out/foo/libbar.jar", "//foo/bar"));
    assertThat(options.indirectJarsToTargets())
        .containsExactlyEntriesIn(
            ImmutableMap.of(
                "blaze-out/foo/libbaz1.jar", "//foo/baz1",
                "blaze-out/foo/libbaz2.jar", "//foo/baz2"));
    assertThat(options.depsArtifacts()).containsExactly("foo.jdeps", "bar.jdeps");
  }

  @Test
  public void classpathArgs() throws Exception {
    String[] lines = {
      "--classpath",
      "liba.jar:libb.jar:libc.jar",
      "--processorpath",
      "libpa.jar:libpb.jar:libpc.jar",
    };

    TurbineOptions options =
        TurbineOptionsParser.parse(Iterables.concat(BASE_ARGS, Arrays.asList(lines)));

    assertThat(options.classPath()).containsExactly("liba.jar", "libb.jar", "libc.jar").inOrder();
    assertThat(options.processorPath())
        .containsExactly("libpa.jar", "libpb.jar", "libpc.jar")
        .inOrder();
  }

  @Test
  public void paramsFile() throws Exception {
    Iterable<String> paramsArgs =
        Iterables.concat(BASE_ARGS, Arrays.asList("--javacopts", "-source", "8", "-target", "8"));
    Path params = tmpFolder.newFile("params.txt").toPath();
    Files.write(params, paramsArgs, StandardCharsets.UTF_8);

    // @ is a prefix for external repository targets, and the prefix for params files. Targets
    // are disambiguated by prepending an extra @.
    String[] lines = {
      "@" + params.toAbsolutePath(), "--target_label", "//custom/label",
    };

    TurbineOptions options = TurbineOptionsParser.parse(Arrays.asList(lines));

    // assert that options were read from params file
    assertThat(options.javacOpts()).containsExactly("-source", "8", "-target", "8").inOrder();
    // ... and directly from the command line
    assertThat(options.targetLabel()).isEqualTo("//custom/label");
  }

  @Test
  public void escapedExternalRepositoryLabel() throws Exception {
    // @ is a prefix for external repository targets, and the prefix for params files. Targets
    // are disambiguated by prepending an extra @.
    String[] lines = {
      "--target_label", "@@other-repo//foo:local-jam",
    };

    TurbineOptions options =
        TurbineOptionsParser.parse(Iterables.concat(BASE_ARGS, Arrays.asList(lines)));

    assertThat(options.targetLabel()).isEqualTo("@@other-repo//foo:local-jam");
  }

  @Test
  public void failIfMissingExpectedArgs() throws Exception {
    try {
      TurbineOptions.builder().build();
      fail();
    } catch (NullPointerException e) {
      assertThat(e).hasMessage("output must not be null");
    }
  }
}

// Copyright 2006 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.rules.filegroup;

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.assertThrows;

import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.actions.util.ActionsTestUtil;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.OutputGroupInfo;
import com.google.devtools.build.lib.analysis.configuredtargets.FileConfiguredTarget;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.rules.java.JavaSemantics;
import com.google.devtools.build.lib.util.FileType;
import com.google.testing.junit.testparameterinjector.TestParameter;
import com.google.testing.junit.testparameterinjector.TestParameterInjector;
import java.io.IOException;
import org.junit.Test;
import org.junit.runner.RunWith;

/** Tests for {@link Filegroup}. */
@RunWith(TestParameterInjector.class)
public class FilegroupConfiguredTargetTest extends BuildViewTestCase {

  @Test
  public void testGroup() throws Exception {
    scratch.file(
        "nevermore/BUILD",
        """
        filegroup(name  = 'staticdata',
                  srcs = ['staticdata/spam.txt', 'staticdata/good.txt'])
        """);
    ConfiguredTarget groupTarget = getConfiguredTarget("//nevermore:staticdata");
    assertThat(ActionsTestUtil.prettyArtifactNames(getFilesToBuild(groupTarget)))
        .containsExactly("nevermore/staticdata/spam.txt", "nevermore/staticdata/good.txt");
  }

  @Test
  public void testDependencyGraph() throws Exception {
    scratch.file(
        "java/com/google/test/BUILD",
        """
        load("@rules_java//java:defs.bzl", "java_binary")
        java_binary(name  = 'test_app',
            resources = [':data'],
            create_executable = 0,
            srcs  = ['InputFile.java', 'InputFile2.java'])
        filegroup(name  = 'data',
                  srcs = ['b.txt', 'a.txt'])
        """);
    FileConfiguredTarget appOutput =
        getFileConfiguredTarget("//java/com/google/test:test_app.jar");
    assertThat(actionsTestUtil().predecessorClosureOf(appOutput.getArtifact(), FileType.of(".txt")))
        .isEqualTo("b.txt a.txt");
  }

  @Test
  public void testEmptyGroupIsAnOk() throws Exception {
    scratchConfiguredTarget("empty", "empty",
        "filegroup(name='empty', srcs=[])");
  }

  @Test
  public void testEmptyGroupInGenruleIsOk() throws Exception {
    scratchConfiguredTarget("empty", "genempty",
        "filegroup(name='empty', srcs=[])",
        "genrule(name='genempty', tools=[':empty'], outs=['nothing'], cmd='touch $@')");
  }

  private void writeTest() throws IOException {
    scratch.file(
        "another/BUILD",
        """
        filegroup(name  = 'another',
                  srcs = ['another.txt'])
        """);
    scratch.file(
        "test/BUILD",
        """
        filegroup(name  = 'a',
                  srcs = ['a.txt'])
        filegroup(name  = 'b',
                  srcs = ['a.txt'])
        filegroup(name  = 'c',
                  srcs = ['a', 'b.txt'])
        filegroup(name  = 'd',
                  srcs = ['//another:another.txt'])
        """);
  }

  @Test
  public void testFileCanBeSrcsOfMultipleRules() throws Exception {
    writeTest();
    assertThat(
            ActionsTestUtil.prettyArtifactNames(getFilesToBuild(getConfiguredTarget("//test:a"))))
        .containsExactly("test/a.txt");
    assertThat(
            ActionsTestUtil.prettyArtifactNames(getFilesToBuild(getConfiguredTarget("//test:b"))))
        .containsExactly("test/a.txt");
  }

  @Test
  public void testRuleCanBeSrcsOfOtherRule() throws Exception {
    writeTest();
    assertThat(
            ActionsTestUtil.prettyArtifactNames(getFilesToBuild(getConfiguredTarget("//test:c"))))
        .containsExactly("test/a.txt", "test/b.txt");
  }

  @Test
  public void testOtherPackageCanBeSrcsOfRule() throws Exception {
    writeTest();
    assertThat(
            ActionsTestUtil.prettyArtifactNames(getFilesToBuild(getConfiguredTarget("//test:d"))))
        .containsExactly("another/another.txt");
  }

  @Test
  public void testIsNotExecutable() throws Exception {
    scratch.file("x/BUILD",
                "filegroup(name = 'not_exec_two_files', srcs = ['bin', 'bin.sh'])");
    assertThat(getExecutable("//x:not_exec_two_files")).isNull();
  }

  @Test
  public void testIsExecutable() throws Exception {
    scratch.file("x/BUILD",
                "filegroup(name = 'exec', srcs = ['bin'])");
    assertThat(getExecutable("//x:exec").getExecPath().getPathString()).isEqualTo("x/bin");
  }

  @Test
  public void testNoDuplicate() throws Exception {
    scratch.file(
        "x/BUILD",
        """
        filegroup(name = 'a', srcs = ['file'])
        filegroup(name = 'b', srcs = ['file'])
        filegroup(name = 'c', srcs = [':a', ':b'])
        """);
    assertThat(ActionsTestUtil.prettyArtifactNames(getFilesToBuild(getConfiguredTarget("//x:c"))))
        .containsExactly("x/file");
  }

  @Test
  public void testGlobMatchesRuleOutputsInsteadOfFileWithTheSameName() throws Exception {
    scratch.file("pkg/file_or_rule");
    scratch.file("pkg/a.txt");
    ConfiguredTarget target = scratchConfiguredTarget("pkg", "my_rule",
                "filegroup(name = 'file_or_rule', srcs = ['a.txt'])",
                "filegroup(name = 'my_rule', srcs = glob(['file_or_rule']))");
    assertThat(ActionsTestUtil.baseArtifactNames(getFilesToBuild(target))).containsExactly("a.txt");
  }

  @Test
  public void outputGroupSourceJars_extractsTransitiveSources() throws Exception {
    scratch.file("pkg/a.java");
    scratch.file("pkg/b.java");
    scratch.file("pkg/c.java");
    scratch.file(
        "pkg/BUILD",
        "load('@rules_java//java:defs.bzl', 'java_library')",
        "java_library(name='lib_a', srcs=['a.java'])",
        "java_library(name='lib_b', srcs=['b.java'], deps = [':lib_c'])",
        "java_library(name='lib_c', srcs=['c.java'])",
        "filegroup(name='group', srcs=[':lib_a', ':lib_b'],"
            + String.format("output_group='%s')", JavaSemantics.SOURCE_JARS_OUTPUT_GROUP));

    ConfiguredTarget group = getConfiguredTarget("//pkg:group");

    assertThat(ActionsTestUtil.prettyArtifactNames(getFilesToBuild(group)))
        .containsExactly("pkg/liblib_a-src.jar", "pkg/liblib_b-src.jar", "pkg/liblib_c-src.jar");
  }

  @Test
  public void outputGroupDirectSourceJars_extractsDirectSources() throws Exception {
    scratch.file("pkg/a.java");
    scratch.file("pkg/b.java");
    scratch.file("pkg/c.java");
    scratch.file(
        "pkg/BUILD",
        "load('@rules_java//java:defs.bzl', 'java_library')",
        "java_library(name='lib_a', srcs=['a.java'])",
        "java_library(name='lib_b', srcs=['b.java'], deps = [':lib_c'])",
        "java_library(name='lib_c', srcs=['c.java'])",
        "filegroup(name='group', srcs=[':lib_a', ':lib_b'],"
            + String.format("output_group='%s')", JavaSemantics.DIRECT_SOURCE_JARS_OUTPUT_GROUP));

    ConfiguredTarget group = getConfiguredTarget("//pkg:group");

    assertThat(ActionsTestUtil.prettyArtifactNames(getFilesToBuild(group)))
        .containsExactly("pkg/liblib_a-src.jar", "pkg/liblib_b-src.jar");
  }

  @Test
  public void testErrorForIllegalOutputGroup() throws Exception {
    scratch.file("pkg/a.cc");
    scratch.file(
        "pkg/BUILD",
        "cc_library(name='lib_a', srcs=['a.cc'])",
        String.format(
            "filegroup(name='group', srcs=[':lib_a'], output_group='%s')",
            OutputGroupInfo.HIDDEN_TOP_LEVEL));
    AssertionError e = assertThrows(AssertionError.class, () -> getConfiguredTarget("//pkg:group"));
    assertThat(e)
        .hasMessageThat()
        .contains(
            String.format(Filegroup.ILLEGAL_OUTPUT_GROUP_ERROR, OutputGroupInfo.HIDDEN_TOP_LEVEL));
  }

  @Test
  public void testDefaultInfo(@TestParameter boolean filegroupRunfilesForData) throws Exception {
    scratch.file(
        "x/defs.bzl",
        """
        def _default_info_impl(ctx):
            files = depset(transitive = [t[DefaultInfo].files for t in ctx.attr.files])
            default_runfiles = ctx.runfiles(transitive_files = depset(transitive = [t[DefaultInfo].files for t in ctx.attr.default_runfiles]))
            data_runfiles = ctx.runfiles(transitive_files = depset(transitive = [t[DefaultInfo].files for t in ctx.attr.data_runfiles]))
            return [
                DefaultInfo(
                    files = files,
                    default_runfiles = default_runfiles,
                    data_runfiles = data_runfiles,
                )
            ]
        default_info = rule(
            implementation = _default_info_impl,
            attrs = {
                "files": attr.label_list(allow_files=True),
                "default_runfiles": attr.label_list(allow_files=True),
                "data_runfiles": attr.label_list(allow_files=True),
            },
        )
        """);
    scratch.file(
        "x/BUILD",
        """
        load(":defs.bzl", "default_info")

        default_info(
            name = "default_info_srcs",
            files = ["srcs_files_file"],
            default_runfiles = ["srcs_default_runfiles_file"],
            data_runfiles = ["srcs_data_runfiles_file"],
        )

        default_info(
            name = "default_info_data",
            files = ["data_files"],
            default_runfiles = ["data_default_runfiles_file"],
            data_runfiles = ["data_data_runfiles_file"],
        )

        filegroup(
            name = "filegroup",
            srcs = [
                ":default_info_srcs",
                "srcs_file",
            ],
            data = [
                ":default_info_data",
                "data_file",
            ],
        )
        """);

    useConfiguration("--incompatible_filegroup_runfiles_for_data=" + filegroupRunfilesForData);
    var filegroup = getConfiguredTarget("//x:filegroup");

    assertThat(ActionsTestUtil.prettyArtifactNames(getFilesToBuild(filegroup)))
        .containsExactly("x/srcs_file", "x/srcs_files_file");
    assertThat(ActionsTestUtil.prettyArtifactNames(getDefaultRunfiles(filegroup).getArtifacts()))
        .containsExactly(
            "x/srcs_default_runfiles_file",
            "x/data_file",
            "x/data_files",
            "x/data_data_runfiles_file");
    var expectedDataRunfiles =
        ImmutableSet.<String>builder()
            .add(
                "x/srcs_file",
                "x/srcs_files_file",
                "x/data_file",
                "x/data_files",
                "x/data_data_runfiles_file");
    if (filegroupRunfilesForData) {
      expectedDataRunfiles.add("x/srcs_data_runfiles_file");
    }
    assertThat(ActionsTestUtil.prettyArtifactNames(getDataRunfiles(filegroup).getArtifacts()))
        .containsExactlyElementsIn(expectedDataRunfiles.build());
  }
}

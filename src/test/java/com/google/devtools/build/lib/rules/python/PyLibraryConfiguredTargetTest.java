// Copyright 2018 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.rules.python;

import static com.google.common.truth.Truth.assertThat;
import static com.google.devtools.build.lib.rules.python.PythonTestUtils.getPyLoad;

import com.google.devtools.build.lib.actions.util.ActionsTestUtil;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.OutputGroupInfo;
import com.google.devtools.build.lib.analysis.configuredtargets.FileConfiguredTarget;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@code py_library}. */
@RunWith(JUnit4.class)
public class PyLibraryConfiguredTargetTest extends PyBaseConfiguredTargetTestBase {

  public PyLibraryConfiguredTargetTest() {
    super("py_library");
  }

  @Test
  public void pyRuntimeInfoIsNotPresent() throws Exception {
    scratch.file(
        "pkg/BUILD", //
        getPyLoad("py_library"),
        "py_library(",
        "    name = 'foo',",
        "    srcs = [':foo.py'],",
        ")");
    assertThat(PyRuntimeInfo.fromTargetNullable(getConfiguredTarget("//pkg:foo"))).isNull();
  }

  @Test
  public void versionPy3ByDefault() throws Exception {
    scratch.file(
        "pkg/BUILD", //
        getPyLoad("py_library"),
        "py_library(",
        "    name = 'foo',",
        "    srcs = ['foo.py'],",
        ")");
    assertThat(getPythonVersion(getConfiguredTarget("//pkg:foo"))).isEqualTo(PythonVersion.PY3);
  }

  @Test
  public void filesToBuild() throws Exception {
    scratch.file(
        "pkg/BUILD", //
        getPyLoad("py_library"),
        "py_library(",
        "    name = 'foo',",
        "    srcs = ['foo.py'])");
    ConfiguredTarget target = getConfiguredTarget("//pkg:foo");
    FileConfiguredTarget srcFile = getFileConfiguredTarget("//pkg:foo.py");
    assertThat(getFilesToBuild(target).toList()).containsExactly(srcFile.getArtifact());
  }

  @Test
  public void srcsCanContainRuleGeneratingPyAndNonpyFiles() throws Exception {
    scratchConfiguredTarget(
        "pkg",
        "foo",
        // build file:
        getPyLoad("py_binary"),
        "py_binary(",
        "    name = 'foo',",
        "    srcs = ['foo.py', ':bar'])",
        "genrule(",
        "    name = 'bar',",
        "    outs = ['bar.cc', 'bar.py'],",
        "    cmd = 'touch $(OUTS)')");
    assertNoEvents();
  }

  @Test
  public void whatIfSrcsContainsRuleGeneratingNoPyFiles() throws Exception {
    // In Bazel it's an error, in Blaze it's a warning.
    String[] lines = {
      getPyLoad("py_binary"),
      "py_binary(",
      "    name = 'foo',",
      "    srcs = ['foo.py', ':bar'])",
      "genrule(",
      "    name = 'bar',",
      "    outs = ['bar.cc'],",
      "    cmd = 'touch $(OUTS)')"
    };
    if (analysisMock.isThisBazel()) {
      checkError(
          "pkg",
          "foo",
          // error:
          "'//pkg:bar' does not produce any py_binary srcs files",
          // build file:
          lines);
    }
  }

  @Test
  public void filesToCompile() throws Exception {
    ConfiguredTarget lib =
        scratchConfiguredTarget(
            "pkg",
            "lib",
            // build file:
            getPyLoad("py_library"),
            "py_library(name = 'lib', srcs = ['lib.py'], deps = [':bar'])",
            "py_library(name = 'bar', srcs = ['bar.py'], deps = [':baz'])",
            "py_library(name = 'baz', srcs = ['baz.py'])");

    assertThat(
            ActionsTestUtil.baseNamesOf(
                getOutputGroup(lib, OutputGroupInfo.COMPILATION_PREREQUISITES)))
        .isEqualTo("baz.py bar.py lib.py");

    // compilationPrerequisites should be included in filesToCompile.
    assertThat(ActionsTestUtil.baseNamesOf(getOutputGroup(lib, OutputGroupInfo.FILES_TO_COMPILE)))
        .isEqualTo("baz.py bar.py lib.py");
  }

  @Test
  public void libraryTargetCanBeInPackageWithHyphensIfSourcesAreRemote() throws Exception {
    scratch.file(
        "pkg/BUILD", //
        "exports_files(['foo.py'])");
    scratchConfiguredTarget(
        "pkg-with-hyphens", //
        "foo",
        getPyLoad("py_library"),
        "py_library(",
        "    name = 'foo',",
        "    srcs = ['//pkg:foo.py'])");
    assertNoEvents();
  }
}

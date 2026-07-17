// Copyright 2021 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.packages;

import static com.google.common.collect.ImmutableList.toImmutableList;
import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.assertThrows;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.skyframe.ConfiguredTargetAndData;
import com.google.devtools.build.lib.vfs.ModifiedFileSet;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Root;
import java.io.IOException;
import java.util.List;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@code native.glob} function. */
@RunWith(JUnit4.class)
public class NativeGlobTest extends BuildViewTestCase {

  @Test
  public void glob_simple() throws Exception {
    makeFile("test/starlark/file1.txt");
    makeFile("test/starlark/file2.txt");
    makeFile("test/starlark/file3.txt");

    makeGlobFilegroup("test/starlark/BUILD", "glob(['*'])");

    assertAttrLabelList(
        "//test/starlark:files",
        "srcs",
        ImmutableList.of(
            "//test/starlark:BUILD",
            "//test/starlark:file1.txt",
            "//test/starlark:file2.txt",
            "//test/starlark:file3.txt"));
  }

  @Test
  public void glob_not_empty() throws Exception {

    makeGlobFilegroup("test/starlark/BUILD", "glob(['foo*'], allow_empty=False)");

    AssertionError e =
        assertThrows(
            AssertionError.class,
            () -> assertAttrLabelList("//test/starlark:files", "srcs", ImmutableList.of()));
    assertThat(e).hasMessageThat().contains("allow_empty");
  }

  @Test
  public void glob_simple_subdirs() throws Exception {
    makeFile("test/starlark/sub/file1.txt");
    makeFile("test/starlark/sub2/file2.txt");
    makeFile("test/starlark/sub3/file3.txt");

    makeGlobFilegroup("test/starlark/BUILD", "glob(['**'])");

    assertAttrLabelList(
        "//test/starlark:files",
        "srcs",
        ImmutableList.of(
            "//test/starlark:BUILD",
            "//test/starlark:sub/file1.txt",
            "//test/starlark:sub2/file2.txt",
            "//test/starlark:sub3/file3.txt"));
  }

  @Test
  public void glob_incremental() throws Exception {
    makeFile("test/starlark/file1.txt");
    makeGlobFilegroup("test/starlark/BUILD", "glob(['**'])");

    assertAttrLabelList(
        "//test/starlark:files",
        "srcs",
        ImmutableList.of("//test/starlark:BUILD", "//test/starlark:file1.txt"));

    scratch.file("test/starlark/file2.txt");
    scratch.file("test/starlark/sub/subfile3.txt");

    // Poke SkyFrame to tell it what changed.
    invalidateSkyFrameFiles(
        "test/starlark", "test/starlark/file2.txt", "test/starlark/sub/subfile3.txt");

    assertAttrLabelList(
        "//test/starlark:files",
        "srcs",
        ImmutableList.of(
            "//test/starlark:BUILD",
            "//test/starlark:file1.txt",
            "//test/starlark:file2.txt",
            "//test/starlark:sub/subfile3.txt"));
  }

  /**
   * Constructs a BUILD file containing a single rule with uses glob() to list files look for a rule
   * called :files in it.
   */
  private void makeGlobFilegroup(String buildPath, String glob) throws IOException {
    scratch.file(buildPath, "filegroup(", "   name = 'files',", "   srcs = " + glob, ")");
  }

  private void assertAttrLabelList(String target, String attrName, List<String> expectedLabels)
      throws Exception {
    ConfiguredTargetAndData cfgTarget = getConfiguredTargetAndData(target);
    assertThat(cfgTarget).isNotNull();

    ImmutableList<Label> labels =
        expectedLabels.stream().map(this::makeLabel).collect(toImmutableList());

    ConfiguredAttributeMapper configuredAttributeMapper =
        getMapperFromConfiguredTargetAndTarget(cfgTarget);
    assertThat(configuredAttributeMapper.get(attrName, BuildType.LABEL_LIST))
        .containsExactlyElementsIn(labels);
  }

  private Label makeLabel(String label) {
    try {
      return Label.parseCanonical(label);
    } catch (Exception e) {
      // Always fails the test.
      assertThat(e).isNull();
      return null;
    }
  }

  private void invalidateSkyFrameFiles(String... files) throws Exception {
    ModifiedFileSet.Builder builder = ModifiedFileSet.builder();

    for (String f : files) {
      builder.modify(PathFragment.create(f));
    }

    getSkyframeExecutor()
        .invalidateFilesUnderPathForTesting(
            reporter, builder.build(), Root.fromPath(rootDirectory));
  }

  private void makeFile(String fileName) throws IOException {
    scratch.file(fileName, "Content: " + fileName);
  }
}

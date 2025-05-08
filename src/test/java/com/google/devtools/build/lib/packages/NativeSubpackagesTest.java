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
import static org.junit.Assert.fail;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.skyframe.ConfiguredTargetAndData;
import com.google.devtools.build.lib.vfs.ModifiedFileSet;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Root;
import com.google.testing.junit.testparameterinjector.TestParameter;
import com.google.testing.junit.testparameterinjector.TestParameterInjector;
import java.io.IOException;
import java.util.List;
import java.util.stream.Collectors;
import org.junit.Test;
import org.junit.runner.RunWith;

/** Tests for {@code native.subpackages} function. */
@RunWith(TestParameterInjector.class)
public class NativeSubpackagesTest extends BuildViewTestCase {

  private static final String ALL_SUBDIRS = "**";

  @Test
  public void subpackages_simple_subDir() throws Exception {
    makeSubpackageFileGroup("test/starlark/BUILD", ALL_SUBDIRS, null, null);
    makeFilesSubPackage("test/starlark/sub");

    assertAttrLabelList(
        "//test/starlark:files", "srcs", ImmutableList.of("//test/starlark/sub:files"));
  }

  @Test
  public void subpackages_simple_include() throws Exception {
    makeSubpackageFileGroup("test/starlark/BUILD", "sub1", null, null);

    makeFilesSubPackage("test/starlark/sub");
    makeFilesSubPackage("test/starlark/sub1");
    makeFilesSubPackage("test/starlark/sub2");

    assertAttrLabelList(
        "//test/starlark:files", "srcs", ImmutableList.of("//test/starlark/sub1:files"));
  }

  @Test
  public void subpackages_simple_exclude() throws Exception {
    makeSubpackageFileGroup("test/starlark/BUILD", ALL_SUBDIRS, "['sub2/**']", null);

    makeFilesSubPackage("test/starlark/sub");
    makeFilesSubPackage("test/starlark/sub1");
    makeFilesSubPackage("test/starlark/sub2");
    makeFilesSubPackage("test/starlark/sub3");

    assertAttrLabelList(
        "//test/starlark:files",
        "srcs",
        ImmutableList.of(
            "//test/starlark/sub:files",
            "//test/starlark/sub1:files",
            "//test/starlark/sub3:files"));
  }

  @Test
  public void subpackages_simple_empty_allow() throws Exception {
    makeSubpackageFileGroup("test/starlark/BUILD", ALL_SUBDIRS, null, true);
    assertAttrLabelList("//test/starlark:files", "srcs", ImmutableList.of());
  }

  @Test
  public void subpackages_simple_empty_disallow() throws Exception {
    makeSubpackageFileGroup("test/starlark/BUILD", ALL_SUBDIRS, null, null);

    // force evaluation
    AssertionError e =
        assertThrows(AssertionError.class, () -> getConfiguredTarget("//test/starlark:files"));
    assertThat(e).hasMessageThat().contains("subpackages pattern '**' didn't match anything");
  }

  @Test
  public void subpackages_deeplyNested_withSubdirs() throws Exception {
    makeSubpackageFileGroup("test/starlark/BUILD", ALL_SUBDIRS, null, true);

    // Setup a dir with 2 subdirs, 1 a package one not
    makeFilesSubPackage("test/starlark/sub");
    // Should be blocked by 'sub'
    makeFilesSubPackage("test/starlark/sub/sub2");

    makeFilesSubPackage("test/starlark/sub3");
    makeFilesSubPackage("test/starlark/not_sub/sub_is_pkg/eventually");

    scratch.file("test/starlark/not_sub/file1.txt");
    scratch.file("test/starlark/not_sub/double_not_sub/file.txt");

    assertAttrLabelList(
        "//test/starlark:files",
        "srcs",
        ImmutableList.of(
            "//test/starlark/sub:files",
            "//test/starlark/sub3:files",
            "//test/starlark/not_sub/sub_is_pkg/eventually:files"));
  }

  @Test
  public void subpackages_incremental_addSubPkg() throws Exception {
    makeSubpackageFileGroup("test/starlark/BUILD", ALL_SUBDIRS, null, null);

    // Setup a two subdirs one shallow and one deep
    makeFilesSubPackage("test/starlark/sub");
    makeFilesSubPackage("test/starlark/deep/1/2/3");

    assertAttrLabelList(
        "//test/starlark:files",
        "srcs",
        ImmutableList.of("//test/starlark/sub:files", "//test/starlark/deep/1/2/3:files"));

    // Add a 2nd shallow and 2nd deep mid
    makeFilesSubPackage("test/starlark/sub2");

    // Poke Skyframe by invalidating the dirent and files that changed.
    invalidateSkyFrameFiles(
        "test/starlark/sub2", "test/starlark/sub2/BUILD", "test/starlark/sub2/file.txt");

    // We should now be aware of the new one via Skyframe invalidation.
    assertAttrLabelList(
        "//test/starlark:files",
        "srcs",
        ImmutableList.of(
            "//test/starlark/sub:files",
            "//test/starlark/sub2:files",
            "//test/starlark/deep/1/2/3:files"));
  }

  @Test
  public void subpackages_incremental_delSubPkg() throws Exception {
    makeSubpackageFileGroup("test/starlark/BUILD", ALL_SUBDIRS, null, null);

    // Setup a single subdir
    makeFilesSubPackage("test/starlark/sub");
    makeFilesSubPackage("test/starlark/sub2");

    assertAttrLabelList(
        "//test/starlark:files",
        "srcs",
        ImmutableList.of("//test/starlark/sub:files", "//test/starlark/sub2:files"));

    scratch.deleteFile("test/starlark/sub2/BUILD");
    scratch.deleteFile("test/starlark/sub2/file.txt");

    invalidateSkyFrameFiles("test/starlark/sub2/BUILD", "test/starlark/sub2/file.txt");

    // We should now be aware of the new one.
    assertAttrLabelList(
        "//test/starlark:files", "srcs", ImmutableList.of("//test/starlark/sub:files"));
  }

  @Test
  public void subpackages_incremental_convertSubDirToPkg() throws Exception {
    makeSubpackageFileGroup("test/starlark/BUILD", ALL_SUBDIRS, null, null);

    // Setup both immediate and deeply nested sub-dirs with BUILD files.
    makeFilesSubPackage("test/starlark/sub");
    scratch.file("test/starlark/sub2/file2.txt");

    // Initially we have a subdir with 'sub/BUILD' and sub2/file2.txt"
    assertAttrLabelList(
        "//test/starlark:files", "srcs", ImmutableList.of("//test/starlark/sub:files"));

    // Then we add a BUILD file to sub2 making it a package Skyframe should pick
    // that up once invalidated.
    makeFilesSubPackage("test/starlark/sub2");

    // Poke Skyframe by invalidating the dirent and files that changed.
    invalidateSkyFrameFiles("test/starlark/sub2/BUILD", "test/starlark/sub2/file.txt");

    assertAttrLabelList(
        "//test/starlark:files",
        "srcs",
        ImmutableList.of("//test/starlark/sub:files", "//test/starlark/sub2:files"));
  }

  @Test
  public void invalidPositionalParams() throws Exception {
    scratch.file("foo/subdir/BUILD");
    scratch.file("foo/BUILD", "[filegroup(name = p) for p in subpackages(['subdir'])]");

    AssertionError e =
        assertThrows(AssertionError.class, () -> getConfiguredTargetAndData("//foo:subdir"));
    assertThat(e).hasMessageThat().contains("got unexpected positional argument");
  }

  @Test
  public void invalidMissingInclude() throws Exception {
    scratch.file("foo/subdir/BUILD");
    scratch.file("foo/BUILD", "[filegroup(name = p) for p in subpackages()]");

    AssertionError e =
        assertThrows(AssertionError.class, () -> getConfiguredTargetAndData("//foo:subdir"));
    assertThat(e).hasMessageThat().contains("missing 1 required named argument: include");
  }

  @Test
  public void validNoWildCardInclude() throws Exception {
    makeSubpackageFileGroup(
        "test/starlark/BUILD", /*include=*/ ImmutableList.of("sub", "sub2/deep"), null, null);
    makeFilesSubPackage("test/starlark/sub");
    makeFilesSubPackage("test/starlark/sub2/deep");

    assertAttrLabelList(
        "//test/starlark:files",
        "srcs",
        ImmutableList.of("//test/starlark/sub:files", "//test/starlark/sub2/deep:files"));
  }

  @Test
  public void includeValidMatchSubdir() throws Exception {
    scratch.file("foo/subdir/BUILD");
    scratch.file("foo/BUILD", "[filegroup(name = p) for p in subpackages(include = ['subdir'])]");
    getConfiguredTargetAndData("//foo:subdir");
  }

  @Test
  public void includeValidSubMatchSubdir(
      @TestParameter({
            "subdir/*/deeper",
            "subdir/sub*/deeper",
            "subdir/**",
            "subdir/*/deeper/**",
            "subdir/**/deeper/**"
          })
          String expression)
      throws Exception {
    makeFilesSubPackage("test/starlark/subdir/sub/deeper");
    makeFilesSubPackage("test/starlark/subdir/sub2/deeper");
    makeFilesSubPackage("test/starlark/subdir/sub3/deeper");

    makeSubpackageFileGroup("test/starlark/BUILD", expression, null, null);

    assertAttrLabelList(
        "//test/starlark:files",
        "srcs",
        ImmutableList.of(
            "//test/starlark/subdir/sub/deeper:files",
            "//test/starlark/subdir/sub2/deeper:files",
            "//test/starlark/subdir/sub3/deeper:files"));
  }

  /**
   * Constructs a BUILD file with a single filegroup target whose srcs attribute is the list of all
   * //p:files, where //p is a subpackage returned by a call to native.subpackages.
   */
  private void makeSubpackageFileGroup(
      String buildPath, ImmutableList<String> include, String exclude, Boolean allowEmpty)
      throws IOException {
    StringBuilder subpackages = new StringBuilder();
    subpackages.append("subpackages(include = [");
    subpackages.append(include.stream().map(i -> "'" + i + "'").collect(Collectors.joining(", ")));
    subpackages.append("]");

    if (exclude != null) {
      subpackages.append(", exclude = ");
      subpackages.append(exclude);
    }

    if (allowEmpty != null) {
      subpackages.append(", allow_empty = ");
      subpackages.append(allowEmpty ? "True" : "False");
    }
    subpackages.append(")");

    scratch.file(
        buildPath,
        "filegroup(",
        "   name = 'files',",
        "   srcs = [",
        "     '//%s/%s:files' % (package_name(), s) for s in " + subpackages,
        "   ],",
        ")");
  }

  private void makeSubpackageFileGroup(
      String buildPath, String include, String exclude, Boolean allowEmpty) throws IOException {
    makeSubpackageFileGroup(buildPath, ImmutableList.of(include), exclude, allowEmpty);
  }

  /**
   * Creates a BUILD file and single file at the given packagePath, the BUILD file will contain a
   * single filegroup called 'files' which contains the created file.
   */
  private void makeFilesSubPackage(String packagePath) throws IOException {
    scratch.file(packagePath + "/file.txt");
    scratch.file(
        packagePath + "/BUILD", "filegroup(", "   name = 'files',", "   srcs = glob(['*']),", ")");
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
      fail("Unable to construct Label from " + label);
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
}

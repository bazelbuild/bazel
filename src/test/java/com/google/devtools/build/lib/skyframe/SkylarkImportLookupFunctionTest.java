// Copyright 2015 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.skyframe;

import static com.google.common.truth.Truth.assertThat;
import static com.google.devtools.build.skyframe.EvaluationResultSubjectFactory.assertThatEvaluationResult;
import static org.junit.Assert.fail;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.clock.BlazeClock;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.packages.ConstantRuleVisibility;
import com.google.devtools.build.lib.packages.SkylarkSemanticsOptions;
import com.google.devtools.build.lib.pkgcache.PackageCacheOptions;
import com.google.devtools.build.lib.pkgcache.PathPackageLocator;
import com.google.devtools.build.lib.skyframe.SkylarkImportLookupFunction.SkylarkImportFailedException;
import com.google.devtools.build.lib.skyframe.util.SkyframeExecutorTestUtils;
import com.google.devtools.build.lib.util.io.TimestampGranularityMonitor;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.Root;
import com.google.devtools.build.skyframe.ErrorInfo;
import com.google.devtools.build.skyframe.EvaluationResult;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.common.options.Options;
import java.util.UUID;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Tests for SkylarkImportLookupFunction.
 */
@RunWith(JUnit4.class)
public class SkylarkImportLookupFunctionTest extends BuildViewTestCase {

  String preludeLabelRelativePath;

  @Before
  public final void preparePackageLoading() throws Exception  {
    Path alternativeRoot = scratch.dir("/root_2");
    PackageCacheOptions packageCacheOptions = Options.getDefaults(PackageCacheOptions.class);
    packageCacheOptions.defaultVisibility = ConstantRuleVisibility.PUBLIC;
    packageCacheOptions.showLoadingProgress = true;
    packageCacheOptions.globbingThreads = 7;
    getSkyframeExecutor()
        .preparePackageLoading(
            new PathPackageLocator(
                outputBase,
                ImmutableList.of(Root.fromPath(rootDirectory), Root.fromPath(alternativeRoot)),
                BazelSkyframeExecutorConstants.BUILD_FILES_BY_PRIORITY),
            packageCacheOptions,
            Options.getDefaults(SkylarkSemanticsOptions.class),
            "",
            UUID.randomUUID(),
            ImmutableMap.<String, String>of(),
            new TimestampGranularityMonitor(BlazeClock.instance()));
    skyframeExecutor.setActionEnv(ImmutableMap.<String, String>of());
    this.preludeLabelRelativePath =
        getRuleClassProvider().getPreludeLabel().toPathFragment().toString();
  }

  @Test
  public void testSkylarkImportLabels() throws Exception {
    scratch.file("pkg1/BUILD");
    scratch.file("pkg1/ext.bzl");
    checkSuccessfulLookup("//pkg1:ext.bzl");

    scratch.file("pkg2/BUILD");
    scratch.file("pkg2/dir/ext.bzl");
    checkSuccessfulLookup("//pkg2:dir/ext.bzl");

    scratch.file("dir/pkg3/BUILD");
    scratch.file("dir/pkg3/dir/ext.bzl");
    checkSuccessfulLookup("//dir/pkg3:dir/ext.bzl");
  }

  @Test
  public void testSkylarkImportLabelsAlternativeRoot() throws Exception {
    scratch.file("/root_2/pkg4/BUILD");
    scratch.file("/root_2/pkg4/ext.bzl");
    checkSuccessfulLookup("//pkg4:ext.bzl");
  }

  @Test
  public void testSkylarkImportLabelsMultipleBuildFiles() throws Exception {
    scratch.file("dir1/BUILD");
    scratch.file("dir1/dir2/BUILD");
    scratch.file("dir1/dir2/ext.bzl");
    checkSuccessfulLookup("//dir1/dir2:ext.bzl");
  }

  @Test
  public void testLoadFromSkylarkFileInRemoteRepo() throws Exception {
    scratch.deleteFile(preludeLabelRelativePath);
    scratch.overwriteFile("WORKSPACE",
        "local_repository(",
        "    name = 'a_remote_repo',",
        "    path = '/a_remote_repo'",
        ")");
    scratch.file("/a_remote_repo/WORKSPACE");
    scratch.file("/a_remote_repo/remote_pkg/BUILD");
    scratch.file("/a_remote_repo/remote_pkg/ext1.bzl",
        "load(':ext2.bzl', 'CONST')");
    scratch.file("/a_remote_repo/remote_pkg/ext2.bzl",
        "CONST = 17");
    checkSuccessfulLookup("@a_remote_repo//remote_pkg:ext1.bzl");
  }

  @Test
  public void testLoadRelativeLabel() throws Exception {
    scratch.file("pkg/BUILD");
    scratch.file("pkg/ext1.bzl", "a = 1");
    scratch.file("pkg/ext2.bzl", "load(':ext1.bzl', 'a')");
    checkSuccessfulLookup("//pkg:ext2.bzl");
  }

  @Test
  public void testLoadAbsoluteLabel() throws Exception {
    scratch.file("pkg2/BUILD");
    scratch.file("pkg3/BUILD");
    scratch.file("pkg2/ext.bzl", "b = 1");
    scratch.file("pkg3/ext.bzl", "load('//pkg2:ext.bzl', 'b')");
    checkSuccessfulLookup("//pkg3:ext.bzl");
  }

  @Test
  public void testLoadFromSameAbsoluteLabelTwice() throws Exception {
    scratch.file("pkg1/BUILD");
    scratch.file("pkg2/BUILD");
    scratch.file("pkg1/ext.bzl", "a = 1", "b = 2");
    scratch.file("pkg2/ext.bzl", "load('//pkg1:ext.bzl', 'a')", "load('//pkg1:ext.bzl', 'b')");
    checkSuccessfulLookup("//pkg2:ext.bzl");
  }

  @Test
  public void testLoadFromSameRelativeLabelTwice() throws Exception {
    scratch.file("pkg/BUILD");
    scratch.file("pkg/ext1.bzl", "a = 1", "b = 2");
    scratch.file("pkg/ext2.bzl", "load(':ext1.bzl', 'a')", "load(':ext1.bzl', 'b')");
    checkSuccessfulLookup("//pkg:ext2.bzl");
  }

  @Test
  public void testLoadFromRelativeLabelInSubdir() throws Exception {
    scratch.file("pkg/BUILD");
    scratch.file("pkg/subdir/ext1.bzl", "a = 1");
    scratch.file("pkg/subdir/ext2.bzl", "load(':subdir/ext1.bzl', 'a')");
    checkSuccessfulLookup("//pkg:subdir/ext2.bzl");
  }

  private EvaluationResult<SkylarkImportLookupValue> get(SkyKey skylarkImportLookupKey)
      throws Exception {
    EvaluationResult<SkylarkImportLookupValue> result =
        SkyframeExecutorTestUtils.evaluate(
            getSkyframeExecutor(), skylarkImportLookupKey, /*keepGoing=*/ false, reporter);
    if (result.hasError()) {
      fail(result.getError(skylarkImportLookupKey).getException().getMessage());
    }
    return result;
  }

  private SkyKey key(String label) {
    return SkylarkImportLookupValue.key(Label.parseAbsoluteUnchecked(label), false);
  }

  // Ensures that a Skylark file has been successfully processed by checking that the
  // the label in its dependency set corresponds to the requested label.
  private void checkSuccessfulLookup(String label) throws Exception {
    SkyKey skylarkImportLookupKey = key(label);
    EvaluationResult<SkylarkImportLookupValue> result = get(skylarkImportLookupKey);
    assertThat(label)
        .isEqualTo(result.get(skylarkImportLookupKey).getDependency().getLabel().toString());
  }

  @Test
  public void testSkylarkImportLookupNoBuildFile() throws Exception {
    scratch.file("pkg/ext.bzl", "");
    SkyKey skylarkImportLookupKey =
        SkylarkImportLookupValue.key(Label.parseAbsoluteUnchecked("//pkg:ext.bzl"), false);
    EvaluationResult<SkylarkImportLookupValue> result =
        SkyframeExecutorTestUtils.evaluate(
            getSkyframeExecutor(), skylarkImportLookupKey, /*keepGoing=*/ false, reporter);
    assertThat(result.hasError()).isTrue();
    ErrorInfo errorInfo = result.getError(skylarkImportLookupKey);
    String errorMessage = errorInfo.getException().getMessage();
    assertThat(errorMessage)
        .isEqualTo(
            "Extension file not found. Unable to load package for '//pkg:ext.bzl': "
                + "BUILD file not found on package path");
  }

  @Test
  public void testSkylarkImportLookupNoBuildFileForLoad() throws Exception {
    scratch.file("pkg2/BUILD");
    scratch.file("pkg1/ext.bzl", "a = 1");
    scratch.file("pkg2/ext.bzl", "load('//pkg1:ext.bzl', 'a')");
    SkyKey skylarkImportLookupKey =
        SkylarkImportLookupValue.key(Label.parseAbsoluteUnchecked("//pkg:ext.bzl"), false);
    EvaluationResult<SkylarkImportLookupValue> result =
        SkyframeExecutorTestUtils.evaluate(
            getSkyframeExecutor(), skylarkImportLookupKey, /*keepGoing=*/ false, reporter);
    assertThat(result.hasError()).isTrue();
    ErrorInfo errorInfo = result.getError(skylarkImportLookupKey);
    String errorMessage = errorInfo.getException().getMessage();
    assertThat(errorMessage)
        .isEqualTo(
            "Extension file not found. Unable to load package for '//pkg:ext.bzl': "
                + "BUILD file not found on package path");
  }

  @Test
  public void testSkylarkImportFilenameWithControlChars() throws Exception {
    scratch.file("pkg/BUILD", "");
    scratch.file("pkg/ext.bzl", "load('//pkg:oops\u0000.bzl', 'a')");
    try {
      SkyKey skylarkImportLookupKey =
          SkylarkImportLookupValue.key(Label.parseAbsoluteUnchecked("//pkg:ext.bzl"), false);
      EvaluationResult<SkylarkImportLookupValue> result =
          SkyframeExecutorTestUtils.evaluate(
              getSkyframeExecutor(), skylarkImportLookupKey, /*keepGoing=*/ false, reporter);
      fail("Expected exception");
    } catch (AssertionError e) {
      String errorMessage = e.getMessage();
      assertThat(errorMessage)
          .contains(
              "invalid target name 'oops<?>.bzl': "
              + "target names may not contain non-printable characters: '\\x00'");
    }
  }

  @Test
  public void testLoadFromExternalRepoInWorkspaceFileAllowed() throws Exception {
    scratch.deleteFile(preludeLabelRelativePath);
    scratch.overwriteFile("WORKSPACE",
        "local_repository(",
        "    name = 'a_remote_repo',",
        "    path = '/a_remote_repo'",
        ")");
    scratch.file("/a_remote_repo/WORKSPACE");
    scratch.file("/a_remote_repo/remote_pkg/BUILD");
    scratch.file("/a_remote_repo/remote_pkg/ext.bzl",
        "CONST = 17");

    SkyKey skylarkImportLookupKey =
        SkylarkImportLookupValue.key(Label.parseAbsoluteUnchecked(
            "@a_remote_repo//remote_pkg:ext.bzl"), /*inWorkspace=*/ true);
    EvaluationResult<SkylarkImportLookupValue> result =
        SkyframeExecutorTestUtils.evaluate(
            getSkyframeExecutor(), skylarkImportLookupKey, /*keepGoing=*/ false, reporter);

    assertThat(result.hasError()).isFalse();
  }

  @Test
  public void testLoadUsingLabelThatDoesntCrossesBoundaryOfPackage()
      throws Exception {
    scratch.file("a/BUILD");
    scratch.file("a/a.bzl", "load('//a:b/b.bzl', 'b')");
    scratch.file("a/b/b.bzl", "b = 42");

    checkSuccessfulLookup("//a:a.bzl");
  }

  @Test
  public void testLoadUsingLabelThatCrossesBoundaryOfPackage_Allow_OfSamePkg()
      throws Exception {
    scratch.file("a/BUILD");
    scratch.file("a/a.bzl", "load('//a:b/b.bzl', 'b')");
    scratch.file("a/b/BUILD", "");
    scratch.file("a/b/b.bzl", "b = 42");

    checkSuccessfulLookup("//a:a.bzl");
  }

  @Test
  public void testLoadUsingLabelThatCrossesBoundaryOfPackage_Disallow_OfSamePkg()
      throws Exception {
    setSkylarkSemanticsOptions("--incompatible_disallow_load_labels_to_cross_package_boundaries");

    scratch.file("a/BUILD");
    scratch.file("a/a.bzl", "load('//a:b/b.bzl', 'b')");
    scratch.file("a/b/BUILD", "");
    scratch.file("a/b/b.bzl", "b = 42");

    SkyKey skylarkImportLookupKey =
        SkylarkImportLookupValue.key(Label.parseAbsoluteUnchecked("//a:a.bzl"), false);
    EvaluationResult<SkylarkImportLookupValue> result =
        SkyframeExecutorTestUtils.evaluate(
            getSkyframeExecutor(), skylarkImportLookupKey, /*keepGoing=*/ false, reporter);
    assertThat(result.hasError()).isTrue();
    assertThatEvaluationResult(result)
        .hasErrorEntryForKeyThat(skylarkImportLookupKey)
        .hasExceptionThat()
        .isInstanceOf(SkylarkImportFailedException.class);
    assertThatEvaluationResult(result)
        .hasErrorEntryForKeyThat(skylarkImportLookupKey)
        .hasExceptionThat()
        .hasMessageThat()
        .contains(
            "Label '//a:b/b.bzl' crosses boundary of subpackage 'a/b' (perhaps you meant to put "
                + "the colon here: '//a/b:b.bzl'?)");
  }

  @Test
  public void testLoadUsingLabelThatCrossesBoundaryOfPackage_Allow_OfDifferentPkgUnder()
      throws Exception {
    scratch.file("a/BUILD");
    scratch.file("a/a.bzl", "load('//a/b:c/c.bzl', 'c')");
    scratch.file("a/b/BUILD", "");
    scratch.file("a/b/c/BUILD", "");
    scratch.file("a/b/c/c.bzl", "c = 42");

    checkSuccessfulLookup("//a:a.bzl");
  }

  @Test
  public void testLoadUsingLabelThatCrossesBoundaryOfPackage_Disallow_OfDifferentPkgUnder()
      throws Exception {
    setSkylarkSemanticsOptions("--incompatible_disallow_load_labels_to_cross_package_boundaries");

    scratch.file("a/BUILD");
    scratch.file("a/a.bzl", "load('//a/b:c/c.bzl', 'c')");
    scratch.file("a/b/BUILD", "");
    scratch.file("a/b/c/BUILD", "");
    scratch.file("a/b/c/c.bzl", "c = 42");

    SkyKey skylarkImportLookupKey =
        SkylarkImportLookupValue.key(Label.parseAbsoluteUnchecked("//a:a.bzl"), false);
    EvaluationResult<SkylarkImportLookupValue> result =
        SkyframeExecutorTestUtils.evaluate(
            getSkyframeExecutor(), skylarkImportLookupKey, /*keepGoing=*/ false, reporter);
    assertThat(result.hasError()).isTrue();
    assertThatEvaluationResult(result)
        .hasErrorEntryForKeyThat(skylarkImportLookupKey)
        .hasExceptionThat()
        .isInstanceOf(SkylarkImportFailedException.class);
    assertThatEvaluationResult(result)
        .hasErrorEntryForKeyThat(skylarkImportLookupKey)
        .hasExceptionThat()
        .hasMessageThat()
        .contains(
            "Label '//a/b:c/c.bzl' crosses boundary of subpackage 'a/b/c' (perhaps you meant to "
                + "put the colon here: '//a/b/c:c.bzl'?)");
  }

  @Test
  public void testLoadUsingLabelThatCrossesBoundaryOfPackage_Allow_OfDifferentPkgAbove()
      throws Exception {
    scratch.file("a/b/BUILD");
    scratch.file("a/b/b.bzl", "load('//a/c:c/c.bzl', 'c')");
    scratch.file("a/BUILD");
    scratch.file("a/c/c/c.bzl", "c = 42");

    // With the default of
    // --incompatible_disallow_load_labels_to_cross_subpackage_boundaries=false,
    // SkylarkImportLookupValue(//a/b:b.bzl) has an error because ASTFileLookupValue(//a/c:c/c.bzl)
    // because package //a/c doesn't exist. The behavior with
    // --incompatible_disallow_load_labels_to_cross_subpackage_boundaries=true is stricter, but we
    // still have an explicit test for this case so that way we don't forget to think about it when
    // we address the TODO in ASTFuleLookupFunction.

    SkyKey skylarkImportLookupKey =
        SkylarkImportLookupValue.key(Label.parseAbsoluteUnchecked("//a/b:b.bzl"), false);
    EvaluationResult<SkylarkImportLookupValue> result =
        SkyframeExecutorTestUtils.evaluate(
            getSkyframeExecutor(), skylarkImportLookupKey, /*keepGoing=*/ false, reporter);
    assertThat(result.hasError()).isTrue();
    assertThatEvaluationResult(result)
        .hasErrorEntryForKeyThat(skylarkImportLookupKey)
        .hasExceptionThat()
        .isInstanceOf(SkylarkImportFailedException.class);
    assertThatEvaluationResult(result)
        .hasErrorEntryForKeyThat(skylarkImportLookupKey)
        .hasExceptionThat()
        .hasMessageThat()
        .contains(
            "Extension file not found. Unable to load package for '//a/c:c/c.bzl': BUILD file not "
                + "found on package path");
  }

  @Test
  public void testLoadUsingLabelThatCrossesBoundaryOfPackage_Disallow_OfDifferentPkgAbove()
      throws Exception {
    setSkylarkSemanticsOptions("--incompatible_disallow_load_labels_to_cross_package_boundaries");

    scratch.file("a/b/BUILD");
    scratch.file("a/b/b.bzl", "load('//a/c:c/c.bzl', 'c')");
    scratch.file("a/BUILD");
    scratch.file("a/c/c/c.bzl", "c = 42");

    SkyKey skylarkImportLookupKey =
        SkylarkImportLookupValue.key(Label.parseAbsoluteUnchecked("//a/b:b.bzl"), false);
    EvaluationResult<SkylarkImportLookupValue> result =
        SkyframeExecutorTestUtils.evaluate(
            getSkyframeExecutor(), skylarkImportLookupKey, /*keepGoing=*/ false, reporter);
    assertThat(result.hasError()).isTrue();
    assertThatEvaluationResult(result)
        .hasErrorEntryForKeyThat(skylarkImportLookupKey)
        .hasExceptionThat()
        .isInstanceOf(SkylarkImportFailedException.class);
    assertThatEvaluationResult(result)
        .hasErrorEntryForKeyThat(skylarkImportLookupKey)
        .hasExceptionThat()
        .hasMessageThat()
        .contains(
            "Label '//a/c:c/c.bzl' crosses boundary of package 'a' (perhaps you meant to put the "
                + "colon here: '//a:c/c/c.bzl'?)");
  }
}

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
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.fail;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.clock.BlazeClock;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.packages.ConstantRuleVisibility;
import com.google.devtools.build.lib.packages.StarlarkSemanticsOptions;
import com.google.devtools.build.lib.pkgcache.PackageCacheOptions;
import com.google.devtools.build.lib.pkgcache.PathPackageLocator;
import com.google.devtools.build.lib.skyframe.SkylarkImportLookupFunction.SkylarkImportFailedException;
import com.google.devtools.build.lib.skyframe.util.SkyframeExecutorTestUtils;
import com.google.devtools.build.lib.util.io.TimestampGranularityMonitor;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Root;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.skyframe.ErrorInfo;
import com.google.devtools.build.skyframe.EvaluationResult;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.common.options.Options;
import java.util.UUID;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for SkylarkImportLookupFunction. */
@RunWith(JUnit4.class)
public class SkylarkImportLookupFunctionTest extends BuildViewTestCase {

  @Before
  public final void preparePackageLoading() throws Exception {
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
            Options.getDefaults(StarlarkSemanticsOptions.class),
            UUID.randomUUID(),
            ImmutableMap.<String, String>of(),
            new TimestampGranularityMonitor(BlazeClock.instance()));
    skyframeExecutor.setActionEnv(ImmutableMap.<String, String>of());
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
    scratch.overwriteFile(
        "WORKSPACE",
        "local_repository(",
        "    name = 'a_remote_repo',",
        "    path = '/a_remote_repo'",
        ")");
    scratch.file("/a_remote_repo/WORKSPACE");
    scratch.file("/a_remote_repo/remote_pkg/BUILD");
    scratch.file("/a_remote_repo/remote_pkg/ext1.bzl", "load(':ext2.bzl', 'CONST')");
    scratch.file("/a_remote_repo/remote_pkg/ext2.bzl", "CONST = 17");
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

  private static SkyKey key(String label) {
    return SkylarkImportLookupValue.key(Label.parseAbsoluteUnchecked(label));
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
    SkyKey skylarkImportLookupKey = key("//pkg:ext.bzl");
    EvaluationResult<SkylarkImportLookupValue> result =
        SkyframeExecutorTestUtils.evaluate(
            getSkyframeExecutor(), skylarkImportLookupKey, /*keepGoing=*/ false, reporter);
    assertThat(result.hasError()).isTrue();
    ErrorInfo errorInfo = result.getError(skylarkImportLookupKey);
    String errorMessage = errorInfo.getException().getMessage();
    assertThat(errorMessage)
        .contains(
            "Every .bzl file must have a corresponding package, but '//pkg:ext.bzl' does not");
  }

  @Test
  public void testSkylarkImportLookupNoBuildFileForLoad() throws Exception {
    scratch.file("pkg2/BUILD");
    scratch.file("pkg1/ext.bzl", "a = 1");
    scratch.file("pkg2/ext.bzl", "load('//pkg1:ext.bzl', 'a')");
    SkyKey skylarkImportLookupKey = key("//pkg:ext.bzl");
    EvaluationResult<SkylarkImportLookupValue> result =
        SkyframeExecutorTestUtils.evaluate(
            getSkyframeExecutor(), skylarkImportLookupKey, /*keepGoing=*/ false, reporter);
    assertThat(result.hasError()).isTrue();
    ErrorInfo errorInfo = result.getError(skylarkImportLookupKey);
    String errorMessage = errorInfo.getException().getMessage();
    assertThat(errorMessage).contains("Every .bzl file must have a corresponding package");
  }

  @Test
  public void testSkylarkImportFilenameWithControlChars() throws Exception {
    scratch.file("pkg/BUILD", "");
    scratch.file("pkg/ext.bzl", "load('//pkg:oops\u0000.bzl', 'a')");
    SkyKey skylarkImportLookupKey = key("//pkg:ext.bzl");
    AssertionError e =
        assertThrows(
            AssertionError.class,
            () ->
                SkyframeExecutorTestUtils.evaluate(
                    getSkyframeExecutor(), skylarkImportLookupKey, /*keepGoing=*/ false, reporter));
    String errorMessage = e.getMessage();
    assertThat(errorMessage)
        .contains(
            "invalid target name 'oops<?>.bzl': "
                + "target names may not contain non-printable characters: '\\x00'");
  }

  @Test
  public void testLoadFromExternalRepoInWorkspaceFileAllowed() throws Exception {
    Path p =
        scratch.overwriteFile(
            "WORKSPACE",
            "local_repository(",
            "    name = 'a_remote_repo',",
            "    path = '/a_remote_repo'",
            ")");
    scratch.file("/a_remote_repo/WORKSPACE");
    scratch.file("/a_remote_repo/remote_pkg/BUILD");
    scratch.file("/a_remote_repo/remote_pkg/ext.bzl", "CONST = 17");

    RootedPath rootedPath =
        RootedPath.toRootedPath(
            Root.fromPath(p.getParentDirectory()), PathFragment.create("WORKSPACE"));

    SkyKey skylarkImportLookupKey =
        SkylarkImportLookupValue.keyInWorkspace(
            Label.parseAbsoluteUnchecked("@a_remote_repo//remote_pkg:ext.bzl"),
            /* inWorkspace= */
            /* workspaceChunk= */ 0,
            rootedPath);
    EvaluationResult<SkylarkImportLookupValue> result =
        SkyframeExecutorTestUtils.evaluate(
            getSkyframeExecutor(), skylarkImportLookupKey, /*keepGoing=*/ false, reporter);

    assertThat(result.hasError()).isFalse();
  }

  @Test
  public void testLoadUsingLabelThatDoesntCrossBoundaryOfPackage() throws Exception {
    scratch.file("a/BUILD");
    scratch.file("a/a.bzl", "load('//a:b/b.bzl', 'b')");
    scratch.file("a/b/b.bzl", "b = 42");

    checkSuccessfulLookup("//a:a.bzl");
  }

  @Test
  public void testLoadUsingLabelThatCrossesBoundaryOfPackage_Disallow_OfSamePkg() throws Exception {
    scratch.file("a/BUILD");
    scratch.file("a/a.bzl", "load('//a:b/b.bzl', 'b')");
    scratch.file("a/b/BUILD", "");
    scratch.file("a/b/b.bzl", "b = 42");
    checkStrayLabel(
        "//a:a.bzl",
        "Label '//a:b/b.bzl' is invalid because 'a/b' is a subpackage; perhaps you meant to"
            + " put the colon here: '//a/b:b.bzl'?");
  }

  @Test
  public void testLoadUsingLabelThatCrossesBoundaryOfPackage_Disallow_OfSamePkg_Relative()
      throws Exception {
    scratch.file("a/BUILD");
    scratch.file("a/a.bzl", "load('b/b.bzl', 'b')");
    scratch.file("a/b/BUILD", "");
    scratch.file("a/b/b.bzl", "b = 42");
    checkStrayLabel(
        "//a:a.bzl",
        "Label '//a:b/b.bzl' is invalid because 'a/b' is a subpackage; perhaps you meant to"
            + " put the colon here: '//a/b:b.bzl'?");
  }

  @Test
  public void testLoadUsingLabelThatCrossesBoundaryOfPackage_Disallow_OfDifferentPkgUnder()
      throws Exception {
    scratch.file("a/BUILD");
    scratch.file("a/a.bzl", "load('//a/b:c/c.bzl', 'c')");
    scratch.file("a/b/BUILD", "");
    scratch.file("a/b/c/BUILD", "");
    scratch.file("a/b/c/c.bzl", "c = 42");
    checkStrayLabel(
        "//a:a.bzl",
        "Label '//a/b:c/c.bzl' is invalid because 'a/b/c' is a subpackage; perhaps you meant"
            + " to put the colon here: '//a/b/c:c.bzl'?");
  }

  @Test
  public void testLoadUsingLabelThatCrossesBoundaryOfPackage_Disallow_OfDifferentPkgAbove()
      throws Exception {
    scratch.file("a/b/BUILD");
    scratch.file("a/b/b.bzl", "load('//a/c:c/c.bzl', 'c')");
    scratch.file("a/BUILD");
    scratch.file("a/c/c/c.bzl", "c = 42");
    checkStrayLabel(
        "//a/b:b.bzl",
        "Label '//a/c:c/c.bzl' is invalid because 'a/c' is not a package; perhaps you meant to "
            + "put the colon here: '//a:c/c/c.bzl'?");
  }

  // checkStrayLabel checks that execution of target fails because
  // the label of its load statement strays into a subpackage.
  private void checkStrayLabel(String target, String expectedMessage) throws InterruptedException {
    SkyKey skylarkImportLookupKey = key(target);
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
        .contains(expectedMessage);
  }

  @Test
  public void testWithNonExistentRepository_And_DisallowLoadUsingLabelThatCrossesBoundaryOfPackage()
      throws Exception {
    scratch.file("BUILD", "load(\"@repository//dir:file.bzl\", \"foo\")");

    SkyKey skylarkImportLookupKey = key("@repository//dir:file.bzl");
    EvaluationResult<com.google.devtools.build.lib.skyframe.SkylarkImportLookupValue> result =
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
            "Unable to find package for @repository//dir:file.bzl: The repository '@repository' "
                + "could not be resolved.");
  }

  @Test
  public void testLoadBzlFileFromWorkspaceWithRemapping() throws Exception {
    Path p =
        scratch.overwriteFile(
            "WORKSPACE",
            "local_repository(",
            "    name = 'y',",
            "    path = '/y'",
            ")",
            "local_repository(",
            "    name = 'a',",
            "    path = '/a',",
            "    repo_mapping = {'@x' : '@y'}",
            ")",
            "load('@a//:a.bzl', 'a_symbol')");

    scratch.file("/y/WORKSPACE");
    scratch.file("/y/BUILD");
    scratch.file("/y/y.bzl", "y_symbol = 5");

    scratch.file("/a/WORKSPACE");
    scratch.file("/a/BUILD");
    scratch.file("/a/a.bzl", "load('@x//:y.bzl', 'y_symbol')", "a_symbol = y_symbol");

    Root root = Root.fromPath(p.getParentDirectory());
    RootedPath rootedPath = RootedPath.toRootedPath(root, PathFragment.create("WORKSPACE"));

    SkyKey skylarkImportLookupKey =
        SkylarkImportLookupValue.keyInWorkspace(
            Label.parseAbsoluteUnchecked("@a//:a.bzl"), 1, rootedPath);

    EvaluationResult<SkylarkImportLookupValue> result =
        SkyframeExecutorTestUtils.evaluate(
            getSkyframeExecutor(), skylarkImportLookupKey, /*keepGoing=*/ false, reporter);

    assertThat(result.get(skylarkImportLookupKey).getEnvironmentExtension().getBindings())
        .containsEntry("a_symbol", 5);
  }
}
